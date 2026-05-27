"""DeepSeek V4-Flash MLA prefill KV gather (paged-to-contiguous workspace).

Backed by ``tasks/blackwell/mla_v4_prefill_gather_sm100.cuh`` (task name
``"mla_v4_prefill_gather_sm100"``).

This is the **gather** sibling of the ``mla_v4_prefill_sm100`` MLA prefill
kernel. The prefill kernel consumes a contiguous ``[T_kv, head_dim]`` bf16
buffer — this module is responsible for materializing that buffer from the
paged SWA cache + MPK runtime meta-tensors (``paged_kv_indptr_buffer``,
``paged_kv_indices_buffer``, ``paged_kv_last_page_len_buffer``).

v1 implements the ``compress_ratio == 0`` (SWA-only) path:

.. code-block:: python

    for kv_pos in range(T_kv):
        page_id = page_table[kv_pos // page_size]
        slot    = kv_pos %  page_size
        gathered_kv[kv_pos, :] = swa_cache[page_id, slot, :]

The optional compressed-cache + indexer-topk inputs described in
``docs/mpk/deepseek_v4/attention.md`` §4 are reserved for v2.

The module carries no ``nn.Parameter`` — it is a pure layout transform
keyed on runtime meta-tensors.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["MLAv4PrefillGather"]


class MLAv4PrefillGather(MPKModule):
    """Paged-to-contiguous KV gather for V4-Flash MLA prefill.

    Constructor args:
      head_dim   : ``D`` — per-row KV width (= 512 + 64 = 576 in V4-Flash).
      page_size  : paged-cache page length. v1 default is 64, matching the
                    SWA cache page convention in V4-Flash.
      prefix     : MPK kernel-tensor name prefix.

    There are no learnable weights — the gather is keyed entirely on the
    MPK runtime ``paged_kv_*`` meta-tensors maintained by the persistent
    kernel scheduler. ``compile`` does NOT take page-table tensors; those
    live in the ``PersistentKernel.meta_tensors`` dict and are dispatched
    by the generated CUDA code at runtime.
    """

    def __init__(
        self,
        head_dim: int,
        page_size: int = 64,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if head_dim <= 0 or head_dim % 8 != 0:
            raise ValueError(
                f"head_dim={head_dim} must be a positive multiple of 8 "
                "(uint4-vectorized copy requires 8-bf16 alignment)"
            )
        if page_size <= 0:
            raise ValueError(f"page_size={page_size} must be positive")
        self.head_dim = head_dim
        self.page_size = page_size

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        swa_cache: torch.Tensor,
        page_table: torch.Tensor,
        num_kv_tokens: int,
    ) -> torch.Tensor:
        """Reference (PyTorch) implementation.

        Args:
          swa_cache    : ``[num_pages, page_size, head_dim]`` bf16.
          page_table   : ``[num_active_pages]`` int32/int64. Flat mapping
                          from block id ``= kv_pos // page_size`` to a
                          page index into ``swa_cache``.
          num_kv_tokens: ``T_kv`` — number of contiguous KV rows to emit.

        Returns:
          ``gathered_kv [num_kv_tokens, head_dim]`` bf16.
        """
        if swa_cache.dim() != 3:
            raise ValueError(
                "swa_cache must have shape [num_pages, page_size, head_dim]; "
                f"got {tuple(swa_cache.shape)}"
            )
        if swa_cache.shape[1] != self.page_size:
            raise ValueError(
                f"swa_cache.shape[1]={swa_cache.shape[1]} != "
                f"page_size={self.page_size}"
            )
        if swa_cache.shape[2] != self.head_dim:
            raise ValueError(
                f"swa_cache.shape[2]={swa_cache.shape[2]} != "
                f"head_dim={self.head_dim}"
            )
        if page_table.dim() != 1:
            raise ValueError(
                f"page_table must be 1-D; got {tuple(page_table.shape)}"
            )

        device = swa_cache.device
        pt = page_table.to(device=device, dtype=torch.long)

        # Flatten the cache to [num_pages * page_size, head_dim], then
        # index by (page_id * page_size + slot) for each kv_pos.
        flat = swa_cache.reshape(-1, self.head_dim)
        kv_pos = torch.arange(num_kv_tokens, dtype=torch.long, device=device)
        block_id = kv_pos // self.page_size
        slot = kv_pos % self.page_size
        # page_table maps block_id -> physical page index.
        phys_page = pt.index_select(0, block_id)
        flat_idx = phys_page * self.page_size + slot
        return flat.index_select(0, flat_idx).contiguous()

    # ------------------------------------------------------------------
    # Grid / block heuristics
    # ------------------------------------------------------------------
    def auto_grid_dim(self, gathered_kv_dt: DTensor) -> GridDim:
        """One CTA per KV row, capped at ``num_workers``.

        Each CTA derives its ``kv_pos`` from ``task_metadata.token_offset``.
        """
        from ... import context as _ctx
        pk = _ctx.current_pk()
        n = gathered_kv_dt.dim(0)
        return (max(1, min(n, pk.num_workers)), 1, 1)

    def default_block_dim(self) -> BlockDim:
        """128 threads/CTA — matches the V3 ``mla_kv_gather_sm100`` kernel.

        The kernel itself only uses the first 128 lanes (``NUM_THREADS=128``);
        we launch the runtime worker block dim so the surrounding
        ``__syncthreads()`` is well-defined.
        """
        from ... import context as _ctx
        pk = _ctx.current_pk()
        return (128, 1, 1) if pk.target_cc < 90 else (256, 1, 1)

    # ------------------------------------------------------------------
    # MPK task registration
    # ------------------------------------------------------------------
    def compile(
        self,
        swa_cache: DTensor,
        *,
        gathered_kv: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``mla_v4_prefill_gather_sm100`` task.

        Tensor contract:
          swa_cache   : ``[num_pages, page_size, head_dim]``  bf16, input.
          gathered_kv : ``[T_kv_max, head_dim]``               bf16, output.

        Notes:
          * ``paged_kv_indptr_buffer`` / ``paged_kv_indices_buffer`` /
            ``paged_kv_last_page_len_buffer`` are read directly from
            ``runtime_config`` inside the generated code — they are NOT
            passed as task inputs.
          * The grid is one CTA per KV row; each CTA addresses its row via
            ``task_metadata.token_offset``.
          * ``T_kv_max`` defaults to ``current_pk().max_num_batched_tokens``
            when ``gathered_kv`` is not pre-allocated.
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        prefix = self.prefix or "mla_v4_prefill_gather_"

        # Validate swa_cache shape.
        if swa_cache.num_dims != 3:
            raise ValueError(
                "MLAv4PrefillGather.compile expects a 3-D swa_cache DTensor; "
                f"got num_dims={swa_cache.num_dims}"
            )
        if swa_cache.dim(1) != self.page_size:
            raise ValueError(
                f"swa_cache.dim(1)={swa_cache.dim(1)} != "
                f"page_size={self.page_size}"
            )
        if swa_cache.dim(2) != self.head_dim:
            raise ValueError(
                f"swa_cache.dim(2)={swa_cache.dim(2)} != "
                f"head_dim={self.head_dim}"
            )

        # Resolve the output buffer.
        def _attach_out(buf, default_dims, name):
            if buf is None:
                return pk.new_tensor(
                    dims=default_dims, dtype=mi.bfloat16, name=name
                )
            if isinstance(buf, torch.Tensor):
                if buf.dtype != torch.bfloat16:
                    raise ValueError(
                        f"{name} must have dtype torch.bfloat16; got {buf.dtype}"
                    )
                return pk.attach_input(buf, name=name)
            if isinstance(buf, DTensor):
                return buf
            raise TypeError(
                f"{name} must be None, torch.Tensor, or DTensor; got "
                f"{type(buf).__name__}"
            )

        t_kv_max = pk.max_num_batched_tokens
        gathered_kv_dt = _attach_out(
            gathered_kv, (t_kv_max, self.head_dim), f"{prefix}gathered_kv"
        )
        if gathered_kv_dt.num_dims != 2:
            raise ValueError(
                "gathered_kv must be a 2-D DTensor "
                "[T_kv_max, head_dim]; got "
                f"num_dims={gathered_kv_dt.num_dims}"
            )
        if gathered_kv_dt.dim(1) != self.head_dim:
            raise ValueError(
                f"gathered_kv.dim(1)={gathered_kv_dt.dim(1)} != "
                f"head_dim={self.head_dim}"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(gathered_kv_dt)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # All tensors are addressed via ``task_metadata.token_offset`` (not
        # via TBGraph partitioning), so all map dims are (-1, -1, -1).
        # Matches the inv_rope_fp8_quant_o_sm100 convention.
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(swa_cache, (-1, -1, -1), -1, True)
        tb_graph.new_input(gathered_kv_dt, (-1, -1, -1), -1, True)
        pk.kn_graph.customized([swa_cache, gathered_kv_dt], tb_graph)
        pk.kn_graph.register_task(
            tb_graph,
            "mla_v4_prefill_gather_sm100",
            [self.head_dim, self.page_size],
        )
        return gathered_kv_dt
