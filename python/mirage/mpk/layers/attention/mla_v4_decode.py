"""DeepSeek V4-Flash MLA decode catalog module (v1: SWA-only).

Backed by ``tasks/blackwell/mla_v4_decode_sm100.cuh`` (task name
``"mla_v4_decode_sm100"``).

V4-Flash MLA decode reads two caches and applies an optional sparse
top-K indexing scheme. For v1 we support **only** the simplest path:

  * ``compress_ratio == 0`` — single SWA cache.
  * No compressed (``extra_k_cache``) cache.
  * No sparse top-K indices.

The dual-cache + sparse paths are documented here as v2 follow-ups and
the ``compile`` keyword-only API reserves the slots; v1 raises
:class:`NotImplementedError` when they are non-None.

Math (single decode step at position ``pos``):

    attn_logits[h, p] = (q_rope[h] @ kv_rope[p].T
                        + q_nope[h] @ kv_nope[p].T) * softmax_scale
    attn_weights[h, p] = softmax_over_p(attn_logits + attn_sink_h)
    out[h, d] = sum_p(attn_weights[h, p] * kv[p, d])      # MLA: K == V

The rope/nope split is conceptual — both halves contribute to a single
dot product over the full ``head_dim``. The kernel does not split the
heads internally in v1; the spec reserves the option to do so for v2.

Reference call sites:

  * vLLM   ``deps/vllm/vllm/model_executor/layers/deepseek_v4_attention.py:855``
    ``flash_mla_with_kvcache(...)``.
  * Official ``deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py``
    ``Attention.forward`` decode branch.
"""
from __future__ import annotations

import struct
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["MLAv4Decode"]


class MLAv4Decode(MPKModule):
    """V4-Flash MLA decode (v1 scope: ``compress_ratio == 0``).

    Constructor args:
      num_heads        : per-token Q head count (e.g. 64 in V4-Flash).
      head_dim         : total per-head Q/KV width (e.g. 512 in V4-Flash;
                          ``nope_dim + rope_dim``).
      qk_rope_head_dim : trailing rope-applied portion of ``head_dim``
                          (kept for v2 — v1 does not split internally).
      softmax_scale    : scalar logit scale (typically ``1/sqrt(head_dim)``
                          or YARN-adjusted).
      prefix           : ``state_dict`` key prefix.

    State:
      attn_sink (nn.Parameter) : ``[num_heads]`` fp32. Persistent
                                   per-head additive logit sink (one
                                   extra softmax lane with no KV row).
                                   Reference: ``model.py:456``.

    v2 follow-ups (reserved):
      * compressed_cache (``extra_k_cache``) for ``compress_ratio > 0``.
      * topk_indices for sparse attention (ratio 4 / 128).
      * paged-KV ring-buffer indexing via the MPK runtime indptr buffer.
        v1 assumes the caller stages the SWA cache as a contiguous
        ``[num_pages, page_size, head_dim]`` (or ``[swa_total, head_dim]``)
        slab where the linear position index equals the absolute
        position. Tests use the 2-D form for simplicity.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        qk_rope_head_dim: int,
        softmax_scale: float,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0; got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0; got {head_dim}")
        if qk_rope_head_dim < 0 or qk_rope_head_dim > head_dim:
            raise ValueError(
                "qk_rope_head_dim must be in [0, head_dim]; got "
                f"{qk_rope_head_dim} with head_dim={head_dim}"
            )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.softmax_scale = float(softmax_scale)
        # Per-head additive logit sink (fp32 in the official model).
        self.attn_sink = nn.Parameter(
            torch.zeros(num_heads, dtype=torch.float32)
        )

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,
        swa_cache_view: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """v1 reference: SWA-only decode (single-row MLA: K == V).

        Args:
          q              : ``[T, num_heads, head_dim]`` bf16. Already
                            normalised + rotary-applied per-head Q.
          swa_cache_view : ``[swa_total, head_dim]`` bf16 (logical
                            linearised cache). For each token at position
                            ``pos`` we attend over rows
                            ``swa_cache_view[0:pos]``.
          positions      : ``[T]`` int32/int64 absolute positions.

        Returns ``o`` ``[T, num_heads, head_dim]`` bf16.
        """
        if q.dim() != 3 or q.size(1) != self.num_heads or q.size(
            2
        ) != self.head_dim:
            raise ValueError(
                "q must be (T, num_heads, head_dim); got "
                f"{tuple(q.shape)}, expected "
                f"(*, {self.num_heads}, {self.head_dim})"
            )
        if swa_cache_view.dim() == 3:
            # Flatten [num_pages, page_size, head_dim] -> [total, head_dim].
            swa_cache_view = swa_cache_view.reshape(
                -1, swa_cache_view.size(-1)
            )
        if swa_cache_view.dim() != 2 or swa_cache_view.size(-1) != self.head_dim:
            raise ValueError(
                "swa_cache_view must be (total, head_dim); got "
                f"{tuple(swa_cache_view.shape)}"
            )
        if positions.dim() != 1 or positions.size(0) != q.size(0):
            raise ValueError(
                "positions must be 1-D with the same leading dim as q; "
                f"got {tuple(positions.shape)} vs q {tuple(q.shape)}"
            )

        T = q.size(0)
        H = self.num_heads
        D = self.head_dim
        # FP32 reference math, cast back to bf16 at the end.
        q_f = q.to(torch.float32)
        kv_f = swa_cache_view.to(torch.float32)
        sink = self.attn_sink.to(torch.float32)
        out = torch.zeros(T, H, D, dtype=torch.float32, device=q.device)
        for t in range(T):
            pos = int(positions[t].item())
            if pos <= 0:
                # No valid KV rows -- the only softmax lane is the sink,
                # which has no KV contribution -> output is zero.
                continue
            kv = kv_f[:pos]                                     # [pos, D]
            logits = (
                q_f[t] @ kv.transpose(-1, -2)
            ) * self.softmax_scale                              # [H, pos]
            # Append attn_sink as an extra lane (no KV row).
            sink_lane = sink.unsqueeze(-1).expand(H, 1)          # [H, 1]
            full = torch.cat([logits, sink_lane], dim=-1)        # [H, pos+1]
            w = full.softmax(dim=-1)                             # [H, pos+1]
            w_kv = w[:, :pos]                                    # [H, pos]
            out[t] = w_kv @ kv                                   # [H, D]
        return out.to(q.dtype)

    # ------------------------------------------------------------------
    # Grid / block heuristics
    # ------------------------------------------------------------------
    def auto_grid_dim(self, q_dt: DTensor) -> GridDim:
        """One CTA per token; CTAs derive ``t`` from ``token_offset``."""
        from ... import context as _ctx
        pk = _ctx.current_pk()
        n = q_dt.dim(0)
        return (max(1, min(n, pk.num_workers)), 1, 1)

    def default_block_dim(self) -> BlockDim:
        from ... import context as _ctx
        pk = _ctx.current_pk()
        return (128, 1, 1) if pk.target_cc < 90 else (256, 1, 1)

    # ------------------------------------------------------------------
    # MPK task registration
    # ------------------------------------------------------------------
    def compile(
        self,
        q: DTensor,
        swa_cache: Union[torch.Tensor, DTensor],
        positions: Union[torch.Tensor, DTensor],
        *,
        attn_sink_override: Optional[Union[torch.Tensor, DTensor]] = None,
        compressed_cache: Optional[Any] = None,
        topk_indices: Optional[Any] = None,
        o: Optional[Union[torch.Tensor, DTensor]] = None,
        sliding_window: Optional[int] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``mla_v4_decode_sm100`` task.

        Tensor contract:
          q          : ``[T, num_heads, head_dim]``       bf16, input.
          swa_cache  : ``[swa_total, head_dim]`` OR
                        ``[num_pages, page_size, head_dim]`` bf16
                        (flattened internally to 2-D).
          positions  : ``[T]``                            int32.
          attn_sink  : ``[num_heads]``                    fp32 (auto-
                        attached from ``self.attn_sink``; override via
                        ``attn_sink_override`` if needed).
          o          : ``[T, num_heads, head_dim]``       bf16 (auto-
                        allocated if None).

        v1 scope: ``compressed_cache`` and ``topk_indices`` MUST be None
        — they are reserved API surface for v2 (dual-cache decode +
        sparse top-K). Raises :class:`NotImplementedError` otherwise.
        """
        if compressed_cache is not None:
            raise NotImplementedError(
                "MLAv4Decode v1 supports only compress_ratio=0 "
                "(SWA-only). Pass compressed_cache=None; v2 will add "
                "the dual-cache decode path."
            )
        if topk_indices is not None:
            raise NotImplementedError(
                "MLAv4Decode v1 does not support sparse top-K. Pass "
                "topk_indices=None; v2 will add the Indexer/precomputed "
                "topk routing."
            )

        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        prefix = self.prefix or "mla_v4_decode_"

        # Validate q.
        if q.num_dims != 3:
            raise ValueError(
                "q must be a 3-D DTensor (T, num_heads, head_dim); got "
                f"num_dims={q.num_dims}"
            )
        if q.dim(1) != self.num_heads:
            raise ValueError(
                f"q.dim(1)={q.dim(1)} != num_heads={self.num_heads}"
            )
        if q.dim(2) != self.head_dim:
            raise ValueError(
                f"q.dim(2)={q.dim(2)} != head_dim={self.head_dim}"
            )
        T = q.dim(0)

        # Resolve auxiliary inputs.
        def _attach_in(
            buf: Union[torch.Tensor, DTensor],
            expected_dtype_torch: torch.dtype,
            name: str,
        ) -> DTensor:
            if isinstance(buf, DTensor):
                return buf
            if isinstance(buf, torch.Tensor):
                if buf.dtype != expected_dtype_torch:
                    raise ValueError(
                        f"{name} must have dtype {expected_dtype_torch}; "
                        f"got {buf.dtype}"
                    )
                return pk.attach_input(buf, name=name)
            raise TypeError(
                f"{name} must be torch.Tensor or DTensor; got "
                f"{type(buf).__name__}"
            )

        # Flatten 3-D paged-cache input to 2-D before attaching so the
        # codegen's dimension assertions match (HEAD_DIM is the trailing
        # axis either way).
        if isinstance(swa_cache, torch.Tensor) and swa_cache.dim() == 3:
            swa_cache = swa_cache.reshape(-1, swa_cache.size(-1))
        swa_cache_dt = _attach_in(
            swa_cache, torch.bfloat16, f"{prefix}swa_cache"
        )
        if swa_cache_dt.dim(swa_cache_dt.num_dims - 1) != self.head_dim:
            raise ValueError(
                "swa_cache trailing dim "
                f"{swa_cache_dt.dim(swa_cache_dt.num_dims - 1)} != "
                f"head_dim={self.head_dim}"
            )

        positions_dt = _attach_in(
            positions, torch.int32, f"{prefix}positions"
        )
        if positions_dt.num_dims != 1 or positions_dt.dim(0) != T:
            raise ValueError(
                "positions must be 1-D with length T; got "
                f"num_dims={positions_dt.num_dims}, len="
                f"{positions_dt.dim(0)} vs T={T}"
            )

        sink_buf = (
            attn_sink_override
            if attn_sink_override is not None
            else self.attn_sink.data
        )
        if isinstance(sink_buf, torch.Tensor):
            if sink_buf.dtype != torch.float32:
                raise ValueError(
                    "attn_sink must be fp32; got "
                    f"{sink_buf.dtype}"
                )
            sink_dt = pk.attach_input(
                sink_buf, name=f"{prefix}attn_sink"
            )
        elif isinstance(sink_buf, DTensor):
            sink_dt = sink_buf
        else:
            raise TypeError(
                "attn_sink_override must be torch.Tensor or DTensor"
            )
        if sink_dt.num_dims != 1 or sink_dt.dim(0) != self.num_heads:
            raise ValueError(
                f"attn_sink must be [num_heads={self.num_heads}]; got "
                f"shape with dim(0)={sink_dt.dim(0)}"
            )

        # Output.
        def _attach_out(buf, default_dims, name):
            if buf is None:
                return pk.new_tensor(
                    dims=default_dims, dtype=mi.bfloat16, name=name
                )
            if isinstance(buf, torch.Tensor):
                if buf.dtype != torch.bfloat16:
                    raise ValueError(
                        f"{name} must have dtype bfloat16; got {buf.dtype}"
                    )
                return pk.attach_input(buf, name=name)
            if isinstance(buf, DTensor):
                return buf
            raise TypeError(
                f"{name} must be None, torch.Tensor, or DTensor; got "
                f"{type(buf).__name__}"
            )

        o_dt = _attach_out(
            o, (T, self.num_heads, self.head_dim), f"{prefix}o"
        )

        # Default sliding_window: use the full cache (kernel will then
        # attend [0, pos)).
        if sliding_window is None:
            # swa_cache_dt has been normalised to 2-D above (after the
            # 3-D flatten); fall through to grab the trailing-leading
            # dim either way.
            if swa_cache_dt.num_dims == 2:
                sliding_window = swa_cache_dt.dim(0)
            else:
                sliding_window = (
                    swa_cache_dt.dim(0) * swa_cache_dt.dim(1)
                )
        # Pack softmax_scale as int bits so the std::vector<int> param
        # carries the fp32 payload (decoded via memcpy in task_register.cc).
        scale_bits = struct.unpack(
            "i", struct.pack("f", float(self.softmax_scale))
        )[0]
        params = [
            int(sliding_window),
            int(self.qk_rope_head_dim),
            int(scale_bits),
        ]

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(q)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # All inputs/outputs addressed via task_metadata.token_offset
        # (the kernel does its own row arithmetic).
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q, (-1, -1, -1), -1, True)
        tb_graph.new_input(swa_cache_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(positions_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(sink_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(o_dt, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [q, swa_cache_dt, positions_dt, sink_dt, o_dt],
            tb_graph,
        )
        pk.kn_graph.register_task(
            tb_graph, "mla_v4_decode_sm100", params
        )
        return o_dt
