"""DeepSeek V4-Flash MLA prefill over a gathered KV workspace.

Backed by ``tasks/blackwell/mla_v4_prefill_sm100.cuh`` (task name
``"mla_v4_prefill_sm100"``).

V4-Flash's prefill operates on a *gathered* KV workspace that has already
been staged by the sibling ``mla_v4_prefill_gather`` task. For
``compress_ratio == 0`` layers (v1 scope) the gathered KV is simply the
SWA cache contents up to the current sequence length; for ratio ∈ {4, 128}
the gather concatenates the compressed pool with the SWA window.

Math (standard MLA attention over a single-row K==V latent, FP32
accumulators in the kernel, FlashAttn-style online softmax)::

    logits[t, h, k]   = (q[t, h] . gathered_kv[k]) * softmax_scale
    weights[t, h, :]  = softmax( logits[t, h, :] + causal_mask + sink[h] )
    o[t, h]           = weights[t, h, :] @ gathered_kv

The causal mask trims the inner loop bound to ``min(T_kv, q_pos + 1)``
where ``q_pos == q_tile_offset + q_local`` is the absolute position of
the Q row inside the gathered workspace. ``attn_sink[h]`` is the
DeepSeek-V4 logit-sink parameter that always participates in the softmax
denominator with a value-contribution of zero.

Reference call site (vLLM):
``deps/vllm/vllm/model_executor/layers/deepseek_v4_attention.py:983-991``
``flash_mla_sparse_fwd(q=..., kv=..., indices=..., sm_scale=...,
                      attn_sink=..., topk_length=..., out=...)``
"""
from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["MLAv4Prefill"]


class MLAv4Prefill(MPKModule):
    """MLA prefill kernel over a gathered KV workspace (v1, ratio=0).

    Constructor args:
      num_heads        : ``H`` — number of attention heads (e.g. 64 in V4-Flash).
      head_dim         : ``D`` — per-head dim (= MLA latent width, e.g. 512).
      qk_rope_head_dim : ``D_rope`` — width of the RoPE tail on Q/K. Carried
                          for symmetry with the spec; in v1 the kernel treats
                          Q as already-RoPE'd so this is metadata only.
      softmax_scale    : multiplier on the QK dot before softmax. Typically
                          ``1 / sqrt(head_dim)`` (= ``1/sqrt(512)`` in V4).
      prefix           : MPK kernel-tensor name prefix.

    State:
      attn_sink (nn.Parameter): ``[num_heads]`` fp32. Initialized to 0.

    Constraints:
      * bf16-only Q / gathered_kv / O in v1.
      * Kernel grid is ``(T_q, 1, 1)`` with one CTA per Q row.
      * ``head_dim`` must be a multiple of ``NUM_THREADS`` (auto-picked
        from {128, 64, 32} by the registration function).
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
                f"qk_rope_head_dim={qk_rope_head_dim} must be in "
                f"[0, head_dim={head_dim}]"
            )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.softmax_scale = float(softmax_scale)
        # attn_sink — single fp32 per head, additive logit slot.
        self.attn_sink = nn.Parameter(
            torch.zeros(num_heads, dtype=torch.float32)
        )

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,
        gathered_kv: torch.Tensor,
        *,
        attn_sink_override: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Standard MLA attention with causal mask + attn_sink.

        ``q``           : ``[T_q,  H, D]`` bf16, already projected+RoPE'd.
        ``gathered_kv`` : ``[T_kv, D]``    bf16, single-row K==V.
        Returns ``o``   : ``[T_q,  H, D]`` bf16.
        """
        if q.dim() != 3:
            raise ValueError(f"q must be [T_q, H, D]; got {tuple(q.shape)}")
        if gathered_kv.dim() != 2:
            raise ValueError(
                f"gathered_kv must be [T_kv, D]; got {tuple(gathered_kv.shape)}"
            )
        T_q, H, D = q.shape
        T_kv, D2 = gathered_kv.shape
        if H != self.num_heads or D != self.head_dim:
            raise ValueError(
                f"q shape {tuple(q.shape)} does not match num_heads="
                f"{self.num_heads}, head_dim={self.head_dim}"
            )
        if D2 != self.head_dim:
            raise ValueError(
                f"gathered_kv.dim(1)={D2} does not match head_dim="
                f"{self.head_dim}"
            )

        sink = self.attn_sink if attn_sink_override is None else attn_sink_override
        sink_f = sink.to(torch.float32)  # [H]

        q_f = q.float()
        kv_f = gathered_kv.float()

        # Logits [T_q, H, T_kv]
        # einsum: q[t, h, d] * kv[k, d] -> [t, h, k]
        logits = torch.einsum("thd,kd->thk", q_f, kv_f) * self.softmax_scale
        # Causal mask: kv_pos > q_pos => -inf. Under the v1 SWA-only contract
        # T_q == T_kv and the gathered workspace's absolute positions line up
        # with q's absolute positions, so we use the simple lower-triangular
        # form.
        q_idx = torch.arange(T_q, device=q.device).view(T_q, 1, 1)
        k_idx = torch.arange(T_kv, device=q.device).view(1, 1, T_kv)
        mask = k_idx > q_idx
        logits = logits.masked_fill(mask, float("-inf"))

        # Append attn_sink as an extra "virtual" KV slot per head with
        # value-contribution 0; this affects only the softmax denominator.
        sink_logit = sink_f.view(1, H, 1).expand(T_q, H, 1)
        full_logits = torch.cat([logits, sink_logit], dim=-1)  # [T_q, H, T_kv+1]
        weights_full = torch.softmax(full_logits, dim=-1)
        weights = weights_full[..., :T_kv]                     # [T_q, H, T_kv]

        # Output: weights @ gathered_kv  (V == K under MLA).
        out_f = torch.einsum("thk,kd->thd", weights, kv_f)
        return out_f.to(q.dtype)

    # ------------------------------------------------------------------
    # MPK plumbing
    # ------------------------------------------------------------------
    def auto_grid_dim(self, q_dt: Any) -> GridDim:
        """One CTA per Q row, capped at ``num_workers``.

        Constraint: ``Q_TILE == 1`` in the kernel, so the natural CTA
        count is ``T_q``.
        """
        from ... import context as _ctx
        pk = _ctx.current_pk()
        T_q = q_dt.dim(0) if isinstance(q_dt, DTensor) else int(q_dt.shape[0])
        return (max(1, min(T_q, pk.num_workers)), 1, 1)

    def compile(
        self,
        q: DTensor,
        gathered_kv: Union[torch.Tensor, DTensor],
        *,
        attn_sink_override: Optional[Union[torch.Tensor, DTensor]] = None,
        o: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``mla_v4_prefill_sm100`` task.

        Tensor contract:
          q                  : ``[T_q,  H, D]``  bf16, input.
          gathered_kv        : ``[T_kv, D]``     bf16, input (single-row K==V).
          attn_sink_override : ``[H]``           fp32, optional override of
                                                  ``self.attn_sink``.
          o                  : ``[T_q,  H, D]``  bf16, output (auto-allocated
                                                  if None).

        Notes: grid is ``(T_q, 1, 1)`` with one CTA per Q row. The kernel
        derives its Q row via ``task_metadata.token_offset``. Causal mask
        is applied inside the kernel as ``kv_pos <= q_pos`` where
        ``q_pos = token_offset`` (v1 SWA-only contract: ``T_q == T_kv``).
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        prefix = self.prefix or "mla_v4_prefill_"

        if q.num_dims != 3:
            raise ValueError(
                f"q must be a 3-D DTensor [T_q, H, D]; got num_dims={q.num_dims}"
            )
        T_q = q.dim(0)
        if q.dim(1) != self.num_heads:
            raise ValueError(
                f"q.dim(1)={q.dim(1)} does not match num_heads={self.num_heads}"
            )
        if q.dim(2) != self.head_dim:
            raise ValueError(
                f"q.dim(2)={q.dim(2)} does not match head_dim={self.head_dim}"
            )

        # Attach gathered_kv.
        def _attach_in(buf, expected_dtype_torch, name, expected_dims=None):
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

        gathered_kv_dt = _attach_in(
            gathered_kv, torch.bfloat16, f"{prefix}gathered_kv"
        )
        if gathered_kv_dt.num_dims != 2:
            raise ValueError(
                f"gathered_kv must be 2-D [T_kv, D]; got num_dims="
                f"{gathered_kv_dt.num_dims}"
            )
        T_kv = gathered_kv_dt.dim(0)
        if gathered_kv_dt.dim(1) != self.head_dim:
            raise ValueError(
                f"gathered_kv.dim(1)={gathered_kv_dt.dim(1)} does not match "
                f"head_dim={self.head_dim}"
            )

        # Attach attn_sink — use override if provided, otherwise self.attn_sink.
        if attn_sink_override is None:
            sink_buf = self.attn_sink.data
        else:
            sink_buf = attn_sink_override
        sink_dt = _attach_in(
            sink_buf, torch.float32, f"{prefix}attn_sink"
        )
        if sink_dt.num_dims != 1 or sink_dt.dim(0) != self.num_heads:
            raise ValueError(
                f"attn_sink must be 1-D [num_heads={self.num_heads}]; got "
                f"shape {tuple(sink_dt.dim(i) for i in range(sink_dt.num_dims))}"
            )

        # Resolve / allocate output.
        if o is None:
            o_dt = pk.new_tensor(
                dims=(T_q, self.num_heads, self.head_dim),
                dtype=mi.bfloat16,
                name=f"{prefix}o",
            )
        elif isinstance(o, torch.Tensor):
            if o.dtype != torch.bfloat16:
                raise ValueError(
                    f"o must have dtype bfloat16; got {o.dtype}"
                )
            o_dt = pk.attach_input(o, name=f"{prefix}o")
        elif isinstance(o, DTensor):
            o_dt = o
        else:
            raise TypeError(
                f"o must be None, torch.Tensor, or DTensor; got "
                f"{type(o).__name__}"
            )
        assert o_dt.num_dims == 3
        assert o_dt.dim(0) == T_q
        assert o_dt.dim(1) == self.num_heads
        assert o_dt.dim(2) == self.head_dim

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(q)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # All inputs/outputs are addressed via ``task_metadata.token_offset``
        # (not via TBGraph partitioning), so all map dims are (-1, -1, -1).
        # This matches the mla_v4_q_kv_rmsnorm_sm100 / inv_rope_fp8_quant_o
        # convention: the kernel receives base pointers and does its own
        # ``row = base + t * NUM_HEADS * HEAD_DIM`` arithmetic.
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q, (-1, -1, -1), -1, True)
        tb_graph.new_input(gathered_kv_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(sink_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(o_dt, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [q, gathered_kv_dt, sink_dt, o_dt],
            tb_graph,
        )
        # softmax_scale is passed via params[0] as an fp32 bit-pattern.
        # The .cu codegen reinterprets it back into a float literal.
        import struct as _struct
        scale_bits = _struct.unpack("<i", _struct.pack("<f", self.softmax_scale))[0]
        pk.kn_graph.register_task(
            tb_graph, "mla_v4_prefill_sm100", [scale_bits]
        )
        return o_dt
