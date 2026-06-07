"""V4-Flash ``fused_indexer_q_rope_quant`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fused_indexer_q_rope_quant.md``.

Decision: **NEW**.

Coupling
--------
Class-B sibling under the ``use_fp4_cache`` flag (FP8 vs MXFP4).
Companion FP8 K-side kernel:
``fused_kv_compress_norm_rope_insert_indexer_attn`` (Wave-3A). The
downstream DeepGEMM MQA-logits dispatchers select FP8 vs MXFP4 via the
``(q_values, q_scale)`` tuple — for FP8 we pass ``q_scale=None`` (the
per-(token,head) scalar ``q_scale`` is folded into ``weights_out``).

Rationale
---------
This kernel fuses three operations per (token, head):

1. GPT-J interleaved RoPE on the trailing rope half of ``q``.
2. UE8M0-discrete scalar ``q_scale = 2^ceil(log2(amax/448))`` derived
   from max-abs over the full (nope, r_even, r_odd) head row.
3. FP8 e4m3 divide-and-cast quant + weight fold:
   ``weights_out = weights_in * q_scale * softmax_scale * head_scale``.

There is no existing MPK task with this contract.

So this is a NEW kernel:

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/fused_indexer_q_rope_quant_v4_sm100.cuh``
* Task name:   ``fused_indexer_q_rope_quant_v4_sm100``
* Enum slot:   ``TASK_FUSED_INDEXER_Q_ROPE_QUANT_V4_SM100 = 376``

The Blackwell impl is intentionally NAIVE:

* One CTA per (token, head). ``grid = (num_rows, 1, 1)`` where
  ``num_rows = num_tokens * n_heads``. The catalog flattens leading
  axes so each CTA's row is just a contiguous slice.
* ``block_dim = (256, 1, 1)`` (Blackwell ``WORKER_NUM_THREADS``).
* No TMA / no UMMA / no warp-specialization.

Audit
-----
* dtype: ``q_in`` bf16, ``cos_sin`` fp32, ``weights_in`` bf16; outputs
  ``q_out`` fp8 e4m3, ``weights_out`` fp32. No silent casts; all dtype
  pinning matches the spec.
* layout: row-major contiguous everywhere; per-row partitioning on dim 0.
* multi-batch: by construction (rows are token*head); test uses
  ``num_rows >= 2``.
* ``forward()``: faithful PyTorch reference with the bf16 roundtrip on
  ``(r_even, r_odd)`` for parity.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

import mirage as mi

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor

__all__ = ["V4FusedIndexerQRopeQuant"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


def _float_to_int_bits(x: float) -> int:
    """Bit-cast a float into an int (for the params vector)."""
    import struct

    return int.from_bytes(struct.pack("<f", float(x)), "little", signed=False)


class V4FusedIndexerQRopeQuant(MPKModule):
    """V4-Flash FP8 indexer Q-side naive Blackwell kernel.

    Constructor args:

      * ``head_dim``       -- per-head feature dim (V4-Flash: 128).
      * ``half_rot_dim``   -- HALF of the rope dim (V4-Flash: 32 ->
                              rot_dim = 64; rope is GPT-J interleaved
                              on the trailing 2*half_rot_dim lanes).
      * ``softmax_scale``  -- scalar (default ``head_dim ** -0.5``).
      * ``head_scale``     -- scalar (default ``n_heads ** -0.5``).
                              Passed at construction (the catalog needs
                              n_heads to compute it).
    """

    def __init__(
        self,
        head_dim: int = 128,
        half_rot_dim: int = 32,
        softmax_scale: Optional[float] = None,
        head_scale: float = 0.125,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if head_dim <= 0:
            raise ValueError(
                f"V4FusedIndexerQRopeQuant head_dim must be positive; "
                f"got {head_dim}"
            )
        if half_rot_dim <= 0 or 2 * half_rot_dim > head_dim:
            raise ValueError(
                f"V4FusedIndexerQRopeQuant: invalid half_rot_dim "
                f"({half_rot_dim}); must satisfy 0 < 2*half_rot_dim <= "
                f"head_dim ({head_dim})"
            )
        self.head_dim = head_dim
        self.half_rot_dim = half_rot_dim
        self.rot_dim = 2 * half_rot_dim
        self.nope_dim = head_dim - self.rot_dim
        if softmax_scale is None:
            softmax_scale = head_dim ** -0.5
        self.softmax_scale = float(softmax_scale)
        self.head_scale = float(head_scale)

    # ------------------------------------------------------------------
    # PyTorch reference (faithful to the Triton kernel body)
    # ------------------------------------------------------------------
    def forward(
        self,
        q_in: torch.Tensor,       # bf16 [num_rows, head_dim]
        cos_sin: torch.Tensor,    # fp32 [num_rows, 2*half_rot_dim]
        weights_in: torch.Tensor, # bf16 [num_rows, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert q_in.dim() == 2 and q_in.shape[1] == self.head_dim
        assert cos_sin.dim() == 2 and cos_sin.shape[1] == 2 * self.half_rot_dim
        assert weights_in.dim() == 2 and weights_in.shape[1] == 1
        num_rows = q_in.shape[0]

        q = q_in.to(torch.float32)
        cos = cos_sin[:, : self.half_rot_dim].to(torch.float32)
        sin = cos_sin[:, self.half_rot_dim :].to(torch.float32)

        nope = q[:, : self.nope_dim]
        rope = q[:, self.nope_dim :]
        x_even = rope[:, 0::2]
        x_odd = rope[:, 1::2]
        r_even = x_even * cos - x_odd * sin
        r_odd = x_odd * cos + x_even * sin
        # bf16 roundtrip
        r_even = r_even.to(torch.bfloat16).to(torch.float32)
        r_odd = r_odd.to(torch.bfloat16).to(torch.float32)

        # Rebuild the interleaved rope half.
        rope_out = torch.empty_like(rope)
        rope_out[:, 0::2] = r_even
        rope_out[:, 1::2] = r_odd

        q_full = torch.cat([nope, rope_out], dim=-1)  # [num_rows, head_dim]

        # amax + UE8M0-discrete q_scale.
        amax = q_full.abs().amax(dim=-1)              # [num_rows]
        amax_clamped = torch.clamp(amax, min=1e-4)
        q_scale = torch.exp2(torch.ceil(torch.log2(amax_clamped / 448.0)))

        # Divide-and-cast to fp8 e4m3.
        q_fp8 = (q_full / q_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)

        # Fold weights.
        w = weights_in.squeeze(-1).to(torch.float32)
        w_out = (w * q_scale * self.softmax_scale * self.head_scale).view(num_rows, 1)
        return q_fp8, w_out

    def auto_grid_dim(self, q_dt: Any) -> GridDim:
        pk = current_pk()
        num_rows = q_dt.dim(0)
        return (max(1, min(num_rows, int(pk.num_workers))), 1, 1)

    def compile(
        self,
        q_dt: Any,
        cos_sin_dt: Any,
        weights_in_dt: Any,
        *,
        q_out: Optional[Any] = None,
        weights_out: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register one ``fused_indexer_q_rope_quant_v4_sm100`` task.

        Tensor contract:
          q_dt          : (num_rows, head_dim)         bf16
          cos_sin_dt    : (num_rows, 2*half_rot_dim)   fp32
          weights_in_dt : (num_rows, 1)                bf16
          q_out         : (num_rows, head_dim)         fp8_e4m3, alloc if None
          weights_out   : (num_rows, 1)                fp32,    alloc if None
        """
        from .....core import CyTBGraph
        from .....kernel import TBGraph

        pk = current_pk()

        if q_dt.num_dims != 2 or q_dt.dim(1) != self.head_dim:
            raise ValueError(
                f"V4FusedIndexerQRopeQuant: q_dt must be 2-D "
                f"(num_rows, head_dim={self.head_dim}); got num_dims="
                f"{q_dt.num_dims}, dim(1)={q_dt.dim(1)}"
            )
        if cos_sin_dt.num_dims != 2 or cos_sin_dt.dim(1) != 2 * self.half_rot_dim:
            raise ValueError(
                f"V4FusedIndexerQRopeQuant: cos_sin_dt must be 2-D "
                f"(num_rows, {2 * self.half_rot_dim}); got "
                f"num_dims={cos_sin_dt.num_dims}, dim(1)={cos_sin_dt.dim(1)}"
            )
        if weights_in_dt.num_dims != 2 or weights_in_dt.dim(1) != 1:
            raise ValueError(
                f"V4FusedIndexerQRopeQuant: weights_in_dt must be 2-D "
                f"(num_rows, 1); got num_dims={weights_in_dt.num_dims}, "
                f"dim(1)={weights_in_dt.dim(1)}"
            )

        num_rows = q_dt.dim(0)
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(q_dt)
        if block_dim is None:
            block_dim = self.default_block_dim()

        if q_out is None:
            q_out_dt = pk.new_tensor(
                dims=(num_rows, self.head_dim),
                dtype=mi.float8_e4m3,
                name=f"{self.prefix}q_fp8",
            )
        elif isinstance(q_out, torch.Tensor):
            q_out_dt = pk.attach_input(q_out, name=f"{self.prefix}q_fp8")
        elif isinstance(q_out, DTensor):
            q_out_dt = q_out
        else:
            raise TypeError(
                "V4FusedIndexerQRopeQuant.compile q_out must be None, "
                f"torch.Tensor, or DTensor; got {type(q_out).__name__}"
            )

        if weights_out is None:
            w_out_dt = pk.new_tensor(
                dims=(num_rows, 1),
                dtype=mi.float32,
                name=f"{self.prefix}weights_out",
            )
        elif isinstance(weights_out, torch.Tensor):
            w_out_dt = pk.attach_input(
                weights_out, name=f"{self.prefix}weights_out"
            )
        elif isinstance(weights_out, DTensor):
            w_out_dt = weights_out
        else:
            raise TypeError(
                "V4FusedIndexerQRopeQuant.compile weights_out must be None, "
                f"torch.Tensor, or DTensor; got {type(weights_out).__name__}"
            )

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(cos_sin_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(weights_in_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(q_out_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(w_out_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [q_dt, cos_sin_dt, weights_in_dt, q_out_dt, w_out_dt], tb_graph
        )
        pk.kn_graph.register_task(
            tb_graph,
            "fused_indexer_q_rope_quant_v4_sm100",
            [
                self.head_dim,
                self.half_rot_dim,
                _float_to_int_bits(self.softmax_scale),
                _float_to_int_bits(self.head_scale),
            ],
        )
        return q_out_dt, w_out_dt
