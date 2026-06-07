"""V4-Flash ``fused_indexer_q_rope_mxfp4`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fused_indexer_q_rope_mxfp4.md``.

Decision: **NEW**.

Coupling
--------
Class-B sibling under the ``use_fp4_cache`` flag (FP8 vs MXFP4).
Companion MXFP4 K-side kernel:
``fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn`` (Wave-3A).
DeepGEMM dispatches to the MXFP4 MQA-logits path when ``q_scale`` is
not None (here, it is an int32 view of the per-block UE8M0 byte
scales).

Rationale
---------
This kernel fuses three operations per (token, head):

1. GPT-J interleaved RoPE on the trailing rope half of ``q``.
2. Per-32-element-block UE8M0 scales:
   ``scale = 2^ceil(log2(amax(block) / 6.0))``.
3. E2M1x2 byte-packed MXFP4 (low nibble = even-index, high nibble =
   odd-index) + weight fold:
   ``weights_out = weights_in * softmax_scale * head_scale``
   (NO q_scale fold — q_scale is per-block here and stays alongside
   the values).

There is no existing MPK task with this contract.

So this is a NEW kernel:

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/fused_indexer_q_rope_mxfp4_v4_sm100.cuh``
* Task name:   ``fused_indexer_q_rope_mxfp4_v4_sm100``
* Enum slot:   ``TASK_FUSED_INDEXER_Q_ROPE_MXFP4_V4_SM100 = 377``

Naive design:

* One CTA per (token, head). grid = (num_rows, 1, 1); block = (256, 1, 1).
* No TMA / no UMMA / no warp-spec.

Audit
-----
* dtype: ``q_in`` bf16, ``cos_sin`` fp32, ``weights_in`` bf16; outputs
  ``q_packed`` uint8 (E2M1x2), ``q_scale`` uint8 (UE8M0), ``weights_out``
  fp32. No silent casts.
* layout: row-major contiguous; per-row partitioning on dim 0.
* multi-batch: by construction.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

import mirage as mi

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor

__all__ = ["V4FusedIndexerQRopeMxfp4"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


def _float_to_int_bits(x: float) -> int:
    import struct
    return int.from_bytes(struct.pack("<f", float(x)), "little", signed=False)


# E2M1 representable magnitudes used for the reference quant.
_E2M1_MAGS = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
    dtype=torch.float32,
)


def _fp32_to_e2m1_nibble(x: torch.Tensor) -> torch.Tensor:
    """Reference E2M1 round-to-nearest quant. Returns uint8 nibbles (4 bits)."""
    abs_x = x.abs()
    # Boundaries between adjacent magnitudes:
    # 0/0.5: 0.25; 0.5/1.0: 0.75; 1.0/1.5: 1.25; 1.5/2.0: 1.75;
    # 2.0/3.0: 2.5; 3.0/4.0: 3.5; 4.0/6.0: 5.0.
    bounds = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32, device=x.device)
    # bucket index in [0, 7]
    mag = torch.zeros_like(abs_x, dtype=torch.int32)
    for b in bounds:
        mag = mag + (abs_x >= b).to(torch.int32)
    sign = (x < 0).to(torch.int32) * 8
    return (sign | mag).to(torch.uint8)


class V4FusedIndexerQRopeMxfp4(MPKModule):
    """V4-Flash MXFP4 indexer Q-side naive Blackwell kernel.

    Constructor args:

      * ``head_dim``       -- per-head feature dim (V4-Flash: 128).
      * ``half_rot_dim``   -- HALF of the rope dim (V4-Flash: 32).
      * ``mxfp4_block``    -- MXFP4 block size (V4-Flash: 32).
      * ``softmax_scale``  -- scalar (default head_dim**-0.5).
      * ``head_scale``     -- scalar (default 0.125 = 64**-0.5).
    """

    def __init__(
        self,
        head_dim: int = 128,
        half_rot_dim: int = 32,
        mxfp4_block: int = 32,
        softmax_scale: Optional[float] = None,
        head_scale: float = 0.125,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if head_dim <= 0 or head_dim % 2 != 0:
            raise ValueError(
                f"V4FusedIndexerQRopeMxfp4: head_dim must be positive and "
                f"even; got {head_dim}"
            )
        if half_rot_dim <= 0 or 2 * half_rot_dim > head_dim:
            raise ValueError(
                f"V4FusedIndexerQRopeMxfp4: invalid half_rot_dim "
                f"({half_rot_dim}); must satisfy 0 < 2*half_rot_dim <= "
                f"head_dim ({head_dim})"
            )
        if mxfp4_block <= 0 or mxfp4_block % 2 != 0 or head_dim % mxfp4_block != 0:
            raise ValueError(
                f"V4FusedIndexerQRopeMxfp4: mxfp4_block must be positive, "
                f"even, and divide head_dim; got {mxfp4_block}"
            )
        self.head_dim = head_dim
        self.half_rot_dim = half_rot_dim
        self.rot_dim = 2 * half_rot_dim
        self.nope_dim = head_dim - self.rot_dim
        self.mxfp4_block = mxfp4_block
        self.num_blocks = head_dim // mxfp4_block
        if softmax_scale is None:
            softmax_scale = head_dim ** -0.5
        self.softmax_scale = float(softmax_scale)
        self.head_scale = float(head_scale)

    def forward(
        self,
        q_in: torch.Tensor,        # bf16 [num_rows, head_dim]
        cos_sin: torch.Tensor,     # fp32 [num_rows, 2*half_rot_dim]
        weights_in: torch.Tensor,  # bf16 [num_rows, 1]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        r_even = r_even.to(torch.bfloat16).to(torch.float32)
        r_odd = r_odd.to(torch.bfloat16).to(torch.float32)
        rope_out = torch.empty_like(rope)
        rope_out[:, 0::2] = r_even
        rope_out[:, 1::2] = r_odd
        q_full = torch.cat([nope, rope_out], dim=-1)   # [num_rows, head_dim]

        # Per-block amax → UE8M0 scale.
        q_blk = q_full.view(num_rows, self.num_blocks, self.mxfp4_block)
        amax = q_blk.abs().amax(dim=-1)                # [num_rows, num_blocks]
        lo_floor = 6.0 * (2 ** -126)
        amax = torch.clamp(amax, min=lo_floor)
        log2_ratio = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
        scale = torch.exp2(log2_ratio)
        ue8m0 = (log2_ratio + 127).to(torch.uint8)     # [num_rows, num_blocks]
        inv_scale = 1.0 / scale                          # [num_rows, num_blocks]

        # Quant each (block, lane) → E2M1 nibble.
        q_quant = q_blk * inv_scale.unsqueeze(-1)        # [num_rows, num_blocks, block]
        nibbles = _fp32_to_e2m1_nibble(q_quant)          # uint8 [N, B, M]
        nibbles = nibbles.view(num_rows, self.head_dim)

        # Pack pairs (lo=even, hi=odd) into bytes.
        lo = nibbles[:, 0::2]                            # [N, head_dim/2]
        hi = nibbles[:, 1::2]
        packed = ((hi.to(torch.int32) << 4) | (lo.to(torch.int32) & 0xF)).to(torch.uint8)

        # Folded weights (NO q_scale fold).
        w = weights_in.squeeze(-1).to(torch.float32)
        w_out = (w * self.softmax_scale * self.head_scale).view(num_rows, 1)

        return packed, ue8m0, w_out

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
        q_packed: Optional[Any] = None,
        q_scale: Optional[Any] = None,
        weights_out: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor, DTensor]:
        """Register one ``fused_indexer_q_rope_mxfp4_v4_sm100`` task.

        Tensor contract:
          q_dt          : (num_rows, head_dim)         bf16
          cos_sin_dt    : (num_rows, 2*half_rot_dim)   fp32
          weights_in_dt : (num_rows, 1)                bf16
          q_packed      : (num_rows, head_dim/2)       uint8, alloc if None
          q_scale       : (num_rows, num_blocks)       uint8, alloc if None
          weights_out   : (num_rows, 1)                fp32,  alloc if None
        """
        from .....core import CyTBGraph
        from .....kernel import TBGraph

        pk = current_pk()

        if q_dt.num_dims != 2 or q_dt.dim(1) != self.head_dim:
            raise ValueError(
                f"V4FusedIndexerQRopeMxfp4: q_dt must be 2-D "
                f"(num_rows, {self.head_dim}); got num_dims={q_dt.num_dims}, "
                f"dim(1)={q_dt.dim(1)}"
            )
        if cos_sin_dt.num_dims != 2 or cos_sin_dt.dim(1) != 2 * self.half_rot_dim:
            raise ValueError(
                f"V4FusedIndexerQRopeMxfp4: cos_sin_dt must be 2-D "
                f"(num_rows, {2 * self.half_rot_dim})"
            )
        if weights_in_dt.num_dims != 2 or weights_in_dt.dim(1) != 1:
            raise ValueError(
                f"V4FusedIndexerQRopeMxfp4: weights_in_dt must be 2-D "
                f"(num_rows, 1)"
            )

        num_rows = q_dt.dim(0)
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(q_dt)
        if block_dim is None:
            block_dim = self.default_block_dim()

        if q_packed is None:
            q_packed_dt = pk.new_tensor(
                dims=(num_rows, self.head_dim // 2),
                dtype=mi.uint8,
                name=f"{self.prefix}q_packed",
            )
        elif isinstance(q_packed, torch.Tensor):
            q_packed_dt = pk.attach_input(q_packed, name=f"{self.prefix}q_packed")
        else:
            q_packed_dt = q_packed

        if q_scale is None:
            q_scale_dt = pk.new_tensor(
                dims=(num_rows, self.num_blocks),
                dtype=mi.uint8,
                name=f"{self.prefix}q_scale",
            )
        elif isinstance(q_scale, torch.Tensor):
            q_scale_dt = pk.attach_input(q_scale, name=f"{self.prefix}q_scale")
        else:
            q_scale_dt = q_scale

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
        else:
            w_out_dt = weights_out

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(cos_sin_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(weights_in_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(q_packed_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(q_scale_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(w_out_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [
                q_dt, cos_sin_dt, weights_in_dt,
                q_packed_dt, q_scale_dt, w_out_dt,
            ],
            tb_graph,
        )
        pk.kn_graph.register_task(
            tb_graph,
            "fused_indexer_q_rope_mxfp4_v4_sm100",
            [
                self.head_dim,
                self.half_rot_dim,
                self.mxfp4_block,
                _float_to_int_bits(self.softmax_scale),
                _float_to_int_bits(self.head_scale),
            ],
        )
        return q_packed_dt, q_scale_dt, w_out_dt
