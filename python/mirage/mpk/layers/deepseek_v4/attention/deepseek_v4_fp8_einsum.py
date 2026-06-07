"""V4-Flash ``deepseek_v4_fp8_einsum`` (wo_a o-projection) -- NEW naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/deepseek_v4_fp8_einsum.md``.

Decision: **NEW**.

Rationale
---------
The DeepGEMM ``fp8_einsum("bhr,hdr->bhd", ...)`` consumes the
``o_fp8``/``o_scale`` outputs of :class:`V4FusedInvRopeFP8Quant` (wave-5A)
against the ``wo_a`` weights -- producing the o-projection bf16 output
``z[T, n_groups, o_lora_rank]``.

REUSE/EXTEND analysis
~~~~~~~~~~~~~~~~~~~~~
The existing :class:`LinearFP8BMM` (``linear_fp8_bmm_sm100.cuh``) is the
closest V3 sibling:

* **Mismatched activation layout**. ``LinearFP8BMM`` expects
  ``x_fp8: [N, H, D_in]`` (token-major-within-head) with row-major
  contiguous final dim. Wave-5A's producer writes
  ``o_fp8: [T, n_groups, d_in]`` -- the same final-dim layout, BUT the
  activation per-(token, head) row is read against a swapAB-style
  weight TMA descriptor that needs SFA grid-aligned columns.
* **Mismatched scale dtype layout**. ``LinearFP8BMM`` reads
  ``x_scale: [N, H, D_in/128]`` uint32 UE8M0-packed where each uint32
  packs 4 K-blocks along the D_in axis (i.e. one uint32 covers
  D_in lanes ``[k*128, k*128+128)`` -- a row of 4 packed bytes).
  Wave-5A's ``o_scale`` is also uint32 UE8M0-packed but the per-CTA
  scale_inner count is ``(num_blocks_per_head * heads_per_group) / 4``;
  same encoding but consumed at the (T, n_groups) granularity.
* **Mismatched output dtype**. The einsum output is bf16 -- which
  ``LinearFP8BMM`` already produces -- but ``LinearFP8BMM`` does
  ``out[n, h, :] = x[n, h, :] @ w[h, :, :].T``, requiring the weight
  contracted dim to be ``D_in``. For the o-projection
  ``out[t, h, dout] = sum_r o_fp8[t, h, r] * wo_a[h, dout, r]``,
  the contraction axis is ``r = d_in = heads_per_group * head_dim``,
  which matches the LinearFP8BMM shape exactly.

After staring at this for a while: the math is **identical** to
``LinearFP8BMM`` (per-head FP8 BMM with UE8M0 scales) modulo the
*reshape* on the activation. The wave-5A producer writes scales in
``[T, n_groups, scale_inner]`` layout where ``scale_inner =
ceil(num_blocks_per_head*heads_per_group / 4)`` and the underlying
encoding is the same UE8M0-packed 4-bytes-per-int32. **However**,
plumbing through the V3 path requires assuming a specific scale
column stride and TMA descriptor; the V3 kernel uses TMA loads which
this naive port intentionally avoids.

Decision: implement a NEW naive kernel that mirrors the V3 math without
relying on the V3 TMA / swapAB plumbing. The math is in
``include/mirage/persistent_kernel/tasks/blackwell/deepseek_v4_fp8_einsum_v4_sm100.cuh``.

Naive design
------------
* One CTA per (token, group). ``grid = (T, n_groups, 1)``.
* For each (t, h) the CTA loops over ``d_out`` outputs and reduces a
  fp32 accumulator with shared block scales. UE8M0 byte ``b`` decodes
  to ``scale = 2^(b - 127)``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor


__all__ = ["V4DeepseekFP8Einsum"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


def _ue8m0_packed_to_block_scales(scale_packed: torch.Tensor, n_blocks: int) -> torch.Tensor:
    """Decode int32-packed UE8M0 scales (4 bytes/int32) to fp32 per-block.

    Input: packed scales with last-dim = ceil(n_blocks/4) int32.
    Output: fp32 with last-dim = n_blocks (one scale per 128-wide block).
    """
    leading = scale_packed.shape[:-1]
    inner = scale_packed.shape[-1]
    bytes_view = scale_packed.contiguous().view(torch.uint8).reshape(
        *leading, inner * 4
    )
    exp = bytes_view.to(torch.float32) - 127.0
    scales = torch.pow(torch.tensor(2.0), exp)
    return scales[..., :n_blocks]


class V4DeepseekFP8Einsum(MPKModule):
    """Naive FP8 o-projection einsum.

    Owns two ``nn.Parameter`` weights:
      * ``wo_a_fp8: [n_groups, d_out, d_in]`` raw uint8 = float8_e4m3fn.
      * ``wo_a_scale: [n_groups, d_out, scale_inner_w]`` uint32 UE8M0-packed.

    Constructor args:
        n_groups, d_in, d_out, quant_block_size (default 128).
    """

    def __init__(
        self,
        n_groups: int,
        d_in: int,
        d_out: int,
        *,
        quant_block_size: int = 128,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if d_in % quant_block_size != 0:
            raise ValueError(
                f"d_in ({d_in}) must be divisible by quant_block_size "
                f"({quant_block_size})"
            )
        self.n_groups = int(n_groups)
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.quant_block_size = int(quant_block_size)
        self.n_r_blocks = self.d_in // self.quant_block_size
        # 4 UE8M0 bytes per int32 packed lane.
        self.scale_inner_w = (self.n_r_blocks + 3) // 4

        self.wo_a_fp8 = nn.Parameter(
            torch.empty(
                self.n_groups, self.d_out, self.d_in, dtype=torch.uint8
            ),
            requires_grad=False,
        )
        self.wo_a_scale = nn.Parameter(
            torch.empty(
                self.n_groups, self.d_out, self.scale_inner_w,
                dtype=torch.uint32,
            ),
            requires_grad=False,
        )

    # ------------------------------------------------------------------
    def forward(
        self,
        o_fp8: torch.Tensor,
        o_scale: torch.Tensor,
    ) -> torch.Tensor:
        """Eager reference for the FP8 einsum.

        Inputs:
          o_fp8   : [T, n_groups, d_in] float8_e4m3fn (or uint8 reinterp).
          o_scale : [T, n_groups, scale_inner] int32 UE8M0-packed.
        Returns:
          out     : [T, n_groups, d_out] bf16.
        """
        assert o_fp8.dim() == 3
        T = o_fp8.shape[0]
        assert o_fp8.shape[1] == self.n_groups
        assert o_fp8.shape[2] == self.d_in

        if o_fp8.dtype == torch.float8_e4m3fn:
            o_f32 = o_fp8.to(torch.float32)
        else:
            o_f32 = o_fp8.view(torch.float8_e4m3fn).to(torch.float32)

        w_f32 = self.wo_a_fp8.data.view(torch.float8_e4m3fn).to(torch.float32)

        # Decode block scales.
        o_block_scales = _ue8m0_packed_to_block_scales(o_scale, self.n_r_blocks).to(
            o_f32.device
        )  # [T, n_groups, n_r_blocks]
        w_block_scales = _ue8m0_packed_to_block_scales(
            self.wo_a_scale.data, self.n_r_blocks
        ).to(o_f32.device)  # [n_groups, d_out, n_r_blocks]

        # Apply scales along the r-block axis (broadcast within each
        # quant block of QB lanes).
        QB = self.quant_block_size
        o_expanded = o_block_scales.repeat_interleave(QB, dim=-1)  # [T, n_groups, d_in]
        o_dequant = o_f32 * o_expanded
        w_expanded = w_block_scales.repeat_interleave(QB, dim=-1)  # [n_groups, d_out, d_in]
        w_dequant = w_f32 * w_expanded

        # Einsum "bhr,hdr->bhd"
        out_f32 = torch.einsum("bhr,hdr->bhd", o_dequant, w_dequant)
        return out_f32.to(torch.bfloat16)

    # ------------------------------------------------------------------
    def auto_grid_dim(self, o_fp8_dt: DTensor) -> GridDim:
        pk = current_pk()
        T = o_fp8_dt.dim(0)
        gy = self.n_groups
        # Cap so that gx*gy <= num_workers.
        budget = max(1, int(pk.num_workers) // max(gy, 1))
        gx = max(1, min(T, budget))
        return (gx, gy, 1)

    # ------------------------------------------------------------------
    def compile(
        self,
        o_fp8: DTensor,
        o_scale: DTensor,
        *,
        output: Optional[DTensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``deepseek_v4_fp8_einsum_v4_sm100`` task.

        Tensor contract:
          o_fp8       : (T, n_groups, d_in)  float8_e4m3fn (or uint8 reinterp).
          o_scale     : (T, n_groups, scale_inner) int32 UE8M0-packed.
          wo_a (auto) : (n_groups, d_out, d_in)  fp8.
          wo_a_scale  : (n_groups, d_out, scale_inner_w) int32 UE8M0-packed.
          output      : (T, n_groups, d_out) bf16, allocated if None.
        """
        from .....core import CyTBGraph, bfloat16 as _mi_bf16
        from .....kernel import TBGraph

        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(o_fp8)
        if block_dim is None:
            block_dim = self.default_block_dim()

        if o_fp8.num_dims != 3 or o_fp8.dim(2) != self.d_in:
            raise ValueError(
                f"o_fp8 must be (T, n_groups, {self.d_in}); got dims="
                f"{tuple(o_fp8.dim(i) for i in range(o_fp8.num_dims))}"
            )
        if o_fp8.dim(1) != self.n_groups:
            raise ValueError(
                f"o_fp8 dim 1 ({o_fp8.dim(1)}) != n_groups ({self.n_groups})"
            )

        w_dt = pk.attach_input(self.wo_a_fp8.data, name=f"{self.prefix}wo_a_fp8")
        ws_dt = pk.attach_input(self.wo_a_scale.data, name=f"{self.prefix}wo_a_scale")

        T = o_fp8.dim(0)
        if output is None:
            out_dt = pk.new_tensor(
                dims=(T, self.n_groups, self.d_out),
                dtype=_mi_bf16,
                name=f"{self.prefix}einsum_out",
            )
        elif isinstance(output, torch.Tensor):
            out_dt = pk.attach_input(output, name=f"{self.prefix}einsum_out")
        elif isinstance(output, DTensor):
            out_dt = output
        else:
            raise TypeError(
                f"output must be None, torch.Tensor, or DTensor; got "
                f"{type(output).__name__}"
            )

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # o_fp8: partition on dim 0 (token) AND dim 1 (group). The runtime
        # pre-offsets the pointer to the (t, h) row of length d_in.
        tb_graph.new_input(o_fp8, (0, 1, -1), -1, True)
        tb_graph.new_input(o_scale, (0, 1, -1), -1, True)
        # wo_a_fp8 / wo_a_scale: partition on dim 0 (group) only.
        tb_graph.new_input(w_dt, (-1, 1, -1), -1, True)
        tb_graph.new_input(ws_dt, (-1, 1, -1), -1, True)
        # out: partition on dim 0 (token) AND dim 1 (group).
        tb_graph.new_input(out_dt, (0, 1, -1), -1, True)
        pk.kn_graph.customized(
            [o_fp8, o_scale, w_dt, ws_dt, out_dt], tb_graph
        )
        pk.kn_graph.register_task(tb_graph, "deepseek_v4_fp8_einsum_v4_sm100", [])
        return out_dt
