"""DeepSeek V4-Flash mHC pre-norm GEMM (decomposed v1).

Computes the projection ``residual_flat @ fn.T`` plus the per-row
squared-sum used as the RMSNorm denominator consumed by :class:`MhcPre`.
Backed by ``tasks/blackwell/sum_of_squares_sm100.cuh`` (registered task
name ``"sum_of_squares_sm100"``). v1 is a single naive kernel
(bf16/bf16/bf16 with fp32 accumulators); v2 will widen ``fn`` to fp32 and
emit a fused TF32 GEMM. See ``docs/mpk/deepseek_v4/hc.md`` § 1.7 / 1.8.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from .._base import MPKModule
from ...context import current_pk
from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class MhcPrenormGemm(MPKModule):
    """mHC pre-norm GEMM + per-row squared-sum (v1, single naive task).

    Constructor args:
      hidden_size : per-HC-copy hidden dim ``H``.
      hc_mult     : HC multiplicity (default 4, => ``hc3 = (2 + hc) * hc = 24``).
      prefix      : MPK kernel-tensor name prefix.

    ``fn`` (the projection weight, ``[hc3, hc*H]`` bf16) lives on the
    parent HC block, not on this module — it is passed in at compile-time.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int = 4,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.hc3 = (2 + hc_mult) * hc_mult

    def forward(
        self, residual: torch.Tensor, fn: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """``residual_flat @ fn.T`` (bf16) + per-row squared-sum (fp32)."""
        residual_flat = residual.reshape(residual.shape[0], -1)
        gemm_out_mul = (residual_flat.float() @ fn.float().T).to(torch.bfloat16)
        gemm_out_sqrsum = residual_flat.float().pow(2).sum(dim=-1)
        return gemm_out_mul, gemm_out_sqrsum

    def auto_grid_dim(self, residual_dt: DTensor) -> GridDim:
        """One CTA per token (kernel partitions on dim 0), capped at num_workers."""
        pk = current_pk()
        return (max(1, min(residual_dt.dim(0), pk.num_workers)), 1, 1)

    def compile(
        self,
        residual: DTensor,
        fn: DTensor,
        *,
        gemm_out_mul: Optional[Union[torch.Tensor, DTensor]] = None,
        gemm_out_sqrsum: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register one ``sum_of_squares_sm100`` task.

        Tensor contract:
          residual        : (N, hc*H) bf16, caller flattens [N, hc, H] to 2-D.
          fn              : (hc3, hc*H) bf16, projection weights (broadcast).
          gemm_out_mul    : (N, hc3) bf16, ``residual @ fn.T`` (row-partitioned).
          gemm_out_sqrsum : (N,) fp32, per-row squared-sum (row-partitioned).

        Notes: v1 only — single split, bf16 fn. ``hc3 = (2 + hc_mult) * hc_mult``.
        """
        pk = current_pk()
        if residual.num_dims != 2:
            raise ValueError(
                f"MhcPrenormGemm expects 2-D residual [N, hc*H]; got "
                f"num_dims={residual.num_dims}"
            )
        if fn.num_dims != 2:
            raise ValueError(
                f"MhcPrenormGemm expects 2-D fn [hc3, hc*H]; got "
                f"num_dims={fn.num_dims}"
            )

        n_tokens = residual.dim(0)
        prefix = self.prefix or "mhc_prenorm_gemm_"

        if gemm_out_mul is None:
            mul_dt = pk.new_tensor(
                dims=(n_tokens, self.hc3),
                dtype=residual.dtype,
                name=f"{prefix}gemm_out_mul",
            )
        elif isinstance(gemm_out_mul, torch.Tensor):
            mul_dt = pk.attach_input(gemm_out_mul, name=f"{prefix}gemm_out_mul")
        elif isinstance(gemm_out_mul, DTensor):
            mul_dt = gemm_out_mul
        else:
            raise TypeError(
                "gemm_out_mul must be None, torch.Tensor, or DTensor"
            )

        if gemm_out_sqrsum is None:
            sqrsum_dt = pk.new_tensor(
                dims=(n_tokens,),
                dtype=torch.float32,
                name=f"{prefix}gemm_out_sqrsum",
            )
        elif isinstance(gemm_out_sqrsum, torch.Tensor):
            sqrsum_dt = pk.attach_input(
                gemm_out_sqrsum, name=f"{prefix}gemm_out_sqrsum"
            )
        elif isinstance(gemm_out_sqrsum, DTensor):
            sqrsum_dt = gemm_out_sqrsum
        else:
            raise TypeError(
                "gemm_out_sqrsum must be None, torch.Tensor, or DTensor"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(residual)
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # Layout matches register_sum_of_squares_sm100_task:
        #   inputs : residual (partitioned row-wise), fn (broadcast)
        #   outputs: gemm_out_mul (row-wise), gemm_out_sqrsum (row-wise)
        tb_graph.new_input(residual, (0, -1, -1), -1, True)
        tb_graph.new_input(fn, (-1, -1, -1), -1, True)
        tb_graph.new_input(mul_dt, (0, -1, -1), -1, True)
        tb_graph.new_input(sqrsum_dt, (0, -1, -1), -1, True)
        pk.kn_graph.customized(
            [residual, fn, mul_dt, sqrsum_dt], tb_graph
        )
        pk.kn_graph.register_task(tb_graph, "sum_of_squares_sm100")
        return mul_dt, sqrsum_dt
