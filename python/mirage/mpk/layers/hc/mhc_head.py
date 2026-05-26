"""DeepSeek V4-Flash ``mhc_head`` — final HC collapse before lm_head.

Backed by ``tasks/blackwell/mhc_head_sm100.cuh`` (task name
``"mhc_head_sm100"``). Two-pass per-token kernel: pass 1 accumulates
per-token squared-sum + hc dot-products with rows of ``fn``; pass 2
applies the sigmoid-gated weighted reduction to collapse [N, hc, H] to
[N, H]. See ``docs/mpk/deepseek_v4/hc.md``.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from .._base import MPKModule
from ...context import current_pk
from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class MhcHead(MPKModule):
    """mHC head: per-token RMS + projection + sigmoid-gated HC reduction.

    Constructor args:
      hidden_size : per-HC-copy hidden dim ``H``.
      hc_mult     : HC multiplicity (default 4).
      prefix      : MPK kernel-tensor name prefix.

    No nn.Parameter — ``fn``, ``hc_scale``, ``hc_base`` are passed in at
    compile-time (they live on the parent HC head block).
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

    def forward(
        self,
        residual: torch.Tensor,  # [N, hc, H] bf16
        fn: torch.Tensor,        # [hc, hc*H] fp32
        hc_scale: torch.Tensor,  # [1] fp32
        hc_base: torch.Tensor,   # [hc] fp32
        rms_eps: float = 1e-6,
        hc_eps: float = 1e-6,
    ) -> torch.Tensor:
        """RMS + linear + sigmoid + weighted-sum over HC copies."""
        N, hc, H = residual.shape
        x = residual.flatten(1).float()
        rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
        mixes = torch.nn.functional.linear(x, fn.float()) * rsqrt
        pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
        out = (pre.unsqueeze(-1) * residual.float()).sum(dim=1)
        return out.to(torch.bfloat16)

    def auto_grid_dim(self, residual_dt: DTensor) -> GridDim:
        """One CTA per token; partition on dim 0 (N)."""
        pk = current_pk()
        return (max(1, min(residual_dt.dim(0), pk.num_workers)), 1, 1)

    def compile(
        self,
        residual: DTensor,
        fn: DTensor,
        hc_scale: DTensor,
        hc_base: DTensor,
        *,
        out: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``mhc_head_sm100`` task.

        Tensor contract:
          residual : (N, hc, H) bf16, HC stream (row-partitioned on N).
          fn       : (hc, hc*H) fp32, projection (broadcast).
          hc_scale : (1,) fp32, sigmoid gate scale (broadcast).
          hc_base  : (hc,) fp32, sigmoid gate bias (broadcast).
          out      : (N, H) bf16, collapsed hidden state.

        Notes: param list is ``[hc_mult]``; ``rms_eps = hc_eps = 1e-6`` are
        hard-coded in codegen.
        """
        pk = current_pk()
        assert residual.num_dims == 3
        assert fn.num_dims == 2
        assert hc_scale.num_dims == 1
        assert hc_base.num_dims == 1

        hc = residual.dim(1)
        H = residual.dim(2)
        N = residual.dim(0)
        assert fn.dim(0) == hc
        assert fn.dim(1) == hc * H
        assert hc_scale.dim(0) == 1
        assert hc_base.dim(0) == hc
        prefix = self.prefix or "mhc_head_"

        if out is None:
            out_dt = pk.new_tensor(
                dims=(N, H), dtype=torch.bfloat16, name=f"{prefix}out"
            )
        elif isinstance(out, torch.Tensor):
            out_dt = pk.attach_input(out, name=f"{prefix}out")
        elif isinstance(out, DTensor):
            out_dt = out
        else:
            raise TypeError("out must be None, torch.Tensor, or DTensor")
        assert out_dt.num_dims == 2

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(residual)
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(residual, (0, -1, -1), -1, True)
        tb_graph.new_input(fn, (-1, -1, -1), -1, True)
        tb_graph.new_input(hc_scale, (-1, -1, -1), -1, True)
        tb_graph.new_input(hc_base, (-1, -1, -1), -1, True)
        tb_graph.new_input(out_dt, (0, -1, -1), -1, True)
        pk.kn_graph.customized(
            [residual, fn, hc_scale, hc_base, out_dt], tb_graph
        )
        pk.kn_graph.register_task(
            tb_graph, "mhc_head_sm100", [int(self.hc_mult)]
        )
        return out_dt
