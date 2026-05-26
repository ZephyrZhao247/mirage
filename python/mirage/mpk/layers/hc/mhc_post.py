"""DeepSeek V4-Flash mHC post block (K5 HC expand step).

Backed by ``tasks/blackwell/mhc_post_sm100.cuh`` (task name
``"mhc_post_sm100"``). Combines the attention/FFN block output ``x`` with
the pre-block HC stream snapshot ``residual`` using the per-token
``post_mix`` / ``comb_mix`` produced by :class:`MhcPre` to emit the new
HC stream residual. See ``docs/mpk/deepseek_v4/hc.md``.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from .._base import MPKModule
from ...context import current_pk
from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class MhcPost(MPKModule):
    """mHC K5 expand: rebuild the [N, hc, H] HC residual.

    Constructor args:
      hidden_size : per-HC-copy hidden dim ``H``.
      hc_mult     : HC multiplicity (default 4).
      prefix      : MPK kernel-tensor name prefix.

    No nn.Parameter — mix coefficients are computed by ``MhcPre``.
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
        x: torch.Tensor,
        residual: torch.Tensor,
        post_mix: torch.Tensor,
        comb_mix: torch.Tensor,
    ) -> torch.Tensor:
        """``out[n, hc_o, h] = post_mix * x + sum_{hc_i} comb_mix * residual``."""
        term1 = post_mix.unsqueeze(-1) * x.unsqueeze(-2).float()
        term2 = torch.sum(
            comb_mix.unsqueeze(-1) * residual.unsqueeze(-2).float(), dim=1
        )
        return (term1 + term2).to(torch.bfloat16)

    def auto_grid_dim(self, x_dt: DTensor) -> GridDim:
        """One CTA per token; CTAs derive ``n`` from token_offset."""
        pk = current_pk()
        return (max(1, min(x_dt.dim(0), pk.num_workers)), 1, 1)

    def compile(
        self,
        x: DTensor,
        residual: DTensor,
        post_mix: DTensor,
        comb_mix: DTensor,
        *,
        out: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``mhc_post_sm100`` task.

        Tensor contract:
          x        : (N, H) bf16, attn/ffn block output.
          residual : (N, hc, H) bf16, pre-block HC stream snapshot.
          post_mix : (N, hc) fp32, from ``MhcPre`` (sigmoid-scaled).
          comb_mix : (N, hc, hc) fp32, from ``MhcPre`` (Sinkhorn-normalized).
          out      : (N, hc, H) bf16, new HC stream residual.

        Notes: grid is ``(N, 1, 1)``. The single param is ``num_threads``
        (= ``block_dim[0]``); codegen routes inputs as
        ``[comb_mix, residual, post_mix, x]``.
        """
        pk = current_pk()
        assert x.num_dims == 2
        assert residual.num_dims == 3
        assert post_mix.num_dims == 2
        assert comb_mix.num_dims == 3

        N = residual.dim(0)
        hc = residual.dim(1)
        H = residual.dim(2)
        prefix = self.prefix or "mhc_post_"

        if out is None:
            out_dt = pk.new_tensor(
                dims=(N, hc, H), dtype=torch.bfloat16, name=f"{prefix}out"
            )
        elif isinstance(out, torch.Tensor):
            out_dt = pk.attach_input(out, name=f"{prefix}out")
        elif isinstance(out, DTensor):
            out_dt = out
        else:
            raise TypeError("out must be None, torch.Tensor, or DTensor")
        assert out_dt.num_dims == 3

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(comb_mix, (0, -1, -1), -1, True)
        tb_graph.new_input(residual, (0, -1, -1), -1, True)
        tb_graph.new_input(post_mix, (0, -1, -1), -1, True)
        tb_graph.new_input(x, (0, -1, -1), -1, True)
        tb_graph.new_input(out_dt, (0, -1, -1), -1, True)
        pk.kn_graph.customized(
            [comb_mix, residual, post_mix, x, out_dt], tb_graph
        )
        num_threads = block_dim[0]
        pk.kn_graph.register_task(
            tb_graph, "mhc_post_sm100", [num_threads]
        )
        return out_dt
