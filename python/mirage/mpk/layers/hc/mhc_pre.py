"""DeepSeek V4-Flash mHC pre block (port of vLLM ``mhc_pre_big_fuse_tilelang``).

Backed by ``tasks/blackwell/mhc_pre_sm100.cuh`` (task name
``"mhc_pre_sm100"``). Reduces the split-K GEMM outputs, finalizes the
RMSNorm rsqrt, applies the K2 affine + sigmoid split to produce
``post_mix``, runs Sinkhorn iterations on the 4x4 ``comb_mix``, and emits
the K4 head reduction ``layer_input``. See ``docs/mpk/deepseek_v4/hc.md``.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

from .._base import MPKModule
from ...context import current_pk
from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class MhcPre(MPKModule):
    """mHC pre block (Sinkhorn + sigmoid split + K4 reduction).

    Constructor args:
      hidden_size    : per-HC-copy hidden dim ``H``.
      hc_mult        : HC multiplicity (default 4).
      sinkhorn_iters : Sinkhorn refinement count (default 20).
      prefix         : MPK kernel-tensor name prefix.

    HC weights (``hc_scale``, ``hc_base``) live on the parent block.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int = 4,
        sinkhorn_iters: int = 20,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.sinkhorn_iters = sinkhorn_iters
        self.hc3 = (2 + hc_mult) * hc_mult

    def forward(
        self,
        gemm_out_mul: torch.Tensor,    # [splits, N, hc3] fp32 (or bf16)
        gemm_out_sqrsum: torch.Tensor, # [splits, N] fp32
        residual: torch.Tensor,        # [N, hc, H] bf16
        hc_scale: torch.Tensor,        # [3] fp32
        hc_base: torch.Tensor,         # [hc3] fp32
        rms_eps: float = 1e-6,
        hc_pre_eps: float = 1e-6,
        hc_sinkhorn_eps: float = 1e-6,
        hc_post_mult_value: float = 2.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce splits + Sinkhorn + K2 affine + sigmoid + K4 reduction."""
        N, hc, H = residual.shape
        sqrsum_total = gemm_out_sqrsum.float().sum(0)
        rms = torch.rsqrt(sqrsum_total / (hc * H) + rms_eps)
        mixes = gemm_out_mul.float().sum(0) * rms.unsqueeze(-1)

        pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + hc_pre_eps
        post = hc_post_mult_value * torch.sigmoid(
            mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc]
        )
        cm = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).reshape(N, hc, hc)

        cm = torch.softmax(cm, dim=-1) + hc_sinkhorn_eps
        cm = cm / (cm.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
        for _ in range(self.sinkhorn_iters - 1):
            cm = cm / (cm.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
            cm = cm / (cm.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

        layer_input = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(torch.bfloat16)
        return post.contiguous(), cm.contiguous(), layer_input.contiguous()

    def auto_grid_dim(self, residual_dt: DTensor) -> GridDim:
        """One CTA per token; each CTA processes a full token's hc3 mix."""
        pk = current_pk()
        return (max(1, min(residual_dt.dim(0), pk.num_workers)), 1, 1)

    def compile(
        self,
        gemm_out_mul: DTensor,
        gemm_out_sqrsum: DTensor,
        hc_scale: DTensor,
        hc_base: DTensor,
        residual: DTensor,
        *,
        post_mix: Optional[Union[torch.Tensor, DTensor]] = None,
        comb_mix: Optional[Union[torch.Tensor, DTensor]] = None,
        layer_input: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor, DTensor]:
        """Register one ``mhc_pre_sm100`` task.

        Tensor contract:
          gemm_out_mul    : (splits, N, hc3) fp32 or bf16, broadcast.
          gemm_out_sqrsum : (splits, N) fp32, broadcast.
          hc_scale        : (3,) fp32, broadcast (K2 scale triplet).
          hc_base         : (hc3,) fp32, broadcast (K2 bias).
          residual        : (N, hc, H) bf16, broadcast (CTA picks its row).
          post_mix        : (N, hc) fp32 OUT.
          comb_mix        : (N, hc, hc) or (N, hc*hc) fp32 OUT.
          layer_input     : (N, H) bf16 OUT.

        Notes: each CTA derives ``n`` from ``task_metadata.token_offset``;
        grid is ``(N, 1, 1)``. ``hc3 = (2 + hc_mult) * hc_mult``.
        """
        pk = current_pk()
        assert gemm_out_mul.num_dims == 3
        assert gemm_out_sqrsum.num_dims == 2
        assert hc_scale.num_dims == 1
        assert hc_base.num_dims == 1
        assert residual.num_dims == 3

        N = residual.dim(0)
        hc = residual.dim(1)
        H = residual.dim(2)
        prefix = self.prefix or "mhc_pre_"

        def _attach_out(buf, default_dims, default_dtype, name):
            if buf is None:
                return pk.new_tensor(
                    dims=default_dims, dtype=default_dtype, name=name
                )
            if isinstance(buf, torch.Tensor):
                return pk.attach_input(buf, name=name)
            if isinstance(buf, DTensor):
                return buf
            raise TypeError(f"{name} must be None, torch.Tensor, or DTensor")

        post_dt = _attach_out(
            post_mix, (N, hc), torch.float32, f"{prefix}post_mix"
        )
        comb_dt = _attach_out(
            comb_mix, (N, hc, hc), torch.float32, f"{prefix}comb_mix"
        )
        layer_dt = _attach_out(
            layer_input, (N, H), torch.bfloat16, f"{prefix}layer_input"
        )
        assert post_dt.num_dims == 2
        assert comb_dt.num_dims in (2, 3)
        assert layer_dt.num_dims == 2

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(residual)
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # All inputs/outputs are accessed via task_metadata.token_offset (not
        # via TBGraph partitioning) so all map dims are (-1, -1, -1).
        tb_graph.new_input(gemm_out_mul, (-1, -1, -1), -1, True)
        tb_graph.new_input(gemm_out_sqrsum, (-1, -1, -1), -1, True)
        tb_graph.new_input(hc_scale, (-1, -1, -1), -1, True)
        tb_graph.new_input(hc_base, (-1, -1, -1), -1, True)
        tb_graph.new_input(residual, (-1, -1, -1), -1, True)
        tb_graph.new_input(post_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(comb_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(layer_dt, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual,
             post_dt, comb_dt, layer_dt],
            tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "mhc_pre_sm100")
        return post_dt, comb_dt, layer_dt
