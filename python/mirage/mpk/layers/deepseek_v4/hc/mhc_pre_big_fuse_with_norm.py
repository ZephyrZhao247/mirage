"""V4-Flash ``mhc_pre_big_fuse_with_norm`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/mhc_pre_big_fuse_with_norm_tilelang.md``.

Decision: **NEW**.

Rationale
---------
This is the V4-Flash **canonical** ``hc_pre`` path -- every decoder layer
hits this kernel via ``attn_norm.weight`` / ``ffn_norm.weight`` being
non-None. It extends :class:`V4MhcPreBigFuse` by fusing a second-pass
RMSNorm (with learnable ``norm_weight``) on top of the pre-mix weighted
sum: the bf16-rounded ``y`` is squared-summed in-kernel, an
``rsqrt_norm`` is computed, then ``layer_input = bf16(y * rsqrt_norm *
norm_weight)``.

Because the second-pass denominator differs from
``gemm_out_sqrsum / (HC * H)`` (the latter is residual.flatten.pow(2)
summed; the former is ``y``'s squared sum), we recompute it from the
in-flight ``y`` -- exactly matching the spec.

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/mhc_pre_big_fuse_with_norm_v4_sm100.cuh``.
* Task name: ``mhc_pre_big_fuse_with_norm_v4_sm100``.
* Enum slot: ``TASK_MHC_PRE_BIG_FUSE_WITH_NORM_V4_SM100 = 355``.

Audit
-----
* dtype: ``norm_weight`` bf16. Other dtypes match the no-norm variant.
* layout: row-major contiguous everywhere.
* multi-batch: partitioned on token dim; multi-batch from day 1.
* ``forward()``: faithful PyTorch reference -- includes the bf16
  round-trip on ``y`` before the second-pass sumsq, exactly as the spec
  pins for numerical equivalence with ``attn_norm(y.to(bf16))`` reference.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor

from .mhc_pre_big_fuse import _sinkhorn


__all__ = ["V4MhcPreBigFuseWithNorm"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class V4MhcPreBigFuseWithNorm(MPKModule):
    """V4-Flash mhc_pre_big_fuse_with_norm.

    Owns one ``nn.Parameter`` weight:
      * ``norm_weight: [hidden_size]`` bf16 (RMSNorm gamma)

    Tensor contract:
      gemm_out_mul     : (N_SPLITS, T, HC_MULT3) fp32
      gemm_out_sqrsum  : (N_SPLITS, T)           fp32
      hc_scale         : (3,)                    fp32 (broadcast)
      hc_base          : (HC_MULT3,)             fp32 (broadcast)
      residual         : (T, HC_MULT, HIDDEN)    bf16
      norm_weight (auto): (HIDDEN,)              bf16
      ---
      post_mix         : (T, HC_MULT)            fp32
      comb_mix         : (T, HC_MULT*HC_MULT)    fp32
      layer_input      : (T, HIDDEN)             bf16 (RMSNorm'd)

    Constants baked into codegen (matching V4-Flash):
      rms_eps         = 1e-6
      hc_pre_eps      = 1e-6
      hc_sinkhorn_eps = 1e-6
      hc_post_alpha   = 2.0
      norm_eps        = 1e-6
      sinkhorn_repeat = 20
    """

    HC_POST_ALPHA = 2.0
    HC_PRE_EPS = 1e-6
    HC_SINKHORN_EPS = 1e-6
    RMS_EPS = 1e-6
    NORM_EPS = 1e-6
    SINKHORN_REPEAT = 20

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if hidden_size <= 0 or hc_mult <= 0:
            raise ValueError(
                f"V4MhcPreBigFuseWithNorm: positive sizes required"
            )
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.hc_mult3 = hc_mult * (2 + hc_mult)
        self.norm_weight = nn.Parameter(torch.ones(hidden_size))

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        gemm_out_mul: torch.Tensor,
        gemm_out_sqrsum: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T = residual.shape[0]
        HC = self.hc_mult
        H = self.hidden_size

        residual_dtype = residual.dtype

        # 1. Split-K reduce + rsqrt-RMSNorm.
        sq = gemm_out_sqrsum.to(torch.float32).sum(dim=0)
        rsqrt = torch.rsqrt(sq / float(HC * H) + self.RMS_EPS)
        mixes = gemm_out_mul.to(torch.float32).sum(dim=0) * \
            rsqrt.unsqueeze(-1)

        pre_logits = mixes[:, :HC]
        post_logits = mixes[:, HC:2 * HC]
        comb_logits = mixes[:, 2 * HC:].view(T, HC, HC)

        scale = hc_scale.to(torch.float32)
        base = hc_base.to(torch.float32)

        post_mix = torch.sigmoid(
            post_logits * scale[1] + base[HC:2 * HC]
        ) * self.HC_POST_ALPHA

        cm = comb_logits * scale[2] + base[2 * HC:].view(HC, HC)
        cm = _sinkhorn(cm, self.HC_SINKHORN_EPS, self.SINKHORN_REPEAT)
        comb_mix = cm.reshape(T, HC * HC)

        pre_mix = torch.sigmoid(
            pre_logits * scale[0] + base[:HC]
        ) + self.HC_PRE_EPS

        # Pass 1: weighted sum, bf16 round, accumulate sumsq for pass 2.
        y_fp32 = (
            pre_mix.unsqueeze(-1) * residual.to(torch.float32)
        ).sum(dim=1)                                                # (T, H)
        y_bf16 = y_fp32.to(torch.bfloat16)
        # Pass 2: recompute sumsq from y_bf16 (NOT the fp32 y).
        sumsq_y = y_bf16.to(torch.float32).pow(2).sum(dim=-1)        # (T,)
        rsqrt_norm = torch.rsqrt(sumsq_y / float(H) + self.NORM_EPS)

        nw = self.norm_weight.to(torch.float32)
        layer_input = (
            y_bf16.to(torch.float32) * rsqrt_norm.unsqueeze(-1) * nw
        ).to(residual_dtype)

        return post_mix, comb_mix, layer_input

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, residual_dt: DTensor) -> GridDim:
        pk = current_pk()
        num_tokens = residual_dt.dim(0)
        return (max(1, min(num_tokens, pk.num_workers)), 1, 1)

    # ------------------------------------------------------------------
    # MPK registration
    # ------------------------------------------------------------------
    def compile(
        self,
        gemm_out_mul: DTensor,
        gemm_out_sqrsum: DTensor,
        hc_scale: DTensor,
        hc_base: DTensor,
        residual: DTensor,
        *,
        post_mix: Optional[torch.Tensor] = None,
        comb_mix: Optional[torch.Tensor] = None,
        layer_input: Optional[torch.Tensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor, DTensor]:
        pk = current_pk()

        # ----- shape validation (same as no-norm variant) -----------------
        if gemm_out_mul.num_dims != 3 \
                or gemm_out_mul.dim(2) != self.hc_mult3:
            raise ValueError(
                f"V4MhcPreBigFuseWithNorm: gemm_out_mul must be "
                f"(N_SPLITS, T, {self.hc_mult3})"
            )
        if gemm_out_sqrsum.num_dims != 2:
            raise ValueError(
                "V4MhcPreBigFuseWithNorm: gemm_out_sqrsum must be 2-D"
            )
        if residual.num_dims != 3 or residual.dim(1) != self.hc_mult \
                or residual.dim(2) != self.hidden_size:
            raise ValueError(
                f"V4MhcPreBigFuseWithNorm: residual must be (T, "
                f"{self.hc_mult}, {self.hidden_size})"
            )
        T = residual.dim(0)
        if hc_scale.num_dims != 1 or hc_scale.dim(0) != 3:
            raise ValueError(
                "V4MhcPreBigFuseWithNorm: hc_scale must be (3,)"
            )
        if hc_base.num_dims != 1 or hc_base.dim(0) != self.hc_mult3:
            raise ValueError(
                f"V4MhcPreBigFuseWithNorm: hc_base must be ({self.hc_mult3},)"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(residual)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # Attach the norm_weight as an MPK input.
        norm_w_dt = pk.attach_input(
            self.norm_weight.data, name=f"{self.prefix}norm_weight"
        )

        # ----- allocate outputs ------------------------------------------
        def _resolve(out, dims, dtype, label):
            if out is None:
                return pk.new_tensor(
                    dims=dims, dtype=dtype,
                    name=f"{self.prefix}mhc_pre_n_{label}",
                )
            if isinstance(out, torch.Tensor):
                return pk.attach_input(
                    out, name=f"{self.prefix}mhc_pre_n_{label}"
                )
            if isinstance(out, DTensor):
                return out
            raise TypeError(
                f"V4MhcPreBigFuseWithNorm.compile {label} must be "
                f"None/Tensor/DTensor"
            )

        from .....core import float32, bfloat16
        post_mix_dt = _resolve(
            post_mix, (T, self.hc_mult), float32, "post_mix",
        )
        comb_mix_dt = _resolve(
            comb_mix, (T, self.hc_mult * self.hc_mult), float32, "comb_mix",
        )
        layer_input_dt = _resolve(
            layer_input, (T, self.hidden_size), bfloat16, "layer_input",
        )

        # ----- TBGraph -------------------------------------------------
        from .....core import CyTBGraph
        from .....kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(gemm_out_mul, (1, -1, -1), 1, True)
        tb_graph.new_input(gemm_out_sqrsum, (1, -1, -1), 1, True)
        tb_graph.new_input(hc_scale, (-1, -1, -1), 0, True)
        tb_graph.new_input(hc_base, (-1, -1, -1), 0, True)
        tb_graph.new_input(residual, (0, -1, -1), 1, True)
        tb_graph.new_input(norm_w_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(post_mix_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(comb_mix_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(layer_input_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [
                gemm_out_mul,
                gemm_out_sqrsum,
                hc_scale,
                hc_base,
                residual,
                norm_w_dt,
                post_mix_dt,
                comb_mix_dt,
                layer_input_dt,
            ],
            tb_graph,
        )
        pk.kn_graph.register_task(
            tb_graph, "mhc_pre_big_fuse_with_norm_v4_sm100"
        )

        return post_mix_dt, comb_mix_dt, layer_input_dt
