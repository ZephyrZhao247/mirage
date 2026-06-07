"""V4-Flash ``mhc_pre_big_fuse`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/mhc_pre_big_fuse_tilelang.md``.

Decision: **NEW**.

Rationale
---------
The vLLM TileLang kernel completes the tail of ``hc_pre``: it consumes
the (split-K-partial) GEMM outputs ``gemm_out_mul`` and squared-sum
``gemm_out_sqrsum`` and produces three outputs:

  * ``post_mix``  -- sigmoid-gated post coefficients
  * ``comb_mix``  -- Sinkhorn-doubly-stochastic mix matrix
  * ``layer_input`` -- pre-mix-weighted sum across the hc residual streams

There is no existing MPK task that fuses these three steps. The closest
primitives (rmsnorm + linear + softmax + sigmoid) would launch 5-7
separate tasks and miss the Sinkhorn iteration entirely.

Hence: a NEW kernel.
* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/mhc_pre_big_fuse_v4_sm100.cuh``.
* Task name: ``mhc_pre_big_fuse_v4_sm100``.
* Enum slot: ``TASK_MHC_PRE_BIG_FUSE_V4_SM100 = 354``.

Naive Blackwell impl: one CTA per token, NUM_THREADS=256, fp32 reductions,
bf16 stores, sigmoid via ``1/(1+exp(-x))``, Sinkhorn iterations done
sequentially on thread 0 (HC_MULT=4 -> 16 fp32 elements in registers).

Audit
-----
* dtype contract from spec: ``gemm_out_mul``/``gemm_out_sqrsum`` fp32;
  ``hc_scale``/``hc_base`` fp32; ``residual`` bf16; outputs:
  ``post_mix``/``comb_mix`` fp32, ``layer_input`` bf16. The producer of
  ``gemm_out_*`` is ``hc_prenorm_gemm_*`` (Wave-2B) -- both producer and
  this consumer agree on fp32, matching the spec and avoiding the silent
  type-pun flagged in the prior audit.
* layout: row-major contiguous everywhere.
* multi-batch: partitioned on token dim; multi-batch from day 1.
* ``forward()``: faithful PyTorch reference matching the spec's Math
  section -- including the Sinkhorn priming step (row-softmax+eps,
  col-normalize) followed by ``sinkhorn_repeat - 1`` plain (row, col)
  normalizations.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor


__all__ = ["V4MhcPreBigFuse"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


def _sinkhorn(cm: torch.Tensor, eps: float, repeat: int) -> torch.Tensor:
    """Reference Sinkhorn matching the spec (fp32, (T, HC, HC))."""
    # priming: row-softmax + eps, then col-normalize
    cm = torch.softmax(cm, dim=-1) + eps
    cm = cm / (cm.sum(dim=-2, keepdim=True) + eps)
    for _ in range(repeat - 1):
        cm = cm / (cm.sum(dim=-1, keepdim=True) + eps)
        cm = cm / (cm.sum(dim=-2, keepdim=True) + eps)
    return cm


class V4MhcPreBigFuse(MPKModule):
    """V4-Flash mhc_pre_big_fuse (no fused RMSNorm gamma).

    Tensor contract:
      gemm_out_mul     : (N_SPLITS, T, HC_MULT3) fp32
      gemm_out_sqrsum  : (N_SPLITS, T)           fp32
      hc_scale         : (3,)                    fp32 (broadcast)
      hc_base          : (HC_MULT3,)             fp32 (broadcast)
      residual         : (T, HC_MULT, HIDDEN)    bf16
      ---
      post_mix         : (T, HC_MULT)            fp32
      comb_mix         : (T, HC_MULT*HC_MULT)    fp32
      layer_input      : (T, HIDDEN)             bf16

    Constants baked into codegen (matching V4-Flash):
      rms_eps         = 1e-6
      hc_pre_eps      = 1e-6
      hc_sinkhorn_eps = 1e-6
      hc_post_alpha   = 2.0
      sinkhorn_repeat = 20
    """

    HC_POST_ALPHA = 2.0
    HC_PRE_EPS = 1e-6
    HC_SINKHORN_EPS = 1e-6
    RMS_EPS = 1e-6
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
                f"V4MhcPreBigFuse: positive sizes required; got "
                f"hidden_size={hidden_size}, hc_mult={hc_mult}"
            )
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.hc_mult3 = hc_mult * (2 + hc_mult)

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

        assert gemm_out_mul.shape[1] == T
        assert gemm_out_mul.shape[2] == self.hc_mult3
        assert gemm_out_sqrsum.shape[1] == T
        assert hc_scale.shape == (3,)
        assert hc_base.shape == (self.hc_mult3,)
        assert residual.shape == (T, HC, H)

        residual_dtype = residual.dtype

        # 1. Split-K reduce + rsqrt-RMSNorm
        sq = gemm_out_sqrsum.to(torch.float32).sum(dim=0)            # (T,)
        rsqrt = torch.rsqrt(sq / float(HC * H) + self.RMS_EPS)       # (T,)
        mixes = gemm_out_mul.to(torch.float32).sum(dim=0) * \
            rsqrt.unsqueeze(-1)                                     # (T, HC_MULT3)

        # 2. Partition
        pre_logits = mixes[:, :HC]                                  # (T, HC)
        post_logits = mixes[:, HC:2 * HC]                            # (T, HC)
        comb_logits = mixes[:, 2 * HC:].view(T, HC, HC)              # (T, HC, HC)

        scale = hc_scale.to(torch.float32)
        base = hc_base.to(torch.float32)

        # 3. post_mix
        post_mix = torch.sigmoid(
            post_logits * scale[1] + base[HC:2 * HC]
        ) * self.HC_POST_ALPHA

        # 4. Sinkhorn comb_mix
        cm = comb_logits * scale[2] + base[2 * HC:].view(HC, HC)
        cm = _sinkhorn(cm, self.HC_SINKHORN_EPS, self.SINKHORN_REPEAT)
        comb_mix = cm.reshape(T, HC * HC)

        # 5. pre_mix + weighted sum, bf16 store (no RMSNorm gamma)
        pre_mix = torch.sigmoid(
            pre_logits * scale[0] + base[:HC]
        ) + self.HC_PRE_EPS                                          # (T, HC)
        layer_input = (
            pre_mix.unsqueeze(-1) * residual.to(torch.float32)
        ).sum(dim=1).to(residual_dtype)                              # (T, H)

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

        # Shape checks.
        if gemm_out_mul.num_dims != 3:
            raise ValueError(
                f"V4MhcPreBigFuse: gemm_out_mul must be 3-D "
                f"(N_SPLITS, T, HC_MULT3); got num_dims={gemm_out_mul.num_dims}"
            )
        if gemm_out_mul.dim(2) != self.hc_mult3:
            raise ValueError(
                f"V4MhcPreBigFuse: gemm_out_mul.dim(2)={gemm_out_mul.dim(2)} "
                f"!= HC_MULT3={self.hc_mult3}"
            )
        if gemm_out_sqrsum.num_dims != 2:
            raise ValueError(
                f"V4MhcPreBigFuse: gemm_out_sqrsum must be 2-D "
                f"(N_SPLITS, T); got num_dims={gemm_out_sqrsum.num_dims}"
            )
        if residual.num_dims != 3 or residual.dim(1) != self.hc_mult \
                or residual.dim(2) != self.hidden_size:
            raise ValueError(
                f"V4MhcPreBigFuse: residual must be (T, {self.hc_mult}, "
                f"{self.hidden_size})"
            )
        T = residual.dim(0)
        if gemm_out_mul.dim(1) != T or gemm_out_sqrsum.dim(1) != T:
            raise ValueError(
                f"V4MhcPreBigFuse: gemm_out_* dim 1 ({gemm_out_mul.dim(1)}, "
                f"{gemm_out_sqrsum.dim(1)}) must equal num_tokens ({T})"
            )
        if hc_scale.num_dims != 1 or hc_scale.dim(0) != 3:
            raise ValueError(
                f"V4MhcPreBigFuse: hc_scale must be (3,); got "
                f"shape ({hc_scale.dim(0)},)"
            )
        if hc_base.num_dims != 1 or hc_base.dim(0) != self.hc_mult3:
            raise ValueError(
                f"V4MhcPreBigFuse: hc_base must be ({self.hc_mult3},); "
                f"got shape ({hc_base.dim(0)},)"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(residual)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # ----- allocate outputs ------------------------------------------
        def _resolve(out, dims, dtype, label):
            if out is None:
                return pk.new_tensor(
                    dims=dims, dtype=dtype,
                    name=f"{self.prefix}mhc_pre_{label}",
                )
            if isinstance(out, torch.Tensor):
                return pk.attach_input(
                    out, name=f"{self.prefix}mhc_pre_{label}"
                )
            if isinstance(out, DTensor):
                return out
            raise TypeError(
                f"V4MhcPreBigFuse.compile {label} must be None/Tensor/DTensor"
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
        # Token-keyed inputs partition on dim 1 (gemm_out_*: N_SPLITS-major)
        # or dim 0 (residual: T-major).
        tb_graph.new_input(gemm_out_mul, (1, -1, -1), 1, True)
        tb_graph.new_input(gemm_out_sqrsum, (1, -1, -1), 1, True)
        # hc_scale / hc_base: pure broadcast, no partition.
        tb_graph.new_input(hc_scale, (-1, -1, -1), 0, True)
        tb_graph.new_input(hc_base, (-1, -1, -1), 0, True)
        tb_graph.new_input(residual, (0, -1, -1), 1, True)
        # Outputs: token-partitioned on dim 0.
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
                post_mix_dt,
                comb_mix_dt,
                layer_input_dt,
            ],
            tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "mhc_pre_big_fuse_v4_sm100")

        return post_mix_dt, comb_mix_dt, layer_input_dt
