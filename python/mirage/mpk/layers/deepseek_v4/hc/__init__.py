"""DeepSeek V4-Flash Hyper-Connection catalog (subsystem 2).

Eight vLLM-canonical kernels live here, one Python module per spec:

* ``mhc_post.py`` -- NEW (Wave-2A): per-token HC-fused matmul producing
  the next hc-stream residual. Task ``mhc_post_v4_sm100`` at slot 356.
* ``mhc_pre_big_fuse.py`` -- NEW (Wave-2A): split-K-reduce + Sinkhorn +
  weighted sum. Task ``mhc_pre_big_fuse_v4_sm100`` at slot 354.
* ``mhc_pre_big_fuse_with_norm.py`` -- NEW (Wave-2A): same as above but
  with an extra fused output RMSNorm. Slot 355.
* ``mhc_fused.py`` -- NEW (Wave-2A): decode-regime joint mhc_post +
  hc_prenorm_gemm. Slot 360.

* ``hc_prenorm_gemm.py`` -- NEW: per-token GEMM `x @ fn.T` + sum-of-
  squares for the HC pre-block. Task ``hc_prenorm_gemm_v4_sm100`` at
  slot 358.
* ``hc_prenorm_gemm_block_m.py`` -- NEW: same outputs as the
  hc_prenorm_gemm variant; M-blocked tile in the upstream TileLang
  port, identical naive port here. Slot 359.
* ``hc_head_fuse.py`` -- NEW: terminal hc_mult -> 1 collapse with
  sigmoid-gated weighted sum (no post-norm). Slot 357.
* ``tf32_hc_prenorm_gemm.py`` -- NEW: sm_100a-locked DeepGEMM
  alternative (same outputs as hc_prenorm_gemm). Slot 361.

The four GEMM/head modules in this Wave (hc_prenorm_gemm,
hc_prenorm_gemm_block_m, hc_head_fuse, tf32_hc_prenorm_gemm) all share
the same naive backend kernel for the three GEMM siblings; the head
variant is a separate two-pass kernel (sqrsum + sigmoid + weighted sum).
"""
from .hc_prenorm_gemm import V4HcPrenormGemm
from .hc_prenorm_gemm_block_m import V4HcPrenormGemmBlockM
from .hc_head_fuse import V4HcHeadFuse
from .tf32_hc_prenorm_gemm import V4Tf32HcPrenormGemm
# Wave-2A pre/post/fused modules.
from .mhc_post import V4MhcPost
from .mhc_pre_big_fuse import V4MhcPreBigFuse
from .mhc_pre_big_fuse_with_norm import V4MhcPreBigFuseWithNorm
from .mhc_fused import V4MhcFused

__all__ = [
    "V4HcPrenormGemm",
    "V4HcPrenormGemmBlockM",
    "V4HcHeadFuse",
    "V4Tf32HcPrenormGemm",
    "V4MhcPost",
    "V4MhcPreBigFuse",
    "V4MhcPreBigFuseWithNorm",
    "V4MhcFused",
]
