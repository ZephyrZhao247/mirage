"""DeepSeek V4-Flash MoE catalog (subsystems 6 + 7-prep).

Wave-4A scope: four NEW Blackwell naive kernels covering MoE routing,
activation, and the MegaMoE staging step.

* ``topk_softplus_sqrt.py`` -- NEW: dual-branch (USE_HASH true/false)
  top-k routing with sqrt(softplus) scoring, bias-add (scored only),
  optional renormalize, and ``routed_scaling_factor`` scale. Task
  ``topk_softplus_sqrt_v4_sm100`` at slot 380.
* ``dsv3_router_gemm.py`` -- NEW: bf16 in / bf16 weight / fp32 out
  router GEMM. For V4-Flash (H=4096) this is the Tier-3/Tier-4 plain
  matmul. Tier-1 (DSV3 H=7168) and Tier-2 (fp32 H=3072) are noted as
  locked alternative pointers. Task ``dsv3_router_gemm_v4_sm100``
  at slot 381.
* ``silu_and_mul_with_clamp.py`` -- NEW: SwiGLU with one-sided gate
  clamp + two-sided up clamp; bf16 in / bf16 out. The existing
  :class:`mirage.mpk.layers.SiluMul` has NO clamp support so this is
  a new kernel rather than an extension. Task
  ``silu_and_mul_with_clamp_v4_sm100`` at slot 382.
* ``prepare_megamoe_inputs.py`` -- NEW: per-token bf16 -> FP8 E4M3
  quant with UE8M0 packed-int32 group scales (BLOCK_K=128, GROUP_K=32)
  plus int->int64 topk_ids cast and fp32 topk_weights copy. Task
  ``prepare_megamoe_inputs_v4_sm100`` at slot 383.
"""

from .topk_softplus_sqrt import V4TopkSoftplusSqrt
from .dsv3_router_gemm import V4Dsv3RouterGemm
from .silu_and_mul_with_clamp import V4SiluAndMulWithClamp
from .prepare_megamoe_inputs import V4PrepareMegaMoEInputs

__all__ = [
    "V4TopkSoftplusSqrt",
    "V4Dsv3RouterGemm",
    "V4SiluAndMulWithClamp",
    "V4PrepareMegaMoEInputs",
]
