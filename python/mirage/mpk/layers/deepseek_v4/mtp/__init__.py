"""DeepSeek V4-Flash MTP catalog (subsystem 8).

Two vLLM-canonical kernels live here, one Python module per spec:

* ``fused_mtp_input_rmsnorm.py`` -- NEW: naive Blackwell joint MTP
  input RMSNorm (enorm + per-slot hnorm with pos==0 mask). Registered
  as task ``fused_mtp_input_rmsnorm_v4_sm100`` at enum slot
  ``TASK_FUSED_MTP_INPUT_RMSNORM_V4_SM100 = 389``.
* ``mtp_shared_head_rmsnorm.py`` -- REUSE: alias for
  :class:`mirage.mpk.layers.RMSNorm`. The vLLM kernel is a plain
  per-token RMSNorm with bf16 in/out and fp32 reduction; the existing
  ``rmsnorm_hopper`` task (covers SM90 and SM100) matches the contract
  bit-for-bit. V4 enum slot
  ``TASK_MTP_SHARED_HEAD_RMSNORM_V4_SM100 = 390`` is RESERVED but
  UNUSED.
"""

from .fused_mtp_input_rmsnorm import V4FusedMTPInputRMSNorm
from .mtp_shared_head_rmsnorm import V4MTPSharedHeadRMSNorm

__all__ = [
    "V4FusedMTPInputRMSNorm",
    "V4MTPSharedHeadRMSNorm",
]
