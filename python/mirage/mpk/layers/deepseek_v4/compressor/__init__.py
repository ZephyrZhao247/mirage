"""DeepSeek V4-Flash Compressor catalog (subsystem 4).

Two vLLM-canonical kernels in this Wave (Compressor):

* ``save_partial_states.py`` -- NEW: per-token write of the (kv | score
  + ape) state cache. Task ``save_partial_states_v4_sm100`` at slot
  ``TASK_SAVE_PARTIAL_STATES_V4_SM100 = 372``.
* ``fused_kv_compress_norm_rope_insert_sparse_attn.py`` -- NEW: per-
  boundary-token sparse-attention compressor (head_dim=512). Gather
  window + softmax + weighted sum + RMSNorm + UE8M0 block-FP8 quant +
  GPT-J RoPE on the rope tail. Task
  ``fused_kv_compress_norm_rope_insert_sparse_attn_v4_sm100`` at slot
  ``TASK_FUSED_KV_COMPRESS_NORM_ROPE_INSERT_SPARSE_ATTN_V4_SM100 = 373``.

Both kernels use **compressor RoPE** with ``compress_rope_theta=160000``
(NOT the main attention RoPE's 10000). The caller MUST pass a
``cos_sin_cache`` built against the 160000 base for correctness.
"""
from .save_partial_states import V4SavePartialStates
from .fused_kv_compress_norm_rope_insert_sparse_attn import (
    V4FusedKVCompressNormRopeInsertSparseAttn,
)

__all__ = [
    "V4SavePartialStates",
    "V4FusedKVCompressNormRopeInsertSparseAttn",
]
