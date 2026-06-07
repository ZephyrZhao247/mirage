"""DeepSeek V4-Flash std-layer + sampling catalog.

Four vLLM-canonical kernels live here, one Python module per spec:

* ``rms_norm.py`` — alias for :class:`mirage.mpk.layers.RMSNorm`.
  V4 enum slot ``TASK_RMS_NORM_V4_SM100 = 350`` is RESERVED but unused
  (the existing ``rmsnorm_hopper`` task at ``TASK_RMS_NORM = 119`` covers
  the contract bit-for-bit; see the module docstring for the audit).
* ``apply_rotary_emb.py`` — NEW: naive bf16 GPT-J-interleaved RoPE.
  Registered as task ``apply_rotary_emb_v4_sm100`` at enum slot 351.
* ``vocab_parallel_embedding.py`` — alias for
  :class:`mirage.mpk.layers.Embed`.
  V4 enum slot ``TASK_VOCAB_PARALLEL_EMBEDDING_V4_SM100 = 352`` is
  RESERVED but unused (the existing ``embedding`` task covers the
  ``tp_size=1`` contract V4 ships).
* ``logits_processor.py`` — alias for :class:`mirage.mpk.layers.Linear`
  (the lm_head GEMM is the only on-path kernel; soft_cap and scale are
  inactive). V4 enum slot ``TASK_LOGITS_PROCESSOR_V4_SM100 = 353`` is
  RESERVED but unused.
"""

from .rms_norm import V4RMSNorm
from .apply_rotary_emb import V4ApplyRotaryEmb
from .vocab_parallel_embedding import V4VocabParallelEmbedding
from .logits_processor import V4LogitsProcessor

__all__ = [
    "V4RMSNorm",
    "V4ApplyRotaryEmb",
    "V4VocabParallelEmbedding",
    "V4LogitsProcessor",
]
