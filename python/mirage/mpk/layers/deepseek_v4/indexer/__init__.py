"""DeepSeek V4-Flash Indexer catalog (subsystem 5).

Four vLLM-canonical kernels in this Wave (Indexer Q-side + MQA logits):

* ``fused_indexer_q_rope_quant.py`` -- NEW: FP8 Q-side rope + quant.
  Task ``fused_indexer_q_rope_quant_v4_sm100`` at slot 376. Class-B
  sibling with the MXFP4 variant; pairs with the FP8 K-side compressor
  via ``use_fp4_cache=False``.
* ``fused_indexer_q_rope_mxfp4.py`` -- NEW: MXFP4 Q-side rope + block-
  scaled quant. Task ``fused_indexer_q_rope_mxfp4_v4_sm100`` at slot
  377. Class-B sibling with FP8; pairs with the MXFP4 K-side compressor
  via ``use_fp4_cache=True``.
* ``fp8_fp4_paged_mqa_logits.py`` -- NEW: paged-MQA logits (FP8 path).
  Task ``fp8_fp4_paged_mqa_logits_v4_sm100`` at slot 378. Decode-time
  consumer of the (Q, K) indexer outputs.
* ``fp8_fp4_mqa_logits.py`` -- NEW: non-paged MQA logits (FP8 path,
  prefill). Task ``fp8_fp4_mqa_logits_v4_sm100`` at slot 379.

The two Q-side kernels are NAIVE single-CTA-per-(token,head) per the
"naive first, perf later" rule; their UMMA/TMA fast paths in vLLM are
the CuteDSL variants `IndexerQFp8Kernel` / `IndexerQMxFp4Kernel` and
are out of scope here. The two MQA-logits kernels are NAIVE one-CTA-
per-Q-row (looping over kv); the production DeepGEMM kernel uses a
persistent-grid + TMA + UMMA pipeline with warp specialization.
"""
from .fused_kv_compress_norm_rope_insert_indexer_attn import (
    V4FusedKVCompressNormRopeInsertIndexerAttn,
)
from .fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn import (
    V4FusedKVCompressNormRopeInsertIndexerMxfp4Attn,
)
from .fused_indexer_q_rope_quant import V4FusedIndexerQRopeQuant
from .fused_indexer_q_rope_mxfp4 import V4FusedIndexerQRopeMxfp4
from .fp8_fp4_paged_mqa_logits import V4Fp8Fp4PagedMqaLogits
from .fp8_fp4_mqa_logits import V4Fp8Fp4MqaLogits

__all__ = [
    "V4FusedKVCompressNormRopeInsertIndexerAttn",
    "V4FusedKVCompressNormRopeInsertIndexerMxfp4Attn",
    "V4FusedIndexerQRopeQuant",
    "V4FusedIndexerQRopeMxfp4",
    "V4Fp8Fp4PagedMqaLogits",
    "V4Fp8Fp4MqaLogits",
]
