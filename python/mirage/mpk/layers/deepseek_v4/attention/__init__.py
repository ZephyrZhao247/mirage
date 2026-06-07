"""V4-Flash attention subsystem catalog modules.

One module per vLLM-canonical kernel spec at
``docs/mpk/deepseek_v4/vllm_kernels/<kernel>.md``.

This package is assembled by multiple Waves; each wave appends its own
imports + ``__all__`` entries.

Wave-5A attention pre/MLA-core/post entries (this commit):

* :class:`V4FusedQKVRMSNorm`        -- ``fused_q_kv_rmsnorm.md``
* :class:`V4FusedDSV4QNormRopeKVInsert`
      -- ``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.md``
* :class:`V4FlashMLADecode`         -- ``flash_mla_with_kvcache.md``
* :class:`V4FlashMLASparsePrefill`  -- ``flash_mla_sparse_fwd.md``
* :class:`V4FusedInvRopeFP8Quant`   -- ``fused_inv_rope_fp8_quant.md``

Sub-attention cache utilities + o-projection einsum (sibling Waves):

* :class:`V4QuantizeAndInsertK`        -- ``quantize_and_insert_k_kernel.md``
* :class:`V4DequantizeAndGatherK`      -- ``dequantize_and_gather_k_kernel.md``
* :class:`V4ComputeGlobalTopkIndices`
      -- ``compute_global_topk_indices_and_lens.md``
* :class:`V4CombineTopkSwaIndices`     -- ``combine_topk_swa_indices.md``
* :class:`V4DeepseekFP8Einsum`         -- ``deepseek_v4_fp8_einsum.md``
"""
from .fused_q_kv_rmsnorm import V4FusedQKVRMSNorm
from .fused_dsv4_qnorm_rope_kv_insert import V4FusedDSV4QNormRopeKVInsert
from .flash_mla_with_kvcache import V4FlashMLADecode
from .flash_mla_sparse_fwd import V4FlashMLASparsePrefill
from .fused_inv_rope_fp8_quant import V4FusedInvRopeFP8Quant
from .quantize_and_insert_k import V4QuantizeAndInsertK
from .dequantize_and_gather_k import V4DequantizeAndGatherK
from .compute_global_topk_indices import V4ComputeGlobalTopkIndices
from .combine_topk_swa_indices import V4CombineTopkSwaIndices
from .deepseek_v4_fp8_einsum import V4DeepseekFP8Einsum

__all__ = [
    "V4FusedQKVRMSNorm",
    "V4FusedDSV4QNormRopeKVInsert",
    "V4FlashMLADecode",
    "V4FlashMLASparsePrefill",
    "V4FusedInvRopeFP8Quant",
    "V4QuantizeAndInsertK",
    "V4DequantizeAndGatherK",
    "V4ComputeGlobalTopkIndices",
    "V4CombineTopkSwaIndices",
    "V4DeepseekFP8Einsum",
]
