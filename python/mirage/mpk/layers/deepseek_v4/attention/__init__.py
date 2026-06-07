"""V4-Flash attention subsystem catalog modules.

One module per vLLM-canonical kernel spec at
``docs/mpk/deepseek_v4/vllm_kernels/<kernel>.md``.

This package is assembled by multiple Waves; each wave appends its own
imports + ``__all__`` entries. Sub-attention cache utilities +
o-projection einsum live here:

* :class:`V4QuantizeAndInsertK`        -- ``quantize_and_insert_k_kernel.md``
* :class:`V4DequantizeAndGatherK`      -- ``dequantize_and_gather_k_kernel.md``
* :class:`V4ComputeGlobalTopkIndices`
      -- ``compute_global_topk_indices_and_lens.md``
* :class:`V4CombineTopkSwaIndices`     -- ``combine_topk_swa_indices.md``
* :class:`V4DeepseekFP8Einsum`         -- ``deepseek_v4_fp8_einsum.md``
"""
from .quantize_and_insert_k import V4QuantizeAndInsertK
from .dequantize_and_gather_k import V4DequantizeAndGatherK
from .compute_global_topk_indices import V4ComputeGlobalTopkIndices
from .combine_topk_swa_indices import V4CombineTopkSwaIndices
from .deepseek_v4_fp8_einsum import V4DeepseekFP8Einsum

__all__ = [
    "V4QuantizeAndInsertK",
    "V4DequantizeAndGatherK",
    "V4ComputeGlobalTopkIndices",
    "V4CombineTopkSwaIndices",
    "V4DeepseekFP8Einsum",
]
