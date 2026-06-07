// Ampere task impls
#include "tasks/ampere/embedding.cuh"
#include "tasks/ampere/identity.cuh"
#include "tasks/ampere/merge_splitkv.cuh"
#include "tasks/ampere/multitoken_paged_attention_split_kv.cuh"
#include "tasks/ampere/silu_mul.cuh"
#include "tasks/ampere/single_batch_extend.cuh"
#ifdef USE_NVSHMEM
#include "tasks/ampere/allreduce.cuh"
#endif // USE_NVSHMEM
// Hopper task impls
#include "tasks/cute/hopper/gemm_ws.cuh"
#include "tasks/cute/hopper/gemm_ws_cooperative.cuh"
#include "tasks/cute/hopper/gemm_ws_mpk.cuh"
#include "tasks/hopper/linear_hopper.cuh"
#include "tasks/hopper/linear_swapAB_hopper.cuh"
#include "tasks/hopper/multitoken_paged_attention_hopper.cuh"
#include "tasks/hopper/rmsnorm_hopper.cuh"
#include "tasks/hopper/rotary_embedding_hopper.cuh"
#include "tasks/hopper/silu_mul_hopper.cuh"
#if defined(USE_NVSHMEM) && !defined(MIRAGE_GRACE_BLACKWELL)
#include "tasks/hopper/allreduce.cuh"
#endif
// Blackwell task impls
#if defined(USE_NVSHMEM) && defined(MIRAGE_GRACE_BLACKWELL)
#include "tasks/blackwell/allreduce.cuh"
#endif
#include "apply_rotary_emb_v4_sm100.cuh"
#include "argmax_sm100.cuh"
#if defined(USE_NVSHMEM) && defined(MIRAGE_GRACE_BLACKWELL)
#include "nvshmem_argmax_sm100.cuh"
#endif
#include "attention_sm100.cuh"
#include "deepseek_mla_rope_sm100.cuh"
#include "fp8_gemm_dense_decode_splitk_sm100.cuh"
#include "fp8_gemm_dense_fp8out_sm100.cuh"
#include "fp8_gemm_dense_mediumm_sm100.cuh"
#include "fp8_gemm_dense_smallm_sm100.cuh"
#include "fp8_group_gemm_largem_compact_sm100.cuh"
#include "fp8_group_gemm_largem_sm100.cuh"
#include "fp8_group_gemm_sm100.cuh"
#include "fp8_group_gemm_smallm_sm100.cuh"
#include "fused_mtp_input_rmsnorm_v4_sm100.cuh"
#include "fused_rmsnorm_quantize_fp8_sm100.cuh"
#include "linear_fp8_bmm_dense_sm100.cuh"
#include "linear_fp8_bmm_sm100.cuh"
#include "linear_fp8_sm100.cuh"
#include "linear_fp8_swapAB_sm100.cuh"
#include "linear_sm100_mpk.cuh"
#include "mla_dispatch_sm100.cuh"
#include "mla_kv_cache_gather_sm100.cuh"
#include "mla_kv_cache_gather_split_sm100.cuh"
// sm100_ptx.cuh must be included BEFORE mla_mtp_decode_sm100.cuh at top level
// so kernel::sm100_ptx is defined in the correct namespace
#include "assemble_q_decode_sm100.cuh"
#include "elementwise_add_sm100.cuh"
#include "mla_mtp_decode_sm100.cuh"
#include "mla_mtp_decode_tp2_sm100.cuh"
#include "mla_mtp_decode_tp4_sm100.cuh"
#include "mla_mtp_decode_tp8_sm100.cuh"
#include "mla_prefill_sm100.cuh"
#include "mla_prefill_tp8_chunked_sm100.cuh"
#include "mla_prefill_tp8_chunked_splitk_sm100.cuh"
#include "mla_prefill_tp8_sm100.cuh"
#include "mla_reduce_sm100.cuh"
#include "mla_sm100_2sm.cuh"
#include "mla_unified_sm100.cuh"
// V4-Flash HC kernels (naive Blackwell). Specs under
// docs/mpk/deepseek_v4/vllm_kernels/{mhc_post,mhc_pre_big_fuse,
// mhc_pre_big_fuse_with_norm,mhc_fused}_tilelang.md. The
// with_norm header includes the no-norm header (sinkhorn_inplace),
// so order is: base first, then derived.
#include "mhc_pre_big_fuse_v4_sm100.cuh"
#include "mhc_pre_big_fuse_with_norm_v4_sm100.cuh"
#include "mhc_post_v4_sm100.cuh"
#include "mhc_fused_v4_sm100.cuh"
// V4-Flash HC GEMM + head kernels (naive Blackwell). The shared GEMM
// impl header (hc_prenorm_gemm_v4_sm100.cuh) must precede the two thin
// variants that delegate to it (block_m, tf32).
#include "hc_prenorm_gemm_v4_sm100.cuh"
#include "hc_prenorm_gemm_block_m_v4_sm100.cuh"
#include "hc_head_fuse_v4_sm100.cuh"
#include "tf32_hc_prenorm_gemm_v4_sm100.cuh"
// V4-Flash attention kernels (naive Blackwell). Specs:
//   docs/mpk/deepseek_v4/vllm_kernels/fused_q_kv_rmsnorm.md
//   docs/mpk/deepseek_v4/vllm_kernels/fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.md
//   docs/mpk/deepseek_v4/vllm_kernels/flash_mla_with_kvcache.md
//   docs/mpk/deepseek_v4/vllm_kernels/flash_mla_sparse_fwd.md
//   docs/mpk/deepseek_v4/vllm_kernels/fused_inv_rope_fp8_quant.md
#include "fused_q_kv_rmsnorm_v4_sm100.cuh"
#include "fused_dsv4_qnorm_rope_kv_insert_v4_sm100.cuh"
#include "flash_mla_with_kvcache_v4_sm100.cuh"
#include "flash_mla_sparse_fwd_v4_sm100.cuh"
#include "fused_inv_rope_fp8_quant_v4_sm100.cuh"
// V4-Flash attention cache utilities + o-projection einsum (naive Blackwell).
// Specs:
//   docs/mpk/deepseek_v4/vllm_kernels/quantize_and_insert_k_kernel.md
//   docs/mpk/deepseek_v4/vllm_kernels/dequantize_and_gather_k_kernel.md
//   docs/mpk/deepseek_v4/vllm_kernels/compute_global_topk_indices_and_lens.md
//   docs/mpk/deepseek_v4/vllm_kernels/combine_topk_swa_indices.md
//   docs/mpk/deepseek_v4/vllm_kernels/deepseek_v4_fp8_einsum.md
#include "quantize_and_insert_k_v4_sm100.cuh"
#include "dequantize_and_gather_k_v4_sm100.cuh"
#include "compute_global_topk_indices_v4_sm100.cuh"
#include "combine_topk_swa_indices_v4_sm100.cuh"
#include "deepseek_v4_fp8_einsum_v4_sm100.cuh"
// V4-Flash Compressor + Indexer K-side kernels (naive Blackwell). Specs:
//   docs/mpk/deepseek_v4/vllm_kernels/save_partial_states.md
//   docs/mpk/deepseek_v4/vllm_kernels/fused_kv_compress_norm_rope_insert_sparse_attn.md
//   docs/mpk/deepseek_v4/vllm_kernels/fused_kv_compress_norm_rope_insert_indexer_attn.md
//   docs/mpk/deepseek_v4/vllm_kernels/fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn.md
#include "save_partial_states_v4_sm100.cuh"
#include "fused_kv_compress_norm_rope_insert_sparse_attn_v4_sm100.cuh"
#include "fused_kv_compress_norm_rope_insert_indexer_attn_v4_sm100.cuh"
#include "fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn_v4_sm100.cuh"
// V4-Flash indexer Q-side + MQA-logits kernels (naive Blackwell). Specs:
//   docs/mpk/deepseek_v4/vllm_kernels/fused_indexer_q_rope_quant.md
//   docs/mpk/deepseek_v4/vllm_kernels/fused_indexer_q_rope_mxfp4.md
//   docs/mpk/deepseek_v4/vllm_kernels/fp8_fp4_paged_mqa_logits.md
//   docs/mpk/deepseek_v4/vllm_kernels/fp8_fp4_mqa_logits.md
#include "fused_indexer_q_rope_quant_v4_sm100.cuh"
#include "fused_indexer_q_rope_mxfp4_v4_sm100.cuh"
#include "fp8_fp4_paged_mqa_logits_v4_sm100.cuh"
#include "fp8_fp4_mqa_logits_v4_sm100.cuh"
// V4-Flash MoE routing + activation + MegaMoE-prep kernels (naive Blackwell).
// Specs:
//   docs/mpk/deepseek_v4/vllm_kernels/topk_softplus_sqrt.md
//   docs/mpk/deepseek_v4/vllm_kernels/dsv3_router_gemm.md
//   docs/mpk/deepseek_v4/vllm_kernels/silu_and_mul_with_clamp.md
//   docs/mpk/deepseek_v4/vllm_kernels/prepare_megamoe_inputs.md
#include "topk_softplus_sqrt_v4_sm100.cuh"
#include "dsv3_router_gemm_v4_sm100.cuh"
#include "silu_and_mul_with_clamp_v4_sm100.cuh"
#include "prepare_megamoe_inputs_v4_sm100.cuh"
// V4-Flash MoE compute kernels (naive Blackwell). Specs:
//   docs/mpk/deepseek_v4/vllm_kernels/fp8_fp4_mega_moe.md
//   docs/mpk/deepseek_v4/vllm_kernels/fused_moe_kernel.md
//   docs/mpk/deepseek_v4/vllm_kernels/fused_moe_kernel_gptq_awq.md
//   docs/mpk/deepseek_v4/vllm_kernels/write_zeros_to_output.md
//   docs/mpk/deepseek_v4/vllm_kernels/moe_align_block_size.md
#include "fp8_fp4_mega_moe_v4_sm100.cuh"
#include "fused_moe_kernel_v4_sm100.cuh"
#include "fused_moe_kernel_gptq_awq_v4_sm100.cuh"
#include "write_zeros_to_output_v4_sm100.cuh"
#include "moe_align_block_size_v4_sm100.cuh"
#include "moe_linear_sm100.cuh"
#include "moe_permute_sm100.cuh"
#include "moe_unpermute_sm100.cuh"
#include "mul_sum_add_sm100.cuh"
#include "per_token_group_quantize_fp8.cuh"
#include "prob_scatter_sm100.cuh"
#include "sm100_ptx.cuh"
#include "softmax_gather_sm100.cuh"
#include "tasks/common/sampling.cuh"
#include "tasks/speculative_decoding/mtp_token_ops.cuh"
#include "tasks/speculative_decoding/prompt_lookup.cuh"
#include "tasks/speculative_decoding/target_verify.cuh"
#include "tasks/speculative_decoding/target_verify_mtp.cuh"
#include "tensor_init.cuh"
#include "topk_sigmoid_sm100.cuh"
#include "topk_softmax_sm100.cuh"
#include "transpose_scale_sm100.cuh"
