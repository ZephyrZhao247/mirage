# `TaskType` enum registry — V4-Flash MPK kernels

Pre-allocated `TaskType` enum slots in `include/mirage/persistent_kernel/runtime_header.h` for the 41 vLLM-canonical V4-Flash kernels documented in this directory.

**Range**: 350-400 (placeholders `TASK_V4_TASK_BEGIN = 350`, `TASK_V4_TASK_END = 400`). Avoids the existing SM100 range (230-323) and multi-GPU range (300-349). Slots 391-400 reserved for future additions.

**Naming convention**: `TASK_<KERNEL_NAME>_V4_SM100`. The `_V4_SM100` suffix distinguishes from existing MPK kernels (e.g., `TASK_RMS_NORM = 119`) and tags the architecture target.

**Purpose**: parallel per-kernel agents read this registry to learn their pre-allocated slot, so no slot-allocation races occur during the worktree-parallel implementation phase.

## Registry

| Slot | Kernel spec file | TaskType name | Subsystem |
|---|---|---|---|
| 350 | `rms_norm.md` | `TASK_RMS_NORM_V4_SM100` | Std layers |
| 351 | `apply_rotary_emb.md` | `TASK_APPLY_ROTARY_EMB_V4_SM100` | Std layers |
| 352 | `vocab_parallel_embedding.md` | `TASK_VOCAB_PARALLEL_EMBEDDING_V4_SM100` | Std layers |
| 353 | `logits_processor.md` | `TASK_LOGITS_PROCESSOR_V4_SM100` | Std layers |
| 354 | `mhc_pre_big_fuse_tilelang.md` | `TASK_MHC_PRE_BIG_FUSE_V4_SM100` | HC |
| 355 | `mhc_pre_big_fuse_with_norm_tilelang.md` | `TASK_MHC_PRE_BIG_FUSE_WITH_NORM_V4_SM100` | HC |
| 356 | `mhc_post_tilelang.md` | `TASK_MHC_POST_V4_SM100` | HC |
| 357 | `hc_head_fuse_tilelang.md` | `TASK_HC_HEAD_FUSE_V4_SM100` | HC |
| 358 | `hc_prenorm_gemm_tilelang.md` | `TASK_HC_PRENORM_GEMM_V4_SM100` | HC |
| 359 | `hc_prenorm_gemm_block_m_tilelang.md` | `TASK_HC_PRENORM_GEMM_BLOCK_M_V4_SM100` | HC |
| 360 | `mhc_fused_tilelang.md` | `TASK_MHC_FUSED_V4_SM100` | HC |
| 361 | `tf32_hc_prenorm_gemm.md` | `TASK_TF32_HC_PRENORM_GEMM_V4_SM100` | HC |
| 362 | `fused_q_kv_rmsnorm.md` | `TASK_FUSED_Q_KV_RMSNORM_V4_SM100` | Attention |
| 363 | `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.md` | `TASK_FUSED_DSV4_QNORM_ROPE_KV_INSERT_V4_SM100` | Attention |
| 364 | `flash_mla_with_kvcache.md` | `TASK_FLASH_MLA_DECODE_V4_SM100` | Attention |
| 365 | `flash_mla_sparse_fwd.md` | `TASK_FLASH_MLA_SPARSE_PREFILL_V4_SM100` | Attention |
| 366 | `quantize_and_insert_k_kernel.md` | `TASK_QUANTIZE_AND_INSERT_K_V4_SM100` | Attention |
| 367 | `dequantize_and_gather_k_kernel.md` | `TASK_DEQUANTIZE_AND_GATHER_K_V4_SM100` | Attention |
| 368 | `compute_global_topk_indices_and_lens.md` | `TASK_COMPUTE_GLOBAL_TOPK_INDICES_V4_SM100` | Attention |
| 369 | `combine_topk_swa_indices.md` | `TASK_COMBINE_TOPK_SWA_INDICES_V4_SM100` | Attention |
| 370 | `fused_inv_rope_fp8_quant.md` | `TASK_FUSED_INV_ROPE_FP8_QUANT_V4_SM100` | Attention |
| 371 | `deepseek_v4_fp8_einsum.md` | `TASK_DEEPSEEK_V4_FP8_EINSUM_V4_SM100` | Attention |
| 372 | `save_partial_states.md` | `TASK_SAVE_PARTIAL_STATES_V4_SM100` | Compressor |
| 373 | `fused_kv_compress_norm_rope_insert_sparse_attn.md` | `TASK_FUSED_KV_COMPRESS_NORM_ROPE_INSERT_SPARSE_ATTN_V4_SM100` | Compressor |
| 374 | `fused_kv_compress_norm_rope_insert_indexer_attn.md` | `TASK_FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_ATTN_V4_SM100` | Indexer |
| 375 | `fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn.md` | `TASK_FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_MXFP4_ATTN_V4_SM100` | Indexer |
| 376 | `fused_indexer_q_rope_quant.md` | `TASK_FUSED_INDEXER_Q_ROPE_QUANT_V4_SM100` | Indexer |
| 377 | `fused_indexer_q_rope_mxfp4.md` | `TASK_FUSED_INDEXER_Q_ROPE_MXFP4_V4_SM100` | Indexer |
| 378 | `fp8_fp4_paged_mqa_logits.md` | `TASK_FP8_FP4_PAGED_MQA_LOGITS_V4_SM100` | Indexer |
| 379 | `fp8_fp4_mqa_logits.md` | `TASK_FP8_FP4_MQA_LOGITS_V4_SM100` | Indexer |
| 380 | `topk_softplus_sqrt.md` | `TASK_TOPK_SOFTPLUS_SQRT_V4_SM100` | MoE routing |
| 381 | `dsv3_router_gemm.md` | `TASK_DSV3_ROUTER_GEMM_V4_SM100` | MoE routing |
| 382 | `silu_and_mul_with_clamp.md` | `TASK_SILU_AND_MUL_WITH_CLAMP_V4_SM100` | MoE routing |
| 383 | `prepare_megamoe_inputs.md` | `TASK_PREPARE_MEGAMOE_INPUTS_V4_SM100` | MoE compute |
| 384 | `fp8_fp4_mega_moe.md` | `TASK_FP8_FP4_MEGA_MOE_V4_SM100` | MoE compute |
| 385 | `fused_moe_kernel.md` | `TASK_FUSED_MOE_KERNEL_V4_SM100` | MoE compute |
| 386 | `fused_moe_kernel_gptq_awq.md` | `TASK_FUSED_MOE_KERNEL_GPTQ_AWQ_V4_SM100` | MoE compute |
| 387 | `write_zeros_to_output.md` | `TASK_WRITE_ZEROS_TO_OUTPUT_V4_SM100` | MoE compute |
| 388 | `moe_align_block_size.md` | `TASK_MOE_ALIGN_BLOCK_SIZE_V4_SM100` | MoE compute |
| 389 | `fused_mtp_input_rmsnorm.md` | `TASK_FUSED_MTP_INPUT_RMSNORM_V4_SM100` | MTP |
| 390 | `mtp_shared_head_rmsnorm.md` | `TASK_MTP_SHARED_HEAD_RMSNORM_V4_SM100` | MTP |

41 slots used (350-390). Reserve 391-400 for future additions.

## Notes for per-kernel agents

- **Task-name string** (used in `register_task("<name>")` from the Python catalog): use the kernel filename stem, suffixed with `_sm100`. E.g., for slot 350 (`rms_norm.md`), the task name is `"rms_norm_sm100"`. The codegen dispatch in `src/kernel/graph.cc` maps these strings to the `TaskType` enum value above.
- **Reuse vs new**: a few of these kernels are functionally equivalent to existing MPK kernels (e.g., `rms_norm` ≈ existing `TASK_RMS_NORM=119`; `vocab_parallel_embedding` ≈ existing `TASK_EMBEDDING=101`). If you decide the existing kernel covers the spec, your catalog module can wrap the existing TaskType and leave the V4-reserved slot unused (alias). Note your decision in the catalog module's docstring.
- **Catalog test naming**: `tests/runtime_python/layers/test_<kernel_name>.py` matching the spec filename stem.
