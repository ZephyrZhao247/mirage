# `_summary.md` — vLLM canonical kernel spec index for DeepSeek V4-Flash NVIDIA

## Overview

This directory captures the **canonical vLLM kernel specs** for every GPU operator reachable from `vllm/models/deepseek_v4/nvidia/model.py` (the V4-Flash NVIDIA inference path) plus the supporting `csrc/...` and `vllm/third_party/...` kernels they invoke. Target hardware is **B200 / `sm_100a`** (Blackwell). The target checkpoint is **DeepSeek-V4-Flash-Base** with `expert_dtype="fp8"`, `score_func="sqrtsoftplus"`, `compress_ratios=[0, 4, 128]` per-layer, `num_hash_layers=3`, `n_routed_experts=256`, `hc_mult=4`, `hidden_size=4096`, `head_dim=512`, `q_lora_rank=1024`, `index_n_heads=64`, `index_head_dim=128`.

Each per-kernel `.md` follows `_template.md`: `## Identity`, `## Call sites`, `## Inputs`, `## Outputs`, `## Grid / Block`, `## Math`, `## Config-dependent dispatch`. Math sections cite `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:<lines>` as the semantic source of truth and express logic as PyTorch operators that mirror the kernel's fusion structure.

**Next phase**: MPK (Mirage Persistent Kernel) implementation that mirrors these specs. Each spec contains the producer/consumer contract MPK must preserve byte-for-byte (cache layouts, scale-encoding rules, weight-fold conventions, padding requirements). The `## Math` sections are written so an MPK kernel author can implement straight from the spec without re-reading vLLM source.

Path convention: cite relative to the vLLM project root (e.g., `vllm/models/deepseek_v4/...`, `csrc/...`, `vllm/third_party/deep_gemm/include/...`). The `deps/vllm/` prefix is dropped.

## Hyperlinked index (Table)

### Attention pre-processing

| Spec file | Kernel name | Language / DSL | Source (file:lines, repo-relative) | # Call sites | Class A or B | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| [apply_rotary_emb](apply_rotary_emb.md) | `rotary_kernel` (flash-attn generic RoPE) | Triton | `vllm/vllm_flash_attn/ops/triton/rotary.py:12-131` | 0 production | A | NOT in V4-Flash production graph (all V4 RoPE is fused); kept as generic fallback. |
| [fused_q_kv_rmsnorm](fused_q_kv_rmsnorm.md) | `_fused_q_kv_rmsnorm_kernel` | Triton | `vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py:8-54` | 1 (`attention.py:422`) | A | Joint Q-LoRA + KV RMSNorm; bf16 in/out, fp32 reduce. |
| [fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert](fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.md) | `fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel` (+ reduced-grid variant) | CUDA | `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:111-534` | 1 (`attention.py:531`) | A | Monolithic 5-op fuse: Q-RMSNorm + Q-RoPE + KV-RoPE + UE8M0 FP8 quant + paged-cache insert; SWA cache writer. |

### MLA core + cache utilities

| Spec file | Kernel name | Language / DSL | Source (file:lines, repo-relative) | # Call sites | Class A or B | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| [combine_topk_swa_indices](combine_topk_swa_indices.md) | `_combine_topk_swa_indices_kernel` | Triton | `vllm/models/deepseek_v4/common/ops/cache_utils.py:524-594` | 1 (`flashmla.py:403`, prefill) | A | Concat topk + SWA window indices for sparse prefill; pads to 128-alignment for FlashMLA SM100. |
| [compute_global_topk_indices_and_lens](compute_global_topk_indices_and_lens.md) | `_compute_global_topk_indices_and_lens_kernel` | Triton | `vllm/models/deepseek_v4/common/ops/cache_utils.py:417-466` | 1 (`flashmla.py:234`, C4A decode) | A | Decode-only block-table lookup + valid-count + padding-mask. |
| [dequantize_and_gather_k_kernel](dequantize_and_gather_k_kernel.md) | `_dequantize_and_gather_k_kernel` (Triton) | Triton | `vllm/models/deepseek_v4/common/ops/cache_utils.py:197-305` | 2 (`flashmla.py:373, 385`, prefill compressed + SWA) | B (sibling: CuteDSL `DequantGatherKCacheKernel`) | UE8M0 FP8 dequant + paged gather; CuteDSL fast path locked out per D2 (Triton-only documented). |
| [flash_mla_with_kvcache](flash_mla_with_kvcache.md) | `_flashmla_C.sparse_decode_fwd` | External CUDA (FlashMLA `.abi3.so`) | `vllm/third_party/flashmla/flash_mla_interface.py:54-177` | 1 (`flashmla.py:286`, decode) | A | MLA sparse FP8 decode; 656B/token paged layout; tile-scheduler meta cached per layer-type. |
| [flash_mla_sparse_fwd](flash_mla_sparse_fwd.md) | `_flashmla_C.sparse_prefill_fwd` | External CUDA (FlashMLA `.abi3.so`) | `vllm/third_party/flashmla/flash_mla_interface.py:180-217` | 1 (`flashmla.py:416`, prefill) | A | bf16 KV (pre-dequant'd) sparse prefill; B_TOPK ∈ {64,128} alignment. |
| [quantize_and_insert_k_kernel](quantize_and_insert_k_kernel.md) | `quantize_and_insert_k_kernel` | Triton | `vllm/models/deepseek_v4/common/ops/cache_utils.py:23-140` | 0 production (tests only) | A | Reference for FP8+UE8M0 K-cache packing; superseded by C++ fused op on NVIDIA. |

### Compressor (head_dim=512)

| Spec file | Kernel name | Language / DSL | Source (file:lines, repo-relative) | # Call sites | Class A or B | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| [fused_kv_compress_norm_rope_insert_sparse_attn](fused_kv_compress_norm_rope_insert_sparse_attn.md) | `_fused_kv_compress_norm_rope_insert_sparse_attn` | Triton | `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:112-297` | 1 (`compressor.py:357` via dispatcher) | B (CuteDSL sibling locked active path on NVIDIA per D2) | Compressor write for `head_dim=512`; handles `compress_ratio∈{4, 128}` via `OVERLAP` constexpr. |
| [save_partial_states](save_partial_states.md) | `_save_partial_states_kernel` | Triton | `vllm/models/deepseek_v4/common/ops/save_partial_states.py:48-102` | 1 (`compressor.py:314`) | A | Stores `kv` + `score + ape` into the compressor state cache; runs unconditionally per compressor forward. |

### Indexer (head_dim=128)

| Spec file | Kernel name | Language / DSL | Source (file:lines, repo-relative) | # Call sites | Class A or B | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| [fp8_fp4_mqa_logits](fp8_fp4_mqa_logits.md) | `sm100_fp{8,4}_mqa_logits` (DeepGEMM) | CUDA (DeepGEMM, CuTeDSL) | `vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp{8,4}_mqa_logits.cuh` | 1 (`sparse_attn_indexer.py:233`, prefill) | A (FP8/FP4 split = B via `use_fp4_cache`) | Single Python wrapper, two CUDA kernels selected by the runtime `q_scale` tensor. |
| [fp8_fp4_paged_mqa_logits](fp8_fp4_paged_mqa_logits.md) | `sm100_fp{8,4}_paged_mqa_logits` (DeepGEMM) | CUDA (DeepGEMM, CuTeDSL) | `vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp{8,4}_paged_mqa_logits.cuh` | 1 (`sparse_attn_indexer.py:324`, decode) | A (FP8/FP4 split = B via `use_fp4_cache`) | Paged decode MQA logits; schedule-metadata distributes (q_atom, kv_chunk) tuples across SMs. |
| [fused_indexer_q_rope_mxfp4](fused_indexer_q_rope_mxfp4.md) | `_fused_indexer_q_rope_mxfp4_kernel` | Triton | `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:174-282` | 1 (`attention.py:841`) | B (`use_fp4_cache=True` path; CuteDSL `IndexerQMxFp4Kernel` is dropped per D2) | MXFP4 Q-side quant; per-block UE8M0 scales NOT folded into weights (contrast FP8). |
| [fused_indexer_q_rope_quant](fused_indexer_q_rope_quant.md) | `_fused_indexer_q_rope_quant_kernel` | Triton | `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:69-172` | 1 (`attention.py:841`) | B (`use_fp4_cache=False` path; CuteDSL `IndexerQFp8Kernel` dropped per D2) | FP8 Q-side quant; per-(token,head) scalar q_scale **folded into weights**. |
| [fused_kv_compress_norm_rope_insert_indexer_attn](fused_kv_compress_norm_rope_insert_indexer_attn.md) | `_fused_kv_compress_norm_rope_insert_indexer_attn` | Triton | `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:302-474` | 1 (`compressor.py:357` indexer branch) | B (`use_fp4_cache=False` K-side) | Indexer K-side FP8 quant; 128 FP8 + 4 fp32-scale per token. |
| [fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn](fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn.md) | `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn` | Triton | `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:479-667` | 1 (`compressor.py:357` indexer branch) | B (`use_fp4_cache=True` K-side) | Indexer K-side MXFP4 quant; 64 packed-bytes + 4 UE8M0 bytes per token (re-uses same 132B slot as FP8). |

### Post-attention + o-projection

| Spec file | Kernel name | Language / DSL | Source (file:lines, repo-relative) | # Call sites | Class A or B | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| [fused_inv_rope_fp8_quant](fused_inv_rope_fp8_quant.md) | `_fused_inv_rope_fp8_quant_per_head` | Triton | `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py:17-136` | 1 (`attention.py:319`) | A (SM90/SM100 inside one kernel via `TMA_ALIGNED_SCALES`) | Inverse RoPE + UE8M0 FP8 block quant; TMA-aligned scales pre-packed into int32. **Template worked-example.** |
| [deepseek_v4_fp8_einsum](deepseek_v4_fp8_einsum.md) | DeepGEMM `fp8_einsum` (`sm100_a` einsum) | CUDA (DeepGEMM) | `vllm/utils/deep_gemm.py:298-302` + `vllm/third_party/deep_gemm/__init__.py:60-61` | 1 (`attention.py:338`) | A (SM90 alt is the `(1,128,128)` recipe — locked-out on B200) | `bhr,hdr->bhd` for `wo_a`; consumes pre-transformed scales from `fused_inv_rope_fp8_quant`. |

### Hyper-Connections (TileLang)

| Spec file | Kernel name | Language / DSL | Source (file:lines, repo-relative) | # Call sites | Class A or B | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| [hc_head_fuse_tilelang](hc_head_fuse_tilelang.md) | `hc_head_fuse_tilelang` | TileLang | `vllm/model_executor/kernels/mhc/tilelang_kernels.py:718-812` | 1 (`model.py:1095`) | A | Terminal hc → 1 collapse via sigmoid-gated weighted sum + RMS denom. |
| [hc_prenorm_gemm_block_m_tilelang](hc_prenorm_gemm_block_m_tilelang.md) | `hc_prenorm_gemm_block_m_tilelang` | TileLang | `vllm/model_executor/kernels/mhc/tilelang_kernels.py:623-713` | TileLang fallback only | A (locked alt of `hc_prenorm_gemm_tilelang`; only when `T≥1024 and n_splits==1`) | M-blocked variant for large batch — halves `fn` TMA traffic vs per-token CTA. |
| [hc_prenorm_gemm_tilelang](hc_prenorm_gemm_tilelang.md) | `hc_prenorm_gemm_tilelang` | TileLang | `vllm/model_executor/kernels/mhc/tilelang_kernels.py:537-618` | TileLang fallback only | A (DeepGEMM `tf32_hc_prenorm_gemm` is the active path on SM100) | Per-token CTA × split-K; emits `out [n_splits, T, 24]` + `sqrsum [n_splits, T]` partials. |
| [mhc_fused_tilelang](mhc_fused_tilelang.md) | `mhc_fused_tilelang` | TileLang | `vllm/model_executor/kernels/mhc/tilelang_kernels.py:359-477` | 1 (`tilelang.py:461`, T≤16 decode) | A (batch-size locked; sibling `mhc_post_tilelang` for T>16) | Fuses `hc_post` (new residual) with next-layer `hc_prenorm_gemm` partials; decode hot path. |
| [mhc_post_tilelang](mhc_post_tilelang.md) | `mhc_post_tilelang` | TileLang | `vllm/model_executor/kernels/mhc/tilelang_kernels.py:482-532` | 2 (`tilelang.py:477` T>16 prefill; `model.py:1084` final post) | A | Standalone `hc_post`: `c*d + comb @ residual`. |
| [mhc_pre_big_fuse_tilelang](mhc_pre_big_fuse_tilelang.md) | `mhc_pre_big_fuse_tilelang` (no-norm) | TileLang | `vllm/model_executor/kernels/mhc/tilelang_kernels.py:55-189` | 0 on V4 (V4 always passes norm_weight) | A (locked-alt of with-norm sibling) | RMSNorm rsqrt + Sinkhorn + sigmoid mixes + weighted-sum into bf16 `layer_input`. |
| [mhc_pre_big_fuse_with_norm_tilelang](mhc_pre_big_fuse_with_norm_tilelang.md) | `mhc_pre_big_fuse_with_norm_tilelang` | TileLang | `vllm/model_executor/kernels/mhc/tilelang_kernels.py:197-354` | Every layer transition on V4-Flash | A | Active V4-Flash `hc_pre` — fuses `attn_norm`/`ffn_norm` γ into the weighted sum (two RMSNorm denominators in one pass). |

### Hyper-Connections (DeepGEMM)

| Spec file | Kernel name | Language / DSL | Source (file:lines, repo-relative) | # Call sites | Class A or B | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| [tf32_hc_prenorm_gemm](tf32_hc_prenorm_gemm.md) | `sm100_tf32_hc_prenorm_gemm_impl` | CUDA (DeepGEMM, CUTLASS3 / tcgen05) | `vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh:42-346` | 2 (`tilelang.py:195, 491`) | A (SM90 sibling `sm90_tf32_hc_prenorm_gemm.cuh` locked out per D5) | TF32 split-K GEMM + per-token squared-sum; bf16 A → tf32 in tensor memory; fp32 fn. |

### MoE routing + activation

| Spec file | Kernel name | Language / DSL | Source (file:lines, repo-relative) | # Call sites | Class A or B | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| [dsv3_router_gemm](dsv3_router_gemm.md) | 4-tier dispatch (`dsv3_router_gemm` / `fp32_router_gemm` / cuBLASLt / `F.linear`) | CUDA + cuBLAS + PyTorch | `csrc/moe/dsv3_router_gemm_entry.cu`; `csrc/libtorch_stable/fp32_router_gemm.cu`; `vllm/model_executor/layers/fused_moe/router/gate_linear.py:111-144` | 2 (`model.py:555, 598`) | **B / DECISION REQUIRED** | V4-Flash H=4096 misses both Tier 1 (H=7168) and Tier 2 (H=3072) — Tier 3 (cuBLASLt bf16→fp32) is most likely path; Tier 4 if weight is fp32. |
| [silu_and_mul_with_clamp](silu_and_mul_with_clamp.md) | `act_and_mul_kernel<silu, ACT_FIRST=true, HAS_CLAMP=true>` | CUDA (vectorized 128b/256b) | `csrc/libtorch_stable/activation_kernels.cu:78-271` | 1 (`model.py:110` shared expert MLP) | A | Fused gate clamp(max=L) + silu + up clamp(±L) + mul; SM100 256-bit dispatch when `T>128`. |
| [topk_softplus_sqrt](topk_softplus_sqrt.md) | `topkGatingSoftplusSqrt<USE_HASH={true,false}>` | CUDA | `csrc/moe/topk_softplus_sqrt_kernels.cu:85-426` | 1 (via `fused_topk_bias_router.py:132`) | A (USE_HASH branches inside one kernel — Class A merged variants) | Layers 0-2 → `USE_HASH=true` (hash routing from `tid2eid`); layers 3-42 → `USE_HASH=false` (scored). |

### MoE compute — MegaMoE path

| Spec file | Kernel name | Language / DSL | Source (file:lines, repo-relative) | # Call sites | Class A or B | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| [fp8_fp4_mega_moe](fp8_fp4_mega_moe.md) | `sm100_fp8_fp4_mega_moe_impl` | CUDA (DeepGEMM, CUTLASS3, tcgen05) | `vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:21-62+` | 1 (`model.py:382`, conditional) | B (`moe_backend == "deep_gemm_mega_moe"`) | **INCOMPATIBLE with V4-Flash-Base** (`expert_dtype="fp8"` hard-errors at init). Documented for any FP4 V4 checkpoint. |
| [prepare_megamoe_inputs](prepare_megamoe_inputs.md) | `_prepare_megamoe_inputs_kernel` | Triton | `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py:15-115` | 1 (`model.py:366`) | B (MegaMoE path only) | FP8 quant + UE8M0 packed-scale + topk repack into the DeepGEMM symmetric buffer. |

### MoE compute — FusedMoE path

| Spec file | Kernel name | Language / DSL | Source (file:lines, repo-relative) | # Call sites | Class A or B | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| [fused_moe_kernel](fused_moe_kernel.md) | `fused_moe_kernel` | Triton | `vllm/model_executor/layers/fused_moe/fused_moe.py:292-553` | 2 per layer (L1 gate+up, L2 down) | A (V4-Flash active path) | Three quant strategies in one kernel (FP8 W8A8, INT8 W8A8, INT8 W8A16); V4-Flash uses FP8 W8A8 block (1,32). |
| [fused_moe_kernel_gptq_awq](fused_moe_kernel_gptq_awq.md) | `fused_moe_kernel_gptq_awq` | Triton | `vllm/model_executor/layers/fused_moe/fused_moe.py:58-289` | 2 per layer when wna16-Triton path active | A | INT4 W4A16 / INT8 W8A16 group-dequant; **NOT on V4-Flash path** (V4-Flash is FP8 W8A8). |
| [moe_align_block_size](moe_align_block_size.md) | `moe_align_block_size_kernel` + `count_and_sort_expert_tokens_kernel` (+ small-batch single-kernel variant) | CUDA | `csrc/moe/moe_align_sum_kernels.cu:81-587` | 1 per FusedMoE layer (`fused_moe.py:1465`) | A | Bucket-sort + cumsum + block-pad for the M-tiling of `fused_moe_kernel`. |
| [write_zeros_to_output](write_zeros_to_output.md) | `write_zeros_to_output` (Triton device helper) | Triton (inlined) | `vllm/model_executor/layers/fused_moe/fused_moe.py:38-55` | 2 (called inside `fused_moe_kernel` and `fused_moe_kernel_gptq_awq` when `expert_id == -1`) | A | Zero-fills C tile for off-TP-rank experts; essential for `moe_sum`/`topk_weight_and_reduce` correctness. |

### Standard layers + sampling

| Spec file | Kernel name | Language / DSL | Source (file:lines, repo-relative) | # Call sites | Class A or B | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| [logits_processor](logits_processor.md) | `LogitsProcessor.forward` (orchestrator over lm_head GEMM + TP gather + slice + soft-cap/scale) | Python orchestration (cuBLAS GEMM + NCCL) | `vllm/model_executor/layers/logits_processor.py:18-104` | 2 (`model.py:1301`, `mtp.py:252`) | A | Not a single kernel; lm_head GEMM is the dominant op. V4 has `scale=1.0`, `soft_cap=None`. |
| [rms_norm](rms_norm.md) | `rms_norm_kernel<scalar_t, VEC_SIZE, NUM_DIMS>` | CUDA | `csrc/libtorch_stable/layernorm_kernels.cu:13-85` | 1 standalone on V4 trunk (`model.py:1103`); 8 instances are weight containers feeding fused kernels | A | Only 1 of 9 V4 `RMSNorm` instances actually launches `_C.rms_norm`; the other 8 are absorbed by fused Triton/CuteDSL/TileLang/CUDA kernels. |
| [vocab_parallel_embedding](vocab_parallel_embedding.md) | `aten::embedding` → `aten::index_select` (+ torch.compile-fused mask on TP path) | PyTorch native | `vllm/model_executor/layers/vocab_parallel_embedding.py:67-78` | 2 (`model.py:1064`, `mtp.py:205`) | A | NOT a custom kernel — `F.embedding`. TP path adds a fused mask kernel. |

### MTP

| Spec file | Kernel name | Language / DSL | Source (file:lines, repo-relative) | # Call sites | Class A or B | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| [fused_mtp_input_rmsnorm](fused_mtp_input_rmsnorm.md) | `_fused_mtp_input_rmsnorm_kernel` | Triton | `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py:43-97` | 1 (`mtp.py:144`) | A | pos==0 mask + dual RMSNorm on `inputs_embeds` (enorm) and `previous_hidden` (hnorm × HC_MULT slots). |
| [mtp_shared_head_rmsnorm](mtp_shared_head_rmsnorm.md) | `_mtp_shared_head_rmsnorm_kernel` | Triton | `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py:100-121` | 1 (`mtp.py:247`) | A | Plain per-token RMSNorm before MTP `shared_head.head` (LM head); shares `_rmsnorm_row` body with above. |

### Routine cuBLAS Linears

| Spec file | Kernel name | Language / DSL | Source (file:lines, repo-relative) | # Call sites | Class A or B | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| [linear_cublas](linear_cublas.md) | `F.linear` / cuBLAS `gemmEx` family | cuBLAS (or `Fp8LinearMethod.apply` for FP8 weights) | vLLM TP wrappers in `vllm/model_executor/layers/linear.py` | 15 routine instances | A | Collapsed table covering: `fused_wqa_wkv`, `wq_b`, `wo_a` (FP8 BMM), `wo_b`, shared-expert `gate_up_proj`/`down_proj`, `GateLinear` Tier-4 fallback, `embed_tokens`, `lm_head`, MTP `e_proj`/`h_proj`/`shared_head.head`, compressor `fused_wkv_wgate`, indexer `wq_b`/`weights_proj`. |

## Class B decision matrix

| # | Config flag | Variants (linked specs) | vLLM default | V4-Flash-Base default | Recommendation for MPK | Status |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `use_fp4_cache` (**6-kernel coupling**) | Q-side: [fused_indexer_q_rope_quant](fused_indexer_q_rope_quant.md) (FP8) ↔ [fused_indexer_q_rope_mxfp4](fused_indexer_q_rope_mxfp4.md) (MXFP4); K-side: [fused_kv_compress_norm_rope_insert_indexer_attn](fused_kv_compress_norm_rope_insert_indexer_attn.md) (FP8) ↔ [fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn](fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn.md) (MXFP4); logits: [fp8_fp4_mqa_logits](fp8_fp4_mqa_logits.md) + [fp8_fp4_paged_mqa_logits](fp8_fp4_paged_mqa_logits.md) (both dispatch FP8 vs FP4 via runtime tuple) | False (FP8) | configurable via `attention_config.use_fp4_indexer_cache` | **Implement FP8 path first** (simpler — single fp32 scale per token vs 4 UE8M0 bytes per token; folds Q-scale into weights). MXFP4 is performance-only on B200. | Open / both variants specced |
| 2 | CuteDSL fast path — indexer Q quant | [fused_indexer_q_rope_quant](fused_indexer_q_rope_quant.md) + [fused_indexer_q_rope_mxfp4](fused_indexer_q_rope_mxfp4.md) (Triton); `IndexerQFp8Kernel` + `IndexerQMxFp4Kernel` (CuteDSL siblings, source-only) | CuteDSL active when `has_cutedsl()` | CuteDSL on B200 | Use Triton spec as semantic reference | Dropped per user **D2**; Triton-only path documented |
| 3 | CuteDSL fast path — dequant+gather K cache | [dequantize_and_gather_k_kernel](dequantize_and_gather_k_kernel.md) (Triton); `DequantGatherKCacheKernel` (CuteDSL sibling, source-only) | CuteDSL active when `has_cutedsl()` | CuteDSL on B200 | Use Triton spec as semantic reference | Dropped per user **D2**; Triton-only path documented |
| 4 | CuteDSL fast path — sparse-attn compress (head_dim=512) | [fused_kv_compress_norm_rope_insert_sparse_attn](fused_kv_compress_norm_rope_insert_sparse_attn.md) (Triton); `SparseAttnCompressNormRopeStoreC4Kernel` + `SparseAttnCompressKernel`/`SparseAttnNormRopeStoreKernel` (CuteDSL siblings, source-only) | CuteDSL active when `is_cuda() and head_dim==512` | CuteDSL on B200 | Use Triton spec as semantic reference | Dropped per user **D2**; Triton-only path documented |
| 5 | `moe_backend = "deep_gemm_mega_moe"` vs FusedMoE | [fp8_fp4_mega_moe](fp8_fp4_mega_moe.md) + [prepare_megamoe_inputs](prepare_megamoe_inputs.md) (MegaMoE) **vs** [fused_moe_kernel](fused_moe_kernel.md) + [moe_align_block_size](moe_align_block_size.md) + [write_zeros_to_output](write_zeros_to_output.md) (+ [fused_moe_kernel_gptq_awq](fused_moe_kernel_gptq_awq.md) for INT4/8) (FusedMoE) | FusedMoE (Triton) | FusedMoE (MegaMoE hard-errors on `expert_dtype="fp8"`) | **Implement FusedMoE for V4-Flash-Base.** MegaMoE specced for any future FP4 checkpoint. | Both specced per user **D1**; MegaMoE incompatible with V4-Flash-Base |
| 6a | `compress_ratio = 0` (per-layer, SWA-only) | runs [fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert](fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.md) + [dequantize_and_gather_k_kernel](dequantize_and_gather_k_kernel.md) (SWA gather only) + [combine_topk_swa_indices](combine_topk_swa_indices.md) (with `top_k=0`); skips [save_partial_states](save_partial_states.md), the compressor + indexer kernels | per layer (`compress_ratios` list) | several layers per V4-Flash config | All three values run somewhere on V4-Flash → implement all three pipelines | Open / specced |
| 6b | `compress_ratio = 4` (C4A, indexer present) | runs compressor + indexer K-side + Q-side kernels; decode uses [compute_global_topk_indices_and_lens](compute_global_topk_indices_and_lens.md); prefill uses [combine_topk_swa_indices](combine_topk_swa_indices.md) with C4 topk | per layer | several layers per V4-Flash config | as above | Open / specced |
| 6c | `compress_ratio = 128` (C128A, no indexer — pre-computed positional topk) | runs compressor; uses `attn_metadata.c128a_*` (no indexer kernels); decode uses pre-computed `c128a_global_decode_topk_indices`; prefill uses [combine_topk_swa_indices](combine_topk_swa_indices.md) with C128A topk | per layer | several layers per V4-Flash config | as above | Open / specced |
| 7 | `dsv3_router_gemm` tier dispatch | [dsv3_router_gemm](dsv3_router_gemm.md) — 4 tiers (Tier 1 H=7168, Tier 2 H=3072, Tier 3 cuBLASLt, Tier 4 F.linear) | shape-based dispatch | V4-Flash H=4096 → Tier 3 (cuBLASLt bf16→fp32) if weight is bf16; else Tier 4 | **DECISION REQUIRED**: confirm `GateLinear.weight.dtype` at runtime. Tier 3 is the most likely V4-Flash hot path; Tier 4 if `force_fp32_compute=True` somehow fires. | **DECISION REQUIRED** — see `dsv3_router_gemm.md` Config-dependent dispatch table |

## Class A locked-dispatch reference table

| Config | V4-Flash-Base default | Locked-out alternative | Active spec file | Pointer to alternative source |
| --- | --- | --- | --- | --- |
| `expert_dtype` | `fp8` | `fp4` (MegaMoE path is hard-disabled at init via `NotImplementedError`, `model.py:430-434`) | `fused_moe_kernel.md` (FusedMoE Triton FP8 W8A8 block) | `fp8_fp4_mega_moe.md` + `prepare_megamoe_inputs.md` (documented as future-FP4 path; not on V4-Flash-Base) |
| `score_func` | `sqrtsoftplus` | `softmax`, `sigmoid` | `topk_softplus_sqrt.md` (single kernel covers both USE_HASH branches) | `csrc/moe/topk_softmax_kernels.cu`; `csrc/moe/torch_bindings.cpp:11-17` (`topk_sigmoid`) — locked-out, no spec |
| `tma_aligned_scales` | True (`cap.major >= 10` on B200 sm_100a) | False (SM90 fp32 scales) | `fused_inv_rope_fp8_quant.md` (SM100 UE8M0 packed-int32 branch) | Same file, `TMA_ALIGNED_SCALES=False` branch inside the same Triton kernel (lines 50-69, 120-135) |
| `_einsum_recipe` | `(1, 1, 128)` on sm_100a | `(1, 128, 128)` on SM90 | `deepseek_v4_fp8_einsum.md` (DeepGEMM sm_100a UE8M0 path) | Same DeepGEMM entry; recipe alone selects SM90 codepath inside `_C` |
| TF32 hyper-connections GEMM | SM100 DeepGEMM | SM90 DeepGEMM, TileLang fallback (`hc_prenorm_gemm_tilelang.md`, `hc_prenorm_gemm_block_m_tilelang.md`) | `tf32_hc_prenorm_gemm.md` (sm100) | `vllm/third_party/deep_gemm/include/deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh` (locked-out per **D5**) |
| MQA logits (paged decode + prefill) | DeepGEMM `sm100_fp{8,4}_*` | DeepGEMM `sm90_fp8_*` (no FP4 on SM90) | `fp8_fp4_paged_mqa_logits.md`, `fp8_fp4_mqa_logits.md` | `vllm/third_party/deep_gemm/include/deep_gemm/impls/sm90_fp{8}_paged_mqa_logits.cuh` and `sm90_fp8_mqa_logits.cuh` — locked-out per **D5** |
| MoE FP8 MMA | DeepGEMM `sm_100a` mega-MoE | `sm90` variant | `fp8_fp4_mega_moe.md` | `vllm/third_party/deep_gemm/...sm90_fp8_fp4_mega_moe.cuh` (not in scope; tcgen05 + 2-CTA cluster MMA are SM100a-only) |
| Platform | NVIDIA | ROCm | All NVIDIA specs in this directory | `vllm/models/deepseek_v4/amd/...` (e.g., `rocm_inv_rope_einsum`, AMD MHC ops); `attention.py:307` (ROCm `_o_proj_path`) — out of scope |
| Attention backend | FlashMLA | Aiter (closed-source AMD alternative) | `flash_mla_with_kvcache.md`, `flash_mla_sparse_fwd.md` | `vllm/v1/attention/backends/mla/aiter.py` — out of scope per platform locking |
| `RMSNorm` dispatch | `forward_cuda` → `_C.rms_norm` for the one standalone trunk norm; all other `RMSNorm` instances absorbed by fused kernels | `forward_native` (PyTorch fallback); `rms_norm_batch_invariant` when `VLLM_BATCH_INVARIANT=1` | `rms_norm.md` | `vllm/model_executor/layers/layernorm.py:104-116` (the dispatch logic); `vllm/_custom_ops.py:412-428` for `rms_norm_dynamic_per_token_quant` (unused on V4) |
| `apply_rotary_emb` (generic) | Not in V4-Flash production graph | — | `apply_rotary_emb.md` (documented for completeness) | All RoPE in V4-Flash is fused into Q+KV/inv/Indexer/Compressor kernels — no live caller |

## Class A merged-kernel variants (template branches inside one spec)

Several kernels cover multiple V4-Flash paths via `tl.constexpr` / `__device__` template branches inside one source file. The relevant spec documents both:

- [topk_softplus_sqrt](topk_softplus_sqrt.md) — `USE_HASH=true` (layers 0-2: hash routing from `tid2eid`) AND `USE_HASH=false` (layers 3-42: scored routing with `e_score_correction_bias`) in one kernel template. Host dispatcher selects by `tid2eid.has_value()`.
- [fused_inv_rope_fp8_quant](fused_inv_rope_fp8_quant.md) — SM100 (`TMA_ALIGNED_SCALES=True`, UE8M0-packed int32 scales) AND SM90 (`TMA_ALIGNED_SCALES=False`, fp32 scales) in one kernel via `tl.constexpr`.
- [fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert](fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.md) — Full-grid (`num_tokens < 1024`) vs reduced-grid (`num_tokens >= 1024`) variants in the same source file, sharing the per-slot `processDeepseekV4Slot` device function.
- [fused_kv_compress_norm_rope_insert_sparse_attn](fused_kv_compress_norm_rope_insert_sparse_attn.md) — `compress_ratio=4` (`OVERLAP=True`) AND `compress_ratio=128` (`OVERLAP=False`) in one kernel via constexpr.
- [fused_q_kv_rmsnorm](fused_q_kv_rmsnorm.md) / [fused_mtp_input_rmsnorm](fused_mtp_input_rmsnorm.md) — multi-task per CTA: `pid_task` selects Q vs KV (in `fused_q_kv_rmsnorm`) or enorm vs hnorm-slot (in `fused_mtp_input_rmsnorm`).
- HC TileLang variants — [hc_prenorm_gemm_tilelang](hc_prenorm_gemm_tilelang.md) vs [hc_prenorm_gemm_block_m_tilelang](hc_prenorm_gemm_block_m_tilelang.md) are **separate specs** but dispatched by the same wrapper based on the T-size threshold (`n_splits==1 and T>=1024 and use_default_config`).
- [mhc_pre_big_fuse_tilelang](mhc_pre_big_fuse_tilelang.md) vs [mhc_pre_big_fuse_with_norm_tilelang](mhc_pre_big_fuse_with_norm_tilelang.md) — separate specs but dispatched by `norm_weight is None` flag at the wrapper.
- [mhc_fused_tilelang](mhc_fused_tilelang.md) vs [mhc_post_tilelang](mhc_post_tilelang.md) — separate specs but dispatched by `num_tokens <= 16` flag (decode regime fuses `hc_post + hc_prenorm_gemm`, prefill splits them).

## Implementation-priority recommendation for MPK

Ranked list of ~10 kernels MPK should implement first to maximize V4-Flash hot-path coverage. Ranking criteria: (1) hot-path frequency (per-layer per-token), (2) no-substitute (kernels with no PyTorch equivalent or external library), (3) producer/consumer chain (output feeds another targeted kernel).

| # | Kernel (spec link) | Why this kernel? | Effort estimate |
| --- | --- | --- | --- |
| 1 | [fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert](fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.md) | The MLA pre-attention monolith — runs every layer, every token. Produces the FP8 paged K-cache that FlashMLA decode consumes. **No substitute** — without it, attention has no quantized K cache. | **Medium** (CUDA-port from source: 5-op fusion, ~430 LoC kernel). |
| 2 | [flash_mla_with_kvcache](flash_mla_with_kvcache.md) | The decode hot path — every decoder layer, every step. **No substitute** (closed-source FlashMLA). Consumes #1's cache layout. | **High** (external `.abi3.so` — must re-implement or stub via PyTorch reference if MPK can't link FlashMLA). |
| 3 | [tf32_hc_prenorm_gemm](tf32_hc_prenorm_gemm.md) | Every layer transition needs the (24, 16384)-fp32 × (T, 16384)-bf16 GEMM + sqrsum. **No substitute** for the fused sqrsum on SM100 — DeepGEMM is the canonical fast path. Feeds #4. | **High** (DeepGEMM CUTLASS3 + tcgen05; or fall back to TileLang reference at moderate perf loss). |
| 4 | [mhc_pre_big_fuse_with_norm_tilelang](mhc_pre_big_fuse_with_norm_tilelang.md) | Consumes #3's partials; every layer transition. Fuses 5-6 reference ops (rsqrt + Sinkhorn + sigmoid + RMSNorm γ). **No substitute** without re-launching ~6 kernels. | **Medium** (TileLang-port; complex Sinkhorn warp-0 + RMSNorm warp 1-2 split, but well-bounded shapes). |
| 5 | [fused_q_kv_rmsnorm](fused_q_kv_rmsnorm.md) | Every layer's attention input — one bf16 → bf16 RMSNorm grid covering Q-LoRA + KV. **High-frequency** but trivially substituted with two `_C.rms_norm` launches at small perf cost. | **Low** (Triton-port from source: ~50 LoC). |
| 6 | [fused_inv_rope_fp8_quant](fused_inv_rope_fp8_quant.md) | Every layer's post-attention. Inverse RoPE + UE8M0 FP8 block quant + TMA-aligned scale layout. **Producer chain**: output is consumed by `deepseek_v4_fp8_einsum`. Without it MPK can't drive the SM100 FP8 einsum efficiently. | **Low** (Triton-port from source: this is the worked-example template, well-documented). |
| 7 | [topk_softplus_sqrt](topk_softplus_sqrt.md) | Every MoE layer's gate (40+ layers of scored routing, 3 layers of hash routing). **No exact PyTorch substitute** — the bias-add-then-subtract semantics + sqrt-softplus + tie-breaking are hand-tuned. | **Medium** (CUDA-port: warp-cooperative top-k with butterfly reduce; template-heavy but well-bounded). |
| 8 | [fused_moe_kernel](fused_moe_kernel.md) | Every MoE layer's L1 + L2 GEMM. **No-substitute on V4-Flash's FP8 W8A8 block (1,32)** — three quant strategies in one Triton kernel; PyTorch fallback would be ~10× slower. Needs [moe_align_block_size](moe_align_block_size.md) as its M-routing producer. | **Medium** (Triton-port: ~260 LoC kernel; mature codebase). |
| 9 | [silu_and_mul_with_clamp](silu_and_mul_with_clamp.md) | Every shared-expert MLP + dense MLP. Per-element fused clamp+silu+clamp+mul. Substitutable with PyTorch but at significant launch-overhead cost on small decode batches. | **Low** (CUDA-port: ~250 LoC including vectorized 128b/256b dispatch). |
| 10 | [deepseek_v4_fp8_einsum](deepseek_v4_fp8_einsum.md) | The `wo_a` o-projection — every layer. **No substitute** without re-quantizing or doubling bf16 BMM cost. Consumes #6's output directly. | **High** (DeepGEMM `fp8_einsum` is external; or MPK implements `bhr,hdr->bhd` with FP8 block-scale dequant manually). |

**Tier 2** (implement after the top 10): [save_partial_states](save_partial_states.md) (compressor producer, simple Triton), [fused_kv_compress_norm_rope_insert_sparse_attn](fused_kv_compress_norm_rope_insert_sparse_attn.md) (compressor write, head_dim=512, every compressor-using layer), [mhc_post_tilelang](mhc_post_tilelang.md) / [mhc_fused_tilelang](mhc_fused_tilelang.md) (layer transition post-mapping; one of the two per layer based on batch size), [combine_topk_swa_indices](combine_topk_swa_indices.md) (prefill only, but every prefill chunk), [flash_mla_sparse_fwd](flash_mla_sparse_fwd.md) (prefill hot path).

## Cross-cutting observations from the spec writing

**The `use_fp4_cache` 6-kernel coupling.** A single user-config flag flips Q-side ([fused_indexer_q_rope_quant](fused_indexer_q_rope_quant.md) ↔ [fused_indexer_q_rope_mxfp4](fused_indexer_q_rope_mxfp4.md)), K-side ([fused_kv_compress_norm_rope_insert_indexer_attn](fused_kv_compress_norm_rope_insert_indexer_attn.md) ↔ [fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn](fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn.md)), AND the consumer DeepGEMM kernels ([fp8_fp4_mqa_logits](fp8_fp4_mqa_logits.md) prefill + [fp8_fp4_paged_mqa_logits](fp8_fp4_paged_mqa_logits.md) decode). MPK must implement these as a consistent 6-kernel cluster — picking FP8 on one side and MXFP4 on the other corrupts the DeepGEMM block-scaled MMA descriptor. The 132-byte/token cache slot is intentionally over-allocated so the same buffer can serve either path: FP8 uses 128+4 bytes, MXFP4 uses 64+4 bytes with 64 bytes wasted.

**Two RoPE bases coexist per layer.** The main attention RoPE uses `rope_theta=10000` (consumed by [fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert](fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.md), [fused_inv_rope_fp8_quant](fused_inv_rope_fp8_quant.md), and the compressor for head_dim=512). The indexer/compressor uses `compress_rope_theta=160000` (a separate `cos_sin_cache` built by `DeepseekV4Indexer.rotary_emb`). All four indexer kernels ([fused_indexer_q_rope_quant](fused_indexer_q_rope_quant.md), [fused_indexer_q_rope_mxfp4](fused_indexer_q_rope_mxfp4.md), [fused_kv_compress_norm_rope_insert_indexer_attn](fused_kv_compress_norm_rope_insert_indexer_attn.md), [fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn](fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn.md)) share the indexer-side cache. Mixing the two caches corrupts indexer scoring. MPK must maintain both `cos_sin_cache` tensors and route them to the correct kernel.

**Decode (3-step) vs prefill (5-step) HC pipeline difference.** At V4-Flash decode (T ≤ 16), each layer runs a 3-kernel chain: `[mhc_pre_big_fuse_with_norm | attn/ffn | mhc_fused_tilelang]`. The `mhc_fused_tilelang` fuses `hc_post` (new residual) with the next layer's `hc_prenorm_gemm` partials, avoiding an HBM round-trip on `residual_cur`. At prefill (T > 16), the chain expands to 5 kernels: `[mhc_pre_big_fuse_with_norm | attn/ffn | mhc_post_tilelang | tf32_hc_prenorm_gemm | mhc_pre_big_fuse_with_norm]`. The trade-off: small-T saves a kernel launch + round-trip; large-T benefits from separating fusion stages for better SM utilization. MPK should reproduce both regimes and select per-step.

**MegaMoE vs FusedMoE architectural fusion difference.** [fp8_fp4_mega_moe](fp8_fp4_mega_moe.md) is a **single persistent CUDA kernel** that fuses dispatch (NVLink multimem) + L1 GEMM + SwiGLU + L2 GEMM + combine-reduce — one launch covers an entire MoE layer end-to-end with dispatch/expert/epilogue warps cooperating via cluster barriers. FusedMoE is a **6-launch sequence**: `dsv3_router_gemm` (gate) → `topk_softplus_sqrt` (top-k) → `moe_align_block_size` (M-routing) → `fused_moe_kernel` (L1) → `apply_moe_activation` (SwiGLU) → `fused_moe_kernel` (L2) → `topk_weight_and_reduce` (combine). FusedMoE is the only V4-Flash-Base reachable path (MegaMoE hard-errors on `expert_dtype="fp8"` at MoE init). MPK targeting V4-Flash-Base should implement FusedMoE; MegaMoE would only matter for a hypothetical FP4 V4 checkpoint.

**The RMSNorm reality.** [rms_norm](rms_norm.md) documents 9 V4-Flash `RMSNorm` call sites. **Only 1 is a standalone `_C.rms_norm` invocation** — the final trunk norm at `model.py:1103` before `lm_head`. The other 8 (`q_norm`, `kv_norm`, `attn_norm`, `ffn_norm`, final compressor `norm`, MTP `enorm`/`hnorm`/`shared_head.norm`) are **weight containers** — their `forward` is never called; the `weight.data` and `variance_epsilon` are read directly by downstream fused kernels ([fused_q_kv_rmsnorm](fused_q_kv_rmsnorm.md), [mhc_pre_big_fuse_with_norm_tilelang](mhc_pre_big_fuse_with_norm_tilelang.md), [fused_kv_compress_norm_rope_insert_*](fused_kv_compress_norm_rope_insert_indexer_attn.md), [fused_mtp_input_rmsnorm](fused_mtp_input_rmsnorm.md), [mtp_shared_head_rmsnorm](mtp_shared_head_rmsnorm.md), and the C++ [fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert](fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.md)). MPK does not need a fast standalone RMSNorm for V4-Flash hot paths — it needs the fusion variants. The single standalone call at `model.py:1103` runs once per forward and is cheap (T × 4096 bf16).

**Weight-fold contracts are load-bearing.** [fused_indexer_q_rope_quant](fused_indexer_q_rope_quant.md) folds the per-(token, head) FP8 Q-scale into the `index_weights_out` tensor (so the downstream `fp8_fp4_mqa_logits` reads `weights` and gets dequant for free). [fused_inv_rope_fp8_quant](fused_inv_rope_fp8_quant.md) writes scales in TMA-aligned MN-major int32-packed UE8M0 layout that DeepGEMM `fp8_einsum` reads without `transform_sf_into_required_layout`. [prepare_megamoe_inputs](prepare_megamoe_inputs.md) writes int32-packed UE8M0 scales matching DeepGEMM's UTCCP layout for the activation side. If MPK reimplements these kernels with a "looks the same" output but wrong scale-encoding/layout, downstream DeepGEMM consumers silently read garbage.

## Open questions for user

The following items surfaced as genuine **DECISION REQUIRED** during R-1 and have not been resolved:

- **`dsv3_router_gemm` tier-3 vs tier-4 choice for V4-Flash's H=4096** — see [dsv3_router_gemm](dsv3_router_gemm.md) "Config-dependent dispatch" table. V4-Flash misses both Tier 1 (H=7168) and Tier 2 (H=3072). Tier 3 (cuBLASLt bf16→fp32) fires if `GateLinear.weight.dtype == bf16` (default for `ReplicatedLinear`); Tier 4 (`F.linear`) fires if forced to fp32. **Need runtime confirmation** of `GateLinear.weight.dtype` for V4-Flash-Base — neither tier is a hand-tuned CUDA kernel, so the choice is performance-only (Tier 3 ~10-20% faster than Tier 4 on B200).
- **`use_fp4_cache` choice for V4-Flash MPK target** — the 6-kernel cluster needs a consistent FP8/MXFP4 choice. R-1 has specced both variants. Default vLLM config picks FP8 (`use_fp4_indexer_cache=False`). Pick FP8 unless there's a specific reason to use MXFP4 (e.g., a downstream MXFP4 quant pipeline elsewhere in the stack).
- **`compress_ratios` distribution per layer** — V4-Flash-Base config sets `compress_ratios=[?, ?, ...]` per-layer. MPK needs the exact distribution to bake the right pipeline mix. Not surfaced in any spec; **needs config inspection** at `deps/deepseek_v4/DeepSeek-V4-Flash/inference/config.json`.
- **FlashMLA external `.abi3.so` link strategy** — [flash_mla_with_kvcache](flash_mla_with_kvcache.md) and [flash_mla_sparse_fwd](flash_mla_sparse_fwd.md) are NOT in the in-tree source. MPK must either re-implement them (high effort, ~1000+ LoC CUDA each, SM90+SM100 versions) or link the same `.abi3.so` (deployment risk: closed-source binary).
- **TF32 numerical drift tolerance** — [tf32_hc_prenorm_gemm](tf32_hc_prenorm_gemm.md) truncates A to 10-bit mantissa for the MMA. The sqr-sum read uses full fp32. Drift vs PyTorch fp32 reference is ~1e-3 relative. **Confirm acceptable for MPK validation** before re-implementing in TileLang or a custom Mirage GEMM path.

## Files in this directory

- [`_R0_inventory.md`](_R0_inventory.md) — R-0 discovery audit with the deduplicated kernel inventory and the decision-flagging summary (D1-D6).
- [`_template.md`](_template.md) — Canonical worked-example spec template using `fused_inv_rope_fp8_quant`.
- `_summary.md` — This file: index, dispatch matrices, implementation priority, cross-cutting observations.
- [`apply_rotary_emb.md`](apply_rotary_emb.md) — Generic flash-attn Triton RoPE (not in V4-Flash production graph).
- [`combine_topk_swa_indices.md`](combine_topk_swa_indices.md) — Concat topk + SWA indices for sparse prefill (Triton).
- [`compute_global_topk_indices_and_lens.md`](compute_global_topk_indices_and_lens.md) — Decode block-table lookup + valid-count for C4A (Triton).
- [`deepseek_v4_fp8_einsum.md`](deepseek_v4_fp8_einsum.md) — DeepGEMM `fp8_einsum` for `wo_a` o-projection (CUDA).
- [`dequantize_and_gather_k_kernel.md`](dequantize_and_gather_k_kernel.md) — Triton K-cache dequant + paged gather (CuteDSL sibling dropped per D2).
- [`dsv3_router_gemm.md`](dsv3_router_gemm.md) — 4-tier `GateLinear.forward` dispatch (CUDA + cuBLAS + F.linear).
- [`flash_mla_sparse_fwd.md`](flash_mla_sparse_fwd.md) — FlashMLA prefill (external CUDA).
- [`flash_mla_with_kvcache.md`](flash_mla_with_kvcache.md) — FlashMLA sparse decode (external CUDA).
- [`fp8_fp4_mega_moe.md`](fp8_fp4_mega_moe.md) — DeepGEMM persistent MegaMoE kernel (CUDA, sm_100a). INCOMPATIBLE with V4-Flash-Base.
- [`fp8_fp4_mqa_logits.md`](fp8_fp4_mqa_logits.md) — DeepGEMM prefill MQA logits (FP8 + MXFP4 variants, CUDA).
- [`fp8_fp4_paged_mqa_logits.md`](fp8_fp4_paged_mqa_logits.md) — DeepGEMM paged decode MQA logits (FP8 + MXFP4 variants, CUDA).
- [`fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.md`](fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.md) — Monolithic 5-op CUDA kernel: Q-RMSNorm + Q-RoPE + KV-RoPE + UE8M0 FP8 quant + paged insert.
- [`fused_indexer_q_rope_mxfp4.md`](fused_indexer_q_rope_mxfp4.md) — MXFP4 indexer Q-side quant (Triton; Class B sibling of FP8).
- [`fused_indexer_q_rope_quant.md`](fused_indexer_q_rope_quant.md) — FP8 indexer Q-side quant with weight-fold (Triton; Class B sibling of MXFP4).
- [`fused_inv_rope_fp8_quant.md`](fused_inv_rope_fp8_quant.md) — Post-attention inverse RoPE + UE8M0 FP8 block quant (Triton; worked-example template).
- [`fused_kv_compress_norm_rope_insert_indexer_attn.md`](fused_kv_compress_norm_rope_insert_indexer_attn.md) — FP8 indexer K-side compress+norm+rope+insert (Triton; Class B `use_fp4_cache=False`).
- [`fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn.md`](fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn.md) — MXFP4 indexer K-side variant (Triton; Class B `use_fp4_cache=True`).
- [`fused_kv_compress_norm_rope_insert_sparse_attn.md`](fused_kv_compress_norm_rope_insert_sparse_attn.md) — Compressor write for head_dim=512 (Triton; CuteDSL sibling locked active path on NVIDIA per D2).
- [`fused_moe_kernel.md`](fused_moe_kernel.md) — FusedMoE Triton GEMM with 3 quant strategies (FP8 W8A8, INT8 W8A8, INT8 W8A16).
- [`fused_moe_kernel_gptq_awq.md`](fused_moe_kernel_gptq_awq.md) — INT4 W4A16 / INT8 W8A16 group-dequant Triton GEMM (NOT on V4-Flash path).
- [`fused_mtp_input_rmsnorm.md`](fused_mtp_input_rmsnorm.md) — MTP joint enorm + hnorm × HC_MULT with pos==0 mask (Triton).
- [`fused_q_kv_rmsnorm.md`](fused_q_kv_rmsnorm.md) — Joint Q-LoRA + KV RMSNorm before MLA (Triton).
- [`hc_head_fuse_tilelang.md`](hc_head_fuse_tilelang.md) — Terminal hc → 1 collapse via sigmoid-gated weighted sum (TileLang).
- [`hc_prenorm_gemm_block_m_tilelang.md`](hc_prenorm_gemm_block_m_tilelang.md) — M-blocked TileLang fallback for large batch (T ≥ 1024).
- [`hc_prenorm_gemm_tilelang.md`](hc_prenorm_gemm_tilelang.md) — Per-token CTA TileLang fallback (DeepGEMM is the active path on SM100).
- [`linear_cublas.md`](linear_cublas.md) — Routine `nn.Linear` table for the 15 cuBLAS-backed Linear instances.
- [`logits_processor.md`](logits_processor.md) — `LogitsProcessor` orchestration (lm_head GEMM + TP gather + slice + soft-cap/scale).
- [`mhc_fused_tilelang.md`](mhc_fused_tilelang.md) — TileLang fused `hc_post + hc_prenorm_gemm` for decode (T ≤ 16).
- [`mhc_post_tilelang.md`](mhc_post_tilelang.md) — TileLang standalone `hc_post` (`c*d + comb @ residual`) for prefill (T > 16) + final post.
- [`mhc_pre_big_fuse_tilelang.md`](mhc_pre_big_fuse_tilelang.md) — TileLang hc_pre tail without RMSNorm γ (locked-alt for V4-Flash; only when `norm_weight=None`).
- [`mhc_pre_big_fuse_with_norm_tilelang.md`](mhc_pre_big_fuse_with_norm_tilelang.md) — TileLang hc_pre tail with fused `attn_norm`/`ffn_norm` γ (active V4-Flash path).
- [`moe_align_block_size.md`](moe_align_block_size.md) — CUDA bucket-sort + cumsum producing `sorted_token_ids` / `expert_ids` for FusedMoE.
- [`mtp_shared_head_rmsnorm.md`](mtp_shared_head_rmsnorm.md) — MTP plain RMSNorm before `shared_head.head` (LM head); Triton.
- [`prepare_megamoe_inputs.md`](prepare_megamoe_inputs.md) — FP8 quant + UE8M0 packed-scale + topk repack for DeepGEMM MegaMoE (Triton).
- [`quantize_and_insert_k_kernel.md`](quantize_and_insert_k_kernel.md) — Reference Triton K-cache packing (superseded by C++ fused op on V4-Flash NVIDIA).
- [`rms_norm.md`](rms_norm.md) — vLLM native CUDA RMSNorm; only 1 of 9 V4 call sites is a live standalone launch.
- [`save_partial_states.md`](save_partial_states.md) — Compressor state-cache writer with fused APE add (Triton).
- [`silu_and_mul_with_clamp.md`](silu_and_mul_with_clamp.md) — Fused clamp+silu+clamp+mul for shared-expert MLP (CUDA, vectorized 128b/256b).
- [`tf32_hc_prenorm_gemm.md`](tf32_hc_prenorm_gemm.md) — DeepGEMM TF32 split-K GEMM + per-token squared-sum (CUDA, sm_100a).
- [`topk_softplus_sqrt.md`](topk_softplus_sqrt.md) — MoE gating top-k with USE_HASH branch for hash MoE layers (CUDA).
- [`vocab_parallel_embedding.md`](vocab_parallel_embedding.md) — `F.embedding` lookup + TP-masked all-reduce (PyTorch native + torch.compile-fused mask).
- [`write_zeros_to_output.md`](write_zeros_to_output.md) — Triton device helper inlined by FusedMoE GEMM when expert is off-TP-rank.
