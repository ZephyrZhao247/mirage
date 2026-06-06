# R-0 Discovery Inventory — vLLM V4-Flash NVIDIA kernels (deduplicated)

Aggregated output of 6 parallel R-0 discovery agents, with cross-agent
overlaps merged. This is the input checklist for Wave R-1 spec-writers.

**Repo path conventions:** all source paths below are relative to
`/raid/user_data/zepengz/projects/mirage_2/`. Where an agent cited a path
outside `deps/vllm/`, the path has been rewritten to point at our
in-tree checkout. (R-0d cited `/raid/user_data/zepengz/projects/vllm/csrc/...`
because a second vLLM clone exists at that path; the same files are in our
`deps/vllm/csrc/...`.)

**Useful pre-existing reference:** `deps/vllm/DEEPSEEK_V4_KERNELS.md` contains
vLLM-team documentation of their V4 kernels (~400 lines). R-1 agents should
consult it but write specs against the actual source.

---

## 🚨 Decisions flagged for user before R-1 launches

### D1. MegaMoE incompatible with V4-Flash-Base (was Class B → now LOCKED)

R-0e verified at `deps/vllm/vllm/models/deepseek_v4/nvidia/model.py:430-434`:

```python
if self.use_mega_moe and getattr(config, "expert_dtype", "fp4") != "fp4":
    raise NotImplementedError(
        "DeepSeek V4 MegaMoE only supports fp4 experts; got expert_dtype="
        f"{config.expert_dtype!r}. Drop --kernel-config moe_backend="
        "deep_gemm_mega_moe for this checkpoint."
    )
```

V4-Flash-Base ships `expert_dtype="fp8"`, so **MegaMoE path is hard-blocked
at init time**. Only `FusedMoE` Triton path is reachable for our target.

**Proposed**: drop MegaMoE kernels from R-1 scope entirely. Spec the FusedMoE
Triton path (`fused_moe_kernel`, `fused_moe_kernel_gptq_awq`,
`moe_align_block_size`, `write_zeros_to_output`) as the canonical MoE compute
path. Document the MegaMoE existence in `_summary.md` as a "future-FP4
checkpoint" path with source pointers, no spec files.

### D2. CuteDSL fast paths — spec both variants, or just CuteDSL?

Three pairs where vLLM ships both a Triton `common/ops/` kernel AND a CuteDSL
`nvidia/ops/` variant, dispatched by `has_cutedsl()` at runtime:

| Pair | Triton (`common/ops/`) | CuteDSL (`nvidia/ops/`) | Dispatch |
|---|---|---|---|
| Indexer Q quant | `_fused_indexer_q_rope_quant_kernel` (FP8) + `_fused_indexer_q_rope_mxfp4_kernel` (MXFP4) | `IndexerQFp8Kernel` + `IndexerQMxFp4Kernel` | `has_cutedsl()` ? CuteDSL : Triton |
| Dequant + gather K cache | `_dequantize_and_gather_k_kernel` | `DequantGatherKCacheKernel` | `has_cutedsl()` ? CuteDSL : Triton |
| Sparse-attn compress (head_dim=512) | `_fused_kv_compress_norm_rope_insert_sparse_attn` | `compress_norm_rope_store_cutedsl` → C4 + C128 split kernels | `is_cuda() and head_dim==512` ? CuteDSL : Triton |

CuteDSL is the NVIDIA fast path. Triton is the universal fallback.

**Proposed**: spec **both variants** for each pair (per Class B rule). MPK
can later pick which to implement. ~3 extra spec files vs. dropping the
Triton fallbacks.

### D3. `use_fp4_cache` couples Q-side and K-side indexer kernels (Class B)

Single config flag flips both:
- K-side: `_fused_kv_compress_norm_rope_insert_indexer_attn` (FP8) ↔ `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn` (MXFP4)
- Q-side: `_fused_indexer_q_rope_quant_kernel` (FP8) ↔ `_fused_indexer_q_rope_mxfp4_kernel` (MXFP4)

**Proposed**: spec all 4 variants. Mark `use_fp4_cache=True/False` as the
**linked** dispatch in each spec's "Config-dependent dispatch" section so
the user understands they must pick one consistently.

### D4. Hash routing vs scored routing — same kernel, different template

R-0d confirmed both paths use `topkGatingSoftplusSqrt` (CUDA) with a
`USE_HASH` template branch. V4-Flash uses BOTH per layer:
- Layers 0-2 (`layer_idx < num_hash_layers=3`): hash routing
- Layers 3-42: scored routing (`sqrtsoftplus`)

**Proposed**: one spec file `topk_softplus_sqrt.md` documenting both template
variants. Note the per-layer dispatch in the Math section.

### D5. SM90 vs SM100 dispatches — Class A locked to SM100 for V4-Flash

Several kernels behave differently on SM90 vs SM100:
- `fused_inv_rope_fp8_quant` — `tma_aligned_scales` flag (UE8M0 packed int32 on SM100, fp32 on SM90)
- `deepseek_v4_fp8_einsum` — recipe `(1, 128, 128)` SM90 vs `(1, 1, 128)` SM100
- DeepGEMM `tf32_hc_prenorm_gemm` — separate `.cuh` files: `sm90_tf32_hc_prenorm_gemm.cuh` and `sm100_tf32_hc_prenorm_gemm.cuh`

V4-Flash on B200 = SM100. **Proposed**: spec only the SM100 branch (matches B200 target). Note SM90 alternative as a Class A pointer in each affected spec.

### D6. Source path discrepancy (R-0d)

R-0d agent cited paths starting with `/raid/user_data/zepengz/projects/vllm/`
instead of `deps/vllm/`. There are TWO vLLM checkouts on this machine:
- `/raid/user_data/zepengz/projects/vllm/` (standalone)
- `/raid/user_data/zepengz/projects/mirage_2/deps/vllm/` (our in-tree)

The cited files (e.g., `csrc/moe/topk_softplus_sqrt_kernels.cu`) exist in
both. **Proposed**: R-1 agents read from `deps/vllm/` only; paths in this
document have been rewritten where verified.

---

## Class B dispatches summary (after the resolutions above)

| # | Config flag | Variants | Locked for V4-Flash-Base? | Action |
|---|---|---|---|---|
| 1 | `moe_backend = "deep_gemm_mega_moe"` vs FusedMoE | MegaMoE (Cutlass FP4) vs FusedMoE (Triton FP8/INT8/INT4) | **Locked to FusedMoE** (FP4 incompatible with FP8 weights) | D1: drop MegaMoE from spec scope |
| 2 | `has_cutedsl()` (3 sites) | CuteDSL vs Triton common-op | Open — both NVIDIA-reachable | D2: spec both |
| 3 | `use_fp4_cache` (Indexer) | FP8 vs MXFP4 (couples Q+K sides) | Open — needs decision | D3: spec both, link variants |
| 4 | `compress_ratio ∈ {0, 4, 128}` per-layer | SWA-only / SWA+sparse-topk / SWA+precomputed | All three run on V4-Flash (per-layer config) | Spec all three (already covered by separate kernel variants) |
| 5 | `is_hash_moe` per-layer | hash routing vs scored routing | Both run on V4-Flash | D4: one spec, two template branches |

---

## Deduplicated kernel inventory (R-1 input checklist)

Total unique kernel implementations after dedup: **~50** (excludes ~13 routine `nn.Linear` instances that go into `linear_cublas.md`).

### Subsystem A — Attention pre-processing (4 kernels) — owner: R-1a

| # | kernel | source | language | class | notes |
|---|---|---|---|---|---|
| A1 | `_fused_q_kv_rmsnorm_kernel` | `deps/vllm/vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py:8-54` | Triton | A | per-token RMSNorm on Q+KV |
| A2 | `fused_q_kv_rmsnorm` (wrapper) | `deps/vllm/vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py:57-96` | Triton wrapper | A | called at `attention.py:422` |
| A3 | `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` | `deps/vllm/csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:~111-534` | CUDA (`torch.ops._C`) | A | monolithic fused kernel: Q-RMSNorm + Q-RoPE + KV-RoPE + FP8 quant + paged cache insert |
| A4 | `apply_rotary_emb` (generic) | `deps/vllm/vllm/vllm_flash_attn/ops/triton/rotary.py:12-132` | Triton (flash-attn) | A | generic RoPE wrapper used pre-quant in some paths |

### Subsystem B — MLA core + cache utilities (6 kernels) — owner: R-1b

| # | kernel | source | language | class | notes |
|---|---|---|---|---|---|
| B1 | `flash_mla_with_kvcache` (decode) | external `_flashmla_C.abi3.so` via `deps/vllm/vllm/v1/attention/ops/flashmla.py:23-27` | external CUDA | A | called at flashmla.py:286 |
| B2 | `flash_mla_sparse_fwd` (prefill) | external `_flashmla_C.abi3.so` via same | external CUDA | A | called at flashmla.py:416 |
| B3 | `quantize_and_insert_k_kernel` | `deps/vllm/vllm/models/deepseek_v4/common/ops/cache_utils.py:24-140` | Triton | A | UE8M0 FP8 quant + paged insert |
| B4 | `_dequantize_and_gather_k_kernel` (Triton) | `deps/vllm/vllm/models/deepseek_v4/common/ops/cache_utils.py:197-305` | Triton | B | Class B sibling: B5 |
| B5 | `DequantGatherKCacheKernel` (CuteDSL) | `deps/vllm/vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py:32-331` | CuteDSL | B | Class B sibling: B4 |
| B6 | `_compute_global_topk_indices_and_lens_kernel` | `deps/vllm/vllm/models/deepseek_v4/common/ops/cache_utils.py:417-466` | Triton | A | maps local topk → global KV slots |
| B7 | `_combine_topk_swa_indices_kernel` | `deps/vllm/vllm/models/deepseek_v4/common/ops/cache_utils.py:524-594` | Triton | A | concat topk + SWA indices for sparse prefill |

### Subsystem C — Compressor (head_dim=512, attention compressor) (4 kernels) — owner: R-1c

| # | kernel | source | language | class | notes |
|---|---|---|---|---|---|
| C1 | `_save_partial_states_kernel` | `deps/vllm/vllm/models/deepseek_v4/common/ops/save_partial_states.py:48-102` | Triton | A | stores KV state + score to compressor state cache |
| C2 | `_fused_kv_compress_norm_rope_insert_sparse_attn` (Triton, head_dim=512) | `deps/vllm/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:112-297` | Triton | B | Class B sibling: C3a/C3b |
| C3a | `SparseAttnCompressNormRopeStoreC4Kernel` (CuteDSL, C4 fused) | `deps/vllm/vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py:75-460` | CuteDSL | B | compress_ratio=4 path |
| C3b | `SparseAttnCompressKernel` + `SparseAttnNormRopeStoreKernel` (CuteDSL, C128 split) | `deps/vllm/vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py:463-815`, `818-1087` | CuteDSL | B | compress_ratio=128 split path |

### Subsystem D — Indexer (head_dim=128) (8 kernels: 4 K-side, 4 Q-side, all coupled by use_fp4_cache) — owner: R-1c

| # | kernel | source | language | class | notes |
|---|---|---|---|---|---|
| D1 | `_fused_kv_compress_norm_rope_insert_indexer_attn` (FP8) | `deps/vllm/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:302-474` | Triton | B | Class B sibling: D2; coupled with Q1/Q2 via `use_fp4_cache` |
| D2 | `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn` (MXFP4) | `deps/vllm/vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:479-667` | Triton | B | Class B sibling: D1 |
| D3 | `_fused_indexer_q_rope_quant_kernel` (FP8 Q) | `deps/vllm/vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:69-172` | Triton | B | coupled with D1 via `use_fp4_cache` |
| D4 | `_fused_indexer_q_rope_mxfp4_kernel` (MXFP4 Q) | `deps/vllm/vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:174-282` | Triton | B | coupled with D2 |
| D5 | `IndexerQFp8Kernel` (CuteDSL FP8 Q) | `deps/vllm/vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py:428-610` | CuteDSL | B | Class B sibling: D3 (CuteDSL fast path) |
| D6 | `IndexerQMxFp4Kernel` (CuteDSL MXFP4 Q) | `deps/vllm/vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py:253-425` | CuteDSL | B | Class B sibling: D4 |
| D7 | `fp8_fp4_paged_mqa_logits` (paged decode) | DeepGEMM via `deps/vllm/vllm/utils/deep_gemm.py` | CUDA (DeepGEMM) | A | called at sparse_attn_indexer.py:324 (decode path) |
| D8 | `fp8_fp4_mqa_logits` (prefill) | DeepGEMM via `deps/vllm/vllm/utils/deep_gemm.py` | CUDA (DeepGEMM) | A | called at sparse_attn_indexer.py:233 (prefill path) |

### Subsystem E — Post-attention (4 kernels) — owner: R-1d

| # | kernel | source | language | class | notes |
|---|---|---|---|---|---|
| E1 | `_fused_inv_rope_fp8_quant_per_head` | `deps/vllm/vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py:17-136` | Triton | A | inverse RoPE + UE8M0 block FP8 quant (this is the worked-example template) |
| E2 | `deepseek_v4_fp8_einsum` (wo_a) | `deps/vllm/vllm/utils/deep_gemm.py:298-302` wrapper, dispatch in `attention.py:193-199` | cuBLAS or DeepGEMM | A | recipe (1,128,128) SM90 / (1,1,128) SM100 |
| E3 | `wo_b` Linear | `nn.Linear` | cuBLAS | A | routine — into `linear_cublas.md` |
| E4 | `fused_wqa_wkv` Linear | `nn.Linear` (`MergedColumnParallelLinear`) | cuBLAS | A | routine — into `linear_cublas.md` |

### Subsystem F — Hyper-Connections (8 kernels) — owner: R-1e

| # | kernel | source | language | class | notes |
|---|---|---|---|---|---|
| F1 | `mhc_pre_big_fuse_tilelang` | `deps/vllm/vllm/model_executor/kernels/mhc/tilelang_kernels.py:55-189` | TileLang | A | post_mix + comb_mix (Sinkhorn) + layer_input fused |
| F2 | `mhc_pre_big_fuse_with_norm_tilelang` | `deps/vllm/vllm/model_executor/kernels/mhc/tilelang_kernels.py:197-354` | TileLang | A | variant with fused RMSNorm |
| F3 | `hc_prenorm_gemm_tilelang` | `deps/vllm/vllm/model_executor/kernels/mhc/tilelang_kernels.py:537-618` | TileLang | A | GEMM + sqrsum with split-K |
| F4 | `hc_prenorm_gemm_block_m_tilelang` | `deps/vllm/vllm/model_executor/kernels/mhc/tilelang_kernels.py:623-713` | TileLang | A | M-blocked variant for batch ≥ 1024 |
| F5 | `mhc_fused_tilelang` | `deps/vllm/vllm/model_executor/kernels/mhc/tilelang_kernels.py:359-477` | TileLang | A | fused post+pre for small-token regime |
| F6 | `mhc_post_tilelang` | `deps/vllm/vllm/model_executor/kernels/mhc/tilelang_kernels.py:482-532` | TileLang | A | post-mapping `c*d + comb @ residual` |
| F7 | `hc_head_fuse_tilelang` | `deps/vllm/vllm/model_executor/kernels/mhc/tilelang_kernels.py:718-812` | TileLang | A | sigmoid-gated weighted sum + RMS |
| F8 | `tf32_hc_prenorm_gemm` (SM100) | `deps/vllm/vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh` | CUDA (DeepGEMM, Cutlass) | A | SM100 fast path; SM90 sibling at `sm90_tf32_hc_prenorm_gemm.cuh` is locked-alternative pointer per D5 |

### Subsystem G — MoE routing + activation (3 kernels) — owner: R-1e

| # | kernel | source | language | class | notes |
|---|---|---|---|---|---|
| G1 | `topkGatingSoftplusSqrt` | `deps/vllm/csrc/moe/topk_softplus_sqrt_kernels.cu:85-426` | CUDA | A | one kernel, two template branches: `USE_HASH=true` (layers 0-2) and `USE_HASH=false` (layers 3-42) |
| G2 | `silu_and_mul_with_clamp` (active for V4) | `deps/vllm/csrc/libtorch_stable/activation_kernels.cu:266-271` (wrapper) + `act_and_mul_kernel<silu_kernel,true,true>` at lines 78-125 | CUDA | A | fully fused: clamp + silu + mul in one kernel |
| G3 | `dsv3_router_gemm` / `fp32_router_gemm` | `deps/vllm/csrc/libtorch_stable/dsv3_router_gemm*.cu` + `fp32_router_gemm.cu` | CUDA | A | tiered gate projection: DSV3-specialized for E=256/M≤16, fp32-specialized for E=256/M≤32, F.linear fallback otherwise |

### Subsystem H — MoE compute / FFN (FusedMoE Triton path only, per D1) (4 kernels) — owner: R-1f

| # | kernel | source | language | class | notes |
|---|---|---|---|---|---|
| H1 | `fused_moe_kernel` | `deps/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:293-553` | Triton | A | core expert GEMM; FP8/INT8 W8A16/W8A8 support; MUL_ROUTED_WEIGHT flag |
| H2 | `fused_moe_kernel_gptq_awq` | `deps/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:59-289` | Triton | A | INT4/INT8 weight quant with zero-point; older path |
| H3 | `write_zeros_to_output` | `deps/vllm/vllm/model_executor/layers/fused_moe/fused_moe.py:39-55` | Triton | A | utility: zeros for expert==-1 |
| H4 | `moe_align_block_size` | `deps/vllm/vllm/model_executor/layers/fused_moe/moe_align_block_size.py:90` (Python wrapper) → `torch.ops._C` CUDA op | CUDA | A | sorts tokens by expert, pads to block_size |

### Subsystem I — Standard layers (5 kernels) — owner: R-1g

| # | kernel | source | language | class | notes |
|---|---|---|---|---|---|
| I1 | `rms_norm` (vLLM native) | `deps/vllm/vllm/model_executor/layers/layernorm.py:37-128` | CUDA (`_C.rms_norm`) | A | used for `attn_norm`, `ffn_norm`, `q_norm`, `kv_norm`, `enorm`, `hnorm`, `norm` final, `shared_head.norm` |
| I2 | `apply_rotary_emb` (generic) | `deps/vllm/vllm/vllm_flash_attn/ops/triton/rotary.py:12-132` | Triton (flash-attn) | A | same as A4 (already counted; mention here as the standard-layer entry) |
| I3 | `VocabParallelEmbedding` GPU op | `deps/vllm/vllm/model_executor/layers/vocab_parallel_embedding.py` (find Triton/CUDA call) | Triton or torch builtin | A | called for `embed_tokens` |
| I4 | `LogitsProcessor` | `deps/vllm/vllm/model_executor/layers/logits_processor.py:18-104` | Python wrapping `lm_head` apply + sampler | A | generic; no V4-specific kernel |
| I5 | residual add | torch builtin or fused-add-norm | torch builtin | A | flag if a fused `add_residual_norm` exists; otherwise builtin |

### Subsystem J — MTP (2 kernels, plus DecoderLayer reuse) — owner: R-1g

| # | kernel | source | language | class | notes |
|---|---|---|---|---|---|
| J1 | `_fused_mtp_input_rmsnorm_kernel` | `deps/vllm/vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py:44-97` | Triton | A | pos==0 masking + dual RMSNorm (enorm + hnorm) |
| J2 | `_mtp_shared_head_rmsnorm_kernel` | `deps/vllm/vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py:100-121` | Triton | A | shared-head norm for MTP logits |

**Note**: MTP block itself reuses `DeepseekV4DecoderLayer` (full attention + MoE + HC). Do not duplicate those specs for MTP; cross-reference in `_summary.md`.

---

## Routine `nn.Linear` instances (→ `linear_cublas.md`)

| Instance | in_dim | out_dim | dtype | parallelism | caller |
|---|---|---|---|---|---|
| `fused_wqa_wkv` | 4096 | [1536, 512] | bf16 | MergedColumnParallel (replicated, `disable_tp=True`) | `DeepseekV4Attention:658` |
| `wq_b` | 1536 | n_heads*head_dim | bf16 | ColumnParallel | `DeepseekV4Attention:667` |
| `wo_a` | head_dim*n_heads/n_groups | n_groups*o_lora_rank | bf16/fp8 | ColumnParallel `is_bmm=True` | `DeepseekV4Attention:677` (consumed by `deepseek_v4_fp8_einsum`) |
| `wo_b` | n_groups*o_lora_rank | hidden_size | bf16 | RowParallel | `DeepseekV4Attention:687` |
| `gate_up_proj` (shared expert) | 4096 | [2048, 2048] | bf16 | MergedColumnParallel | `DeepseekV4MLP:82` |
| `down_proj` (shared expert) | 2048 | 4096 | bf16 | RowParallel | `DeepseekV4MLP:90` |
| `GateLinear` (router gate) | 4096 | 256 | bf16 in / fp32 out | Replicated (TP+EP) | `DeepseekV4MoE:437-443` |
| `embed_tokens` | vocab_size | 4096 | bf16 | VocabParallel | `DeepseekV4Model.embed_input_ids:1064` |
| `lm_head` | 4096 | vocab_size | bf16 | ParallelLMHead | `DeepseekV4ForCausalLM.compute_logits:1301` |
| `e_proj` (MTP) | 4096 | 4096 | bf16 | Replicated | `DeepSeekV4MultiTokenPredictorLayer:87` |
| `h_proj` (MTP) | 4096 | 4096 | bf16 | Replicated | `DeepSeekV4MultiTokenPredictorLayer:94` |
| `shared_head.head` (MTP LM head) | 4096 | vocab_size | bf16 | ParallelLMHead | `DeepSeekV4MultiTokenPredictor.compute_logits:252` |
| Compressor `fused_wkv_wgate` | hidden | [coff*head_dim, coff*head_dim] | bf16→fp32 | TP-disabled | `compressor.py:225-233` |
| Indexer `wq_b` | 1536 | 64*128 | bf16 | Replicated | `attention.py:753-759` |
| Indexer `weights_proj` | 4096 | 64 | bf16 | Replicated | `attention.py:760-766` |

---

## Cross-scope dedup decisions made

- `mhc_*_tilelang` and `hc_head_fuse_tilelang`: appeared in both R-0c (canonical owner) and R-0f. **R-0c wins** — those are in Subsystem F.
- `_swiglustep_and_mul_kernel` (Triton): mentioned by R-0d (alternative path) and R-0e (alternative for shared expert). V4-Flash uses `silu_and_mul_with_clamp` (CUDA) instead. **The Triton kernel is NOT in scope** for V4-Flash; documented as a Class A locked-alternative pointer.
- `compress_norm_rope_store_triton` dispatcher: mentioned in both R-0a (caller perspective) and R-0b (kernel owner). **R-0b wins** for the dispatcher; the dispatched kernels are listed in Subsystems C and D.
- `dequantize_and_gather_k_cache` dispatcher: appears in R-0a (caller perspective) and R-0b. **R-0b wins**; the dispatched kernel(s) are B4 (Triton) and B5 (CuteDSL).

---

## Final summary

- **Total unique kernel specs to write**: ~46-50 (depending on D1/D2/D5 resolutions).
- **`linear_cublas.md`**: ~15 routine Linear instances in one table.
- **`_summary.md` (R-2 output)**: index + Class B decision matrix (D2, D3, plus pointers to D1/D5 locked).
- **R-1 partition**: subsystems A-J → 7 R-1 batches as listed in each "owner: R-1x" line above.
