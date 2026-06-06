# fp8_fp4_paged_mqa_logits

## Identity
- Python entry: `vllm/utils/deep_gemm.py:407-457` — dispatcher `fp8_fp4_paged_mqa_logits(q, kv_cache, weights, context_lens, block_tables, schedule_metadata, max_model_len, clean_logits)`.
- Symbol resolved (line 229) via `getattr(_dg, "fp8_fp4_paged_mqa_logits", None)` from the `deep_gemm` package (vendored at `vllm/third_party/deep_gemm/`, exported in `vllm/third_party/deep_gemm/__init__.py:65`).
- Underlying CUDA implementations:
  - **FP8 Q + FP8 K**: `vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp8_paged_mqa_logits.cuh:30-439` (template `sm100_fp8_paged_mqa_logits<kNextN, kNumHeads, kHeadDim, BLOCK_KV, kIsContextLens2D, kIsVarlen, kNumQStages, kNumKVStages, SPLIT_KV, kNumSpecializedThreads, kNumMathThreads, logits_dtype_t>`).
  - **MXFP4 Q + MXFP4 K (block-scaled E2M1)**: `vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp4_paged_mqa_logits.cuh:21-510` (template `sm100_fp4_paged_mqa_logits<kNextN, kNumHeads, kHeadDim, BLOCK_KV, kIsContextLens2D, kIsVarlen, kNumQStages, kNumKVStages, SPLIT_KV, kNumSpecializedThreads, kNumMathThreads, logits_dtype_t>`).
  - SM90 fallback (locked-alternative pointer): `vllm/third_party/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh` (FP8 only — no FP4 on SM90).
- Scheduler: `vllm/third_party/deep_gemm/include/deep_gemm/scheduler/paged_mqa_logits.cuh` (`PagedMQALogitsScheduler<kNextN, kIsContextLens2D, kIsVarlen, BLOCK_KV, kNumBlocksPerSplit, kNumNextNAtoms>`). Schedule metadata is built by `get_paged_mqa_logits_metadata(context_lens, block_size, num_sms)` (`vllm/utils/deep_gemm.py:386-404`).
- Language/DSL: **CUDA + CUTLASS/CuTe** (TMA + UMMA block-scaled MMA on SM100 / tcgen05).
- Third-party dep: **DeepGEMM** (`deep_gemm` package). Built via JIT through `_C.init(...)` (`vllm/third_party/deep_gemm/__init__.py:121-124`).
- Registered as opaque custom op: **no** — direct call into the DeepGEMM Python wrapper.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/model_executor/layers/sparse_attn_indexer.py:324` | `sparse_attn_indexer` decode path | `q=(padded_q_quant, padded_q_scale)`, `kv_cache: [num_blocks, kv_block_size, 1, head_width]`, `weights: [B*next_n, n_heads=64]`, `seq_lens: [B, next_n] int32`, `block_table: [B, max_blocks] int32`, `schedule_metadata: [...]`, `max_model_len: int` | FP8 path: q `float8_e4m3fn`, kv uint8 packing FP8+fp32 scale; MXFP4 path: q packed uint8 (viewed as int8) + int32 block scales, kv packed uint8+int32 scales | always selected for decode in V4-Flash NVIDIA path (non-XPU); FP8 vs MXFP4 controlled by `use_fp4_cache` flag — when True, q_scale tuple element is non-None |

V4-Flash uses `kNumHeads = 64` (`index_n_heads`), `kHeadDim = 128` (`index_head_dim`), `kNextN ∈ {1, 1+num_spec_tokens}` (number of speculative draft tokens per request).

## Inputs

### Python wrapper inputs (lines 407-415)

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `q` | tuple `(q_values, q_scale)` | see below | — | FP8 path: `q_values=[B, next_n, 64, 128] float8_e4m3fn`, `q_scale=None` (per-token scale folded into `weights`). MXFP4 path: `q_values=[B, next_n, 64, 64] uint8` viewed as `int8` (kPackedFP4 tag), `q_scale=[B, next_n, 64] int32` (4 UE8M0 bytes per (token,head) packed as one int32). |
| `kv_cache` | `[num_blocks, kv_block_size, 1, head_width]` | uint8 | paged 4-D | FP8 path: `head_width = 128 + 4 = 132` per token (128 FP8 + 4 fp32 scale). MXFP4 path: `head_width = 64 + 4 = 68` per token (64 packed bytes + 4 UE8M0 bytes), reinterpreted via `kv_cache_as_quant_view(..., use_fp4_cache=True)` at `sparse_attn_indexer.py:73-77`. |
| `weights` | `[B * next_n, 64]` | fp32 | row-major | Folded weights from `fused_indexer_q_rope_quant` — see Spec 5/6 for weight-fold semantics. |
| `context_lens` | `[B]` or `[B, next_n]` | int32 | contiguous | Effective context length per batch element. V4-Flash uses 2-D form (`kIsContextLens2D=True`) for native spec decode (`indexer.py:193` comment). |
| `block_tables` | `[B, max_blocks]` | int32 | strided | Maps logical block → physical paged block index. |
| `schedule_metadata` | `[...]` opaque | int32 | — | Per-SM scheduling table built by `get_paged_mqa_logits_metadata(context_lens, block_size, num_sms)`; distributes (q_atom, kv_chunk) tuples across SMs. Layout defined in `paged_mqa_logits.cuh`. |
| `max_model_len` | scalar | int | — | Used to size the logits output (last dim). |
| `clean_logits` | scalar | bool | — | Whether to clean unfilled logit slots to `-inf`. Decode path passes `False` (line 332). |

### CUDA kernel template parameters (`sm100_fp{8,4}_paged_mqa_logits.cuh`)

| Template arg | Value (V4-Flash) | Meaning |
| --- | --- | --- |
| `kNextN` | 1 + num_spec_tokens (typically 1 or 2) | Number of speculative draft tokens per request. Odd N≥3 pad to even via TMA OOB zero-fill (`kPadOddN`). |
| `kNumHeads` | 64 | `index_n_heads`. |
| `kHeadDim` | 128 | `index_head_dim`. |
| `BLOCK_KV` | 128 (single warpgroup × `UMMA_M=128`) | KV block tile. |
| `kIsContextLens2D` | True | 2-D `[B, next_n]` context_lens for native spec decode. |
| `kIsVarlen` | False (typical) | False = same kNextN across batch; True allows per-batch variable next_n. |
| `kNumQStages` / `kNumKVStages` | 2-4 (JIT-tuned) | Software pipeline depths. |
| `SPLIT_KV` | `BLOCK_KV * kNumBlocksPerSplit` (e.g., 256 or 512) | KV split granularity per scheduler iteration. |
| `kNumSpecializedThreads` | 128 (4 specialized warps: TMA-Q, TMA-KV, UMMA, scheduler) | Producer threads. |
| `kNumMathThreads` | 128 × `kNumMathWarpGroups` | Consumer (math) threads. |
| `logits_dtype_t` | fp32 (output dtype) | Logits accumulation/output. |

### CUDA kernel runtime arguments

| Arg | Type | Meaning |
| --- | --- | --- |
| `batch_size` | uint32 | `B`. |
| `logits_stride` | uint32 | Row stride of output logits (= `max_model_len`). |
| `block_table_stride` | uint32 | `block_table.stride(0)`. |
| `context_lens` | `uint32*` | Per-batch (or per-`[B, next_n]`) context lengths. |
| `logits` | `logits_dtype_t*` | Output `[B * next_n, max_model_len]`. |
| `block_table` | `uint32*` | Logical→physical block table. |
| `indices` | `uint32*` | Optional auxiliary indices (e.g., gather indices for some variants — see scheduler). |
| `schedule_meta` | `uint32*` | Schedule metadata table. |
| `tensor_map_q`, `tensor_map_sf_q` (FP4 only), `tensor_map_kv`, `tensor_map_kv_scales` (FP8) / `tensor_map_sf_kv` (FP4), `tensor_map_weights` | `cute::TmaDescriptor` (grid constants) | TMA descriptors for each input. FP8 path has `tensor_map_kv_scales` (fp32 per-token scale); FP4 path has `tensor_map_sf_q` + `tensor_map_sf_kv` (int32 block scales). |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `logits` | `[B * next_n, max_model_len]` | fp32 (`logits_dtype_t`) | row-major | Per-(query, KV-position) MQA logits = `Σ_h weights[q, h] * (q[q, h, :] · k[kv_pos, :])`, with K dequantized in-line using per-token fp32 scale (FP8) or per-block UE8M0 scale (MXFP4). Unfilled slots (`kv_pos < cu_seqlen_ks` or `kv_pos >= cu_seqlen_ke`) are left untouched unless `clean_logits=True`. Decode path uses `clean_logits=False`. |

Downstream: `top_k_per_row_decode` (or `persistent_topk` when `topk_tokens ∈ {512, 1024, 2048}`) — `sparse_attn_indexer.py:337-359`.

## Grid / Block

- **Grid**: `gridDim = (kNumSMs,)` — persistent kernel. Each SM runs the full pipeline (TMA-Q + TMA-KV + UMMA + math), picking up `(q_atom_idx, kv_idx, num_kv)` tuples from the `PagedMQALogitsScheduler` until exhausted (`fetch_next_task` loop). `kNumSMs` is the device SM count (132 on H100/B200), queried via `dg.get_num_sms()` (`vllm/utils/deep_gemm.py:246-251`).
- **Threads/CTA**: `kNumSpecializedThreads + kNumMathThreads`, typically `128 + 128 = 256` or `128 + 256 = 384`. Warp specialization:
  - Warp `kSpecWarpStart + 0`: TMA-Q loader.
  - Warp `kSpecWarpStart + 1`: TMA-KV loader (with block-table broadcast via `__shfl_sync`).
  - Warp `kSpecWarpStart + 2`: UMMA (tcgen05) issuer — drives `UMMA::make_instr_desc_block_scaled` MMA for FP4, or standard FP8 UMMA for the FP8 path.
  - Math warps (warpgroups 0 .. `kNumMathWarpGroups`-1): per-warpgroup `BLOCK_KV=128`-wide accumulation; final fp32 reduction across heads + weight multiply.
- Autotune configs: **JIT-selected** by DeepGEMM at runtime based on `kNextN`, `kNumHeads`, `kHeadDim`, `BLOCK_KV`, `SPLIT_KV`, and SM count. The wrapper does NOT expose tuning knobs to callers.
- Tensor memory (TMEM): allocated via `cute::TMEM::Allocator1Sm().allocate(kNumTmemCols, tmem_ptr_in_smem)`. `kNumTmemCols ≤ 512` (asserted). For FP4: `kNumTmemCols = kNextNAtom * kNumHeads * kNumTmemStages + kNumSFQAtom/32 + kNumSFKV/32` (block-scale lanes added).
- TMA prefetch: kernel issues `cute::prefetch_tma_descriptor` for `tensor_map_q`, `tensor_map_sf_q` (FP4), `tensor_map_kv`, `tensor_map_sf_kv`/`tensor_map_kv_scales`, `tensor_map_weights` at warp `kSpecWarpStart`.
- Grid-dependency sync: `cudaGridDependencySynchronize()` at line 136 (FP8) / line 157 (FP4) — kernel is launched with PDL and waits for the producer (Q/K compress kernels) to complete before TMA-issuing.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:420-421`:
```python
index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
```
The vLLM/DeepGEMM kernel fuses the einsum + weight-multiply + head-sum, **without the ReLU** (the ReLU is handled downstream by the topk kernels' negative-value filtering). Per-token Q quantization is dequant'd in-line via folded `weights` (FP8) or per-block UE8M0 scales (MXFP4).

```python
# Per-(query position q ∈ [0, B*next_n), KV-position kv_pos ∈ [0, context_lens[bb])):
bb, nn = divmod(q, next_n)
context_len = context_lens[bb] if kIsContextLens2D=False else context_lens[bb, nn]
if kv_pos >= context_len:
    continue                                              # masked out

# Physical KV-cache slot lookup via block_table.
block_idx_logical = kv_pos // block_size
block_idx_phys    = block_table[bb, block_idx_logical]
slot              = kv_pos %  block_size

# Dequant K[kv_pos, :] (shape [128]) — path-dependent:
if FP8 path:
    k_fp8   = kv_cache[block_idx_phys, slot, 0, : 128]    # [128] float8_e4m3fn (viewed via uint8)
    k_scale = kv_cache[block_idx_phys, slot, 0, 128:132].view(torch.float32)
    k_fp32  = k_fp8.to(torch.float32) * k_scale           # in-register dequant inside UMMA
else:   # MXFP4 path
    k_packed = kv_cache[block_idx_phys, slot, 0, : 64]    # 64 bytes, 128 E2M1 nibbles
    k_scales = kv_cache[block_idx_phys, slot, 0, 64:68]   # 4 UE8M0 bytes (one per 32-elem block)
    # Dequant via UMMA::make_instr_desc_block_scaled<float_e2m1_t, ..., float_ue8m0_t, ...>:
    # block_scale[b] = 2^(scales[b] - 127); k_fp32[b*32:(b+1)*32] = e2m1[...] * block_scale[b]
    k_fp32 = dequant_block_e2m1_ue8m0(k_packed, k_scales)

# Dequant Q[q, :, :] — path-dependent:
if FP8 path:
    q_fp32 = q_values[bb, nn, :, :].to(torch.float32)     # [64, 128]; per-token-per-head q_scale absorbed into weights
elif MXFP4 path:
    q_packed = q_values[bb, nn, :, :]                     # [64, 64] uint8
    q_scales = q_scale[bb, nn, :]                         # [64] int32 (4 UE8M0 bytes per head)
    q_fp32   = dequant_block_e2m1_ue8m0(q_packed, q_scales)

# MQA logit = Σ_h weights[q, h] * (q_fp32[h] · k_fp32) — fused inside UMMA.
logits[q, kv_pos] = (weights[q, :].unsqueeze(-1) * (q_fp32 * k_fp32).sum(dim=-1)).sum(dim=0)
```

Notes on fusion / numerics:
- **MQA semantics**: K has no head dim (`[kv_pos, head_dim]`, not `[kv_pos, h, head_dim]`). All 64 Q heads attend to the SAME K — that's why the kernel template parameter is `kNumHeads` not `kNumKVHeads`.
- **Block-scaled MMA (FP4 path)**: uses `cutlass::float_e2m1_t` Q and K, `cutlass::float_ue8m0_t` scales, via `cute::UMMA::make_instr_desc_block_scaled<E2M1, E2M1, F32, UE8M0, UMMA_M=128, UMMA_N=kNextNAtom*64, UMMA::Major::K, UMMA::Major::K>` (line 283-284 of `sm100_fp4_paged_mqa_logits.cuh`). UMMA does block dequant in tensor cores — no explicit fp32 round-trip in shared memory.
- **Per-token K scale (FP8 path)**: stored as fp32 (4 bytes), not UE8M0. Inherited from V3.2 cache layout. UMMA descriptor is standard FP8 (no block-scaled variant) and the kernel applies the scalar scale in an explicit fp32 multiply post-UMMA.
- **Weights as in-line per-token Q dequant (FP8 path)**: the `weights` tensor passed to the kernel already includes the per-token-per-head q_scale (folded by `fused_indexer_q_rope_quant`), so the kernel simply multiplies `accum_per_head * weights[q, h]` and sums — there is no separate q_scale tensor.
- **`clean_logits=False`** for decode (line 332 of sparse_attn_indexer.py): the downstream topk kernel masks via `seq_lens`; unfilled logit slots are NOT zeroed. Setting True would invoke `smxx_clean_logits.cuh` to write `-inf` post-hoc.
- **TMA OOB zero-fill** for odd `kNextN ≥ 3` (line 58-59 comment): non-varlen odd N is padded to N+1 via TMA boundary semantics, then the extra row is masked in the math warp.
- **`schedule_metadata` rationale**: SMs grab `(q_atom, kv_block_range)` tuples from a precomputed scheduling table, distributing work so each SM has roughly equal KV-block load even with skewed context lengths. Built by `get_paged_mqa_logits_metadata(context_lens, block_size, num_sms)` at `sparse_attn_indexer.py:262` (call site in `attn_metadata_narrowed.decode.schedule_metadata`).

## Config-dependent dispatch

- Activation condition: always selected on NVIDIA (non-XPU) decode path. XPU goes through `xpu_fp8_paged_mqa_logits` (out of scope).
- Variants on NVIDIA (single Python wrapper, two CUDA kernels selected by the runtime `q_scale` tensor):
  - **FP8 K + FP8 Q** (`use_fp4_cache=False`): wrapper passes `(q_fp8, None)` → DeepGEMM internally dispatches to `sm100_fp8_paged_mqa_logits.cuh`. Coupled with K-side spec `fused_kv_compress_norm_rope_insert_indexer_attn.md` and Q-side spec `fused_indexer_q_rope_quant.md`.
  - **MXFP4 K + MXFP4 Q** (`use_fp4_cache=True`): wrapper passes `(q_packed, q_scale_int32)` → dispatched to `sm100_fp4_paged_mqa_logits.cuh`. Coupled with K-side spec `fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn.md` and Q-side spec `fused_indexer_q_rope_mxfp4.md`.
  - **`use_fp4_cache` coupling (D3)**: same flag flips this kernel AND both compress/Q-side kernels. **DECISION REQUIRED**: pick FP8 OR MXFP4 consistently across the 5-kernel cluster (this kernel + 2 K-side variants + 2 Q-side variants).
  - **SM90 fallback (Class A locked-alternative, D5)**: `sm90_fp8_paged_mqa_logits.cuh` is the SM90 FP8 path — no FP4 on SM90. Locked-out for V4-Flash on B200 (SM100).
- Companion prefill kernel: `fp8_fp4_mqa_logits` (spec `fp8_fp4_mqa_logits.md`) for `has_prefill` case — same FP8 vs MXFP4 dispatch, no paging, uses `cu_seqlen_ks/ke` ranges instead of a block table + schedule.
- Class: **A** (FP8 always-on for V4-Flash with `use_fp4_cache=False`; MXFP4 always-on for `use_fp4_cache=True`). The FP8/FP4 split itself is Class B because it's user-config-driven (`use_fp4_cache`), but the dispatch is unified through one Python wrapper.
- Downstream consumer constraint: output is fp32 `[B*next_n, max_model_len]`, consumed by `top_k_per_row_decode` / `persistent_topk`. The unfilled (masked) slots must be either `-inf` (if `clean_logits=True`) or left to the downstream `seq_lens` mask — current V4-Flash path uses the latter.
