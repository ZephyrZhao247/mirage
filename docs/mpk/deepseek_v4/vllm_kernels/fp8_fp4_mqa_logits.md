# fp8_fp4_mqa_logits

## Identity
- Python entry: `vllm/utils/deep_gemm.py:341-383` — dispatcher `fp8_fp4_mqa_logits(q, kv, weights, cu_seqlen_ks, cu_seqlen_ke, clean_logits)`.
- Symbol resolved (line 228) via `getattr(_dg, "fp8_fp4_mqa_logits", None)` from the `deep_gemm` package (vendored at `vllm/third_party/deep_gemm/`, exported in `vllm/third_party/deep_gemm/__init__.py:63`).
- Underlying CUDA implementations:
  - **FP8 Q + FP8 K**: `vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp8_mqa_logits.cuh:29-403` (template `sm100_fp8_mqa_logits<kNumHeads, kHeadDim, kIsCompressedLogits, BLOCK_Q, BLOCK_KV, kNumQStages, kNumKVStages, kNumSMs, kNumSpecializedThreads, kNumMathThreads, logits_dtype_t>`).
  - **MXFP4 Q + MXFP4 K (block-scaled E2M1)**: `vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp4_mqa_logits.cuh:27-457` (template `sm100_fp4_mqa_logits<kNumHeads, kHeadDim, kIsCompressedLogits, BLOCK_Q, BLOCK_KV, kNumQStages, kNumKVStages, kNumSMs, kNumSpecializedThreads, kNumMathThreads, logits_dtype_t>`).
  - SM90 fallback (locked-alternative pointer): `vllm/third_party/deep_gemm/include/deep_gemm/impls/sm90_fp8_mqa_logits.cuh` (FP8 only — no FP4 on SM90).
- Language/DSL: **CUDA + CUTLASS/CuTe** (TMA + UMMA on SM100 / tcgen05; block-scaled MMA for FP4).
- Third-party dep: **DeepGEMM**.
- Registered as opaque custom op: **no**.

This is the **prefill** companion of `fp8_fp4_paged_mqa_logits` — same MQA logits computation but no KV paging (KV is contiguous per chunk, addressed by `[cu_seqlen_ks, cu_seqlen_ke)` ranges instead of a block table).

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/model_executor/layers/sparse_attn_indexer.py:233` | `sparse_attn_indexer` prefill chunk loop | `q=(q_slice_cast, q_scale_slice)`, `kv=(k_quant_cast, k_scale_cast)`, `weights: [T_chunk, 64]`, `cu_seqlen_ks: [T_chunk] int32`, `cu_seqlen_ke: [T_chunk] int32` | FP8 path: q `float8_e4m3fn`, k `float8_e4m3fn`, k_scale fp32. MXFP4 path: q packed uint8 viewed as int8, q_scale int32; k packed uint8 viewed as int8, k_scale int32. | always selected for prefill in V4-Flash NVIDIA path (non-XPU); FP8 vs MXFP4 controlled by `use_fp4_cache` |

Called once per prefill chunk (`prefill_metadata.chunks` loop, `sparse_attn_indexer.py:192-256`). K is gathered into a workspace via `cp_gather_indexer_k_quant_cache` (line 197-203) before the call — KV is contiguous in `[chunk.total_seq_lens, head_dim_packed]` shape.

V4-Flash uses `kNumHeads = 64`, `kHeadDim = 128`. `BLOCK_Q` and `BLOCK_KV` are JIT-tuned by DeepGEMM.

## Inputs

### Python wrapper inputs (lines 341-371)

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `q` | tuple `(q_values, q_scale)` | see below | — | FP8 path: `q_values=[T_chunk, 64, 128] float8_e4m3fn`, `q_scale=None`. MXFP4 path: `q_values=[T_chunk, 64, 64] uint8` (viewed as `int8`), `q_scale=[T_chunk, 64] int32` (UE8M0 block scales squeezed). |
| `kv` | tuple `(k_packed, k_scales)` | see below | — | FP8 path: `k_packed=[N, 128] float8_e4m3fn`, `k_scales=[N] fp32`. MXFP4 path: `k_packed=[N, 64] uint8` (viewed as `int8`), `k_scales=[N] int32` (4 UE8M0 bytes per token squeezed to int32). N = `chunk.total_seq_lens`. |
| `weights` | `[T_chunk, 64]` | fp32 | row-major | Folded weights from `fused_indexer_q_rope_quant`. |
| `cu_seqlen_ks` | `[T_chunk]` | int32 | contiguous | Start index (inclusive) for valid K per Q position. |
| `cu_seqlen_ke` | `[T_chunk]` | int32 | contiguous | End index (exclusive) for valid K per Q position. |
| `clean_logits` | scalar | bool | — | Whether to clean unfilled slots to `-inf`. Prefill path passes `False` (line 239). |

### CUDA kernel template parameters (`sm100_fp{8,4}_mqa_logits.cuh`)

| Template arg | Value (V4-Flash) | Meaning |
| --- | --- | --- |
| `kNumHeads` | 64 | `index_n_heads`. |
| `kHeadDim` | 128 | `index_head_dim`. |
| `kIsCompressedLogits` | False (typical) | Whether to write compressed logits format. |
| `BLOCK_Q` | JIT-tuned (e.g., 32 or 64) | Q tile per CTA. |
| `BLOCK_KV` | JIT-tuned, typically `kNumMathWarpGroups * UMMA_M = 128 * k` | KV tile. |
| `kNumQStages` / `kNumKVStages` | JIT-tuned | Pipeline depths. |
| `kNumSMs` | 132 (B200) | Persistent kernel grid size — queried via `dg.get_num_sms()`. |
| `kNumSpecializedThreads` | 128 | Producer warps. |
| `kNumMathThreads` | 128 × `kNumMathWarpGroups` | Consumer warps. |
| `logits_dtype_t` | fp32 | Logits dtype. |

### CUDA kernel runtime arguments

| Arg | Type | Meaning |
| --- | --- | --- |
| `seq_len` | uint32 | `T_chunk` (total Q rows in this chunk). |
| `seq_len_kv` | uint32 | `N` (total K rows in this chunk). |
| `max_seqlen_k` | uint32 | Max K-end across the chunk (for TMA bounds). |
| `logits_stride` | uint32 | Output row stride. |
| `cu_seq_len_k_start`, `cu_seq_len_k_end` | `uint32*` | Per-Q K start/end ranges. |
| `logits` | `logits_dtype_t*` | Output `[T_chunk, N]`. |
| `tensor_map_q`, `tensor_map_sf_q` (FP4 only) | `cute::TmaDescriptor` | TMA Q (and scale for FP4). |
| `tensor_map_kv`, `tensor_map_sf_kv` (FP4) / `tensor_map_kv_scales` (FP8) | `cute::TmaDescriptor` | TMA K + scales. FP8 has fp32 per-token scale; FP4 has int32 block scales. |
| `tensor_map_weights` | `cute::TmaDescriptor` | TMA weights. |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `logits` | `[T_chunk, N]` (where `N = chunk.total_seq_lens`) | fp32 | row-major | Per-(Q-row, KV-position) MQA logits = `Σ_h weights[q, h] * (q[q, h, :] · k[kv, :])`, with FP8/MXFP4 dequant in-line. Slots `kv ∉ [cu_seqlen_ks[q], cu_seqlen_ke[q])` are masked by the kernel (left untouched unless `clean_logits=True`). |

Downstream: `top_k_per_row_prefill` — `sparse_attn_indexer.py:247-256`.

## Grid / Block

- **Grid**: `gridDim = (kNumSMs,)` — persistent kernel. Each SM iterates Q-blocks via `for (q_idx = sm_idx; q_idx < num_q_blocks; q_idx += kNumSMs)` (line 188-200 of `sm100_fp4_mqa_logits.cuh`).
- **Threads/CTA**: `kNumSpecializedThreads + kNumMathThreads`. Warp specialization:
  - Warp `kSpecWarpStart + 0`: TMA-Q loader. Issues `cute::SM90_TMA_LOAD_2D` for Q + scale + weights (line 194-198 FP4).
  - Warp `kSpecWarpStart + 1`: TMA-KV loader. Issues per-Q-block KV loads over the `[kv_start, kv_end)` range determined by `load_schedule(q_idx)` (lines 146-160 FP4; intersects all `BLOCK_Q` rows' `cu_seqlen_ks/ke` to find the union KV range).
  - Warp `kSpecWarpStart + 2`: UMMA issuer.
  - Math warps: per-warpgroup accumulator + final fp32 reduction + weight multiply.
- Autotune configs: **JIT-selected** by DeepGEMM.
- TMEM: `kNumTmemCols = BLOCK_Q * kNumHeads * kNumMathWarpGroups + (FP4: kNumSFQ/32 + kNumSFKV/32)` (line 78 FP8; line 113 FP4). Asserted ≤ 512.
- Grid-dependency sync: `cudaGridDependencySynchronize()` at line 180 (FP4) — PDL waits for the K-compress producer to complete.
- KV range coalescing: `load_schedule` (line 146-160 FP4) intersects the `BLOCK_Q` rows' `cu_seqlen_ks/ke` to compute `(kv_start, num_kv_blocks)` for the CTA — TMA loads only the union KV range, not the per-row range. The math warps then apply per-row masking via `seq_k_start[i] / seq_k_end[i]`.
  - TMA scale-factor alignment: `start = start / 4 * 4` (line 158) — KV-scale TMA requires 4-element alignment.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:420-421`:
```python
index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
```
Plus the prefill masking at model.py:424-426:
```python
mask = torch.arange(seqlen // ratio).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
index_score += torch.where(mask, float("-inf"), 0)
```
The vLLM/DeepGEMM kernel fuses the einsum + weight-multiply + head-sum and uses `cu_seqlen_ks/ke` as the explicit causal/window mask (replacing the reference's triangular mask). ReLU is NOT fused — applied by downstream topk.

```python
# Per-(query q ∈ [0, T_chunk), kv ∈ [0, N)) — only writes within valid range.
ks = cu_seqlen_ks[q]; ke = cu_seqlen_ke[q]
if kv < ks or kv >= ke:
    continue                                    # masked; logits[q, kv] untouched

# Dequant K[kv, :] (shape [128]) — path-dependent.
if FP8 path:
    k_fp8   = k_packed[kv, :]                   # [128] float8_e4m3fn
    k_scale = k_scales[kv]                      # scalar fp32
    k_fp32  = k_fp8.to(torch.float32) * k_scale # in-register dequant
else:   # MXFP4 path
    k_packed_v = k_packed[kv, :]                # [64] uint8 (128 E2M1 nibbles)
    k_scales_v = k_scales[kv]                   # int32 (4 UE8M0 bytes packed)
    k_fp32     = dequant_block_e2m1_ue8m0(k_packed_v, k_scales_v)

# Dequant Q[q, :, :].
if FP8 path:
    q_fp32 = q_values[q, :, :].to(torch.float32)  # [64, 128]; per-token q_scale folded into weights
elif MXFP4 path:
    q_fp32 = dequant_block_e2m1_ue8m0(q_values[q], q_scale[q])

# MQA logit (fused inside UMMA).
logits[q, kv] = (weights[q, :].unsqueeze(-1) * (q_fp32 * k_fp32).sum(dim=-1)).sum(dim=0)
```

Notes on fusion / numerics:
- **No paging**: K is contiguous `[N, head_dim_packed]` — TMA descriptor for KV is a single 2-D map, no block-table indirection (contrast with the paged variant which has a block-table broadcast via `__shfl_sync`).
- **`cu_seqlen_ks/ke` ranges**: per-Q-row K-start and K-end. Computed by the prefill metadata builder (`DeepseekV32IndexerMetadata.prefill`) from sequence positions + window size + sparse-topk indices. Each Q row sees a row-specific causal window.
- **Block-scaled MMA (FP4)**: identical to the paged variant — `cute::UMMA::make_instr_desc_block_scaled<E2M1, E2M1, F32, UE8M0, UMMA_M=128, UMMA_N=BLOCK_Q*64, K-major, K-major>`. Per-block scales loaded via TMA into TMEM scale lanes.
- **Per-token K scale (FP8)**: stored as fp32, not UE8M0. Matches the FP8 K-side compressor output (`_fused_kv_compress_norm_rope_insert_indexer_attn`).
- **Weights as in-line per-token Q dequant (FP8)**: `weights` already contains the per-token-per-head q_scale (folded by `fused_indexer_q_rope_quant`).
- **`clean_logits=False`** for prefill (line 239): downstream `top_k_per_row_prefill` masks using `cu_seqlen_ks/ke` directly.

## Config-dependent dispatch

- Activation condition: always selected on NVIDIA prefill path (non-XPU). XPU goes through `xpu_fp8_mqa_logits` (out of scope).
- Variants on NVIDIA (single Python wrapper, two CUDA kernels selected by the runtime `q_scale` tensor):
  - **FP8 K + FP8 Q** (`use_fp4_cache=False`): wrapper passes `q=(q_fp8, None)`, `kv=(k_fp8, k_scales_fp32)` → `sm100_fp8_mqa_logits.cuh`. Coupled with K-side spec `fused_kv_compress_norm_rope_insert_indexer_attn.md` and Q-side spec `fused_indexer_q_rope_quant.md`.
  - **MXFP4 K + MXFP4 Q** (`use_fp4_cache=True`): wrapper passes `q=(q_packed, q_scale_int32)`, `kv=(k_packed, k_scales_int32)` → `sm100_fp4_mqa_logits.cuh`. Coupled with K-side spec `fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn.md` and Q-side spec `fused_indexer_q_rope_mxfp4.md`.
  - **`use_fp4_cache` coupling (D3)**: same flag flips this kernel AND the K-side compress kernels AND the Q-side quant kernels. **DECISION REQUIRED**: pick FP8 OR MXFP4 consistently across the 6-kernel cluster (this kernel + paged sibling + 2 K-side + 2 Q-side).
  - **SM90 fallback (Class A locked-alternative, D5)**: `sm90_fp8_mqa_logits.cuh` is the SM90 FP8 path — no FP4 on SM90. Locked-out for V4-Flash on B200.
- Companion decode kernel: `fp8_fp4_paged_mqa_logits` (spec `fp8_fp4_paged_mqa_logits.md`) for `has_decode` case. The pair shares Python entry semantics but different CUDA implementations because paging changes the KV addressing model.
- Class: **A** (always-on per branch). The FP8/FP4 split is Class B (user config `use_fp4_cache`); single Python wrapper unifies dispatch.
- Downstream consumer constraint: output fp32 `[T_chunk, N]` consumed by `top_k_per_row_prefill`. Masked slots are left untouched (relies on downstream consumer to honor `cu_seqlen_ks/ke`).
