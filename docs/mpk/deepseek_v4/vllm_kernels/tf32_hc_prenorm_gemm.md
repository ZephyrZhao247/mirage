# tf32_hc_prenorm_gemm

## Identity
- Source file: `vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh:42-346` (CUDA `__global__` body `sm100_tf32_hc_prenorm_gemm_impl`; arch-guarded with `#if __CUDA_ARCH__ >= 1000`, asserts `sm_100f` on lower archs at line 344).
- Python wrapper: `vllm/utils/deep_gemm.py:460-483` (`tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_split)` → forwards to `_tf32_hc_prenorm_gemm_impl`, which is the symbol `deep_gemm._C.tf32_hc_prenorm_gemm` re-exported at `vllm/third_party/deep_gemm/__init__.py:70`).
- Language/DSL: **Dense Cutlass-style CUDA** (DeepGEMM v2.5.0). Uses CUTLASS 3 cluster transaction barriers, TMA load/store (`cute::SM90_TMA_STORE_2D/3D`), `tcgen05` tensor-memory MMA (`SM100_MMA_TF32_TS`), and 1-SM tensor-memory allocation (`cute::TMEM::Allocator1Sm`). Compute is **TF32** (`cutlass::tfloat32_t` UMMA) with bf16 A-cast in shared memory.
- Third-party dep: **DeepGEMM** (`vllm/third_party/deep_gemm/`, version 2.5.0). Built into the `deep_gemm._C` Pybind extension; the host-side launcher (block/cluster shape selection, TMA descriptor build, JIT cache) lives inside the `_C` module and is opaque from Python — only the `tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_split)` entry is reachable from vLLM.
- Registered as: plain Python callable (not a `torch.ops` op). vLLM gates availability through `vllm.utils.deep_gemm.is_deep_gemm_supported()`; when False the caller falls back to TileLang (`_tilelang_hc_prenorm_gemm`).

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/model_executor/kernels/mhc/tilelang.py:195` | `mhc_pre_tilelang` (in `torch.ops.vllm.mhc_pre_tilelang`) | `residual_2d: [T, hc_mult * H]`, `fn: [mix_hc, hc_mult * H]`, `gemm_out_mul: [n_splits, T, mix_hc]`, `gemm_out_sqrsum: [n_splits, T]` | A: bf16, B: fp32, D: fp32 | `use_deep_gemm = is_deep_gemm_supported()` (SM100 + `deep_gemm._C` loaded). Fallback: `_tilelang_hc_prenorm_gemm` (tilelang.py:203). |
| `vllm/model_executor/kernels/mhc/tilelang.py:491` | `mhc_fused_post_pre_tilelang` (in `torch.ops.vllm.mhc_fused_post_pre_tilelang`) — only when `use_small_fma` is False (the unfused post→pre path) | Same shapes as above; here `residual_cur_2d` replaces `residual_2d` | Same | Same `use_deep_gemm` gate. |

Top-level callers on V4-Flash: `DeepseekV4DecoderLayer.forward` (`vllm/models/deepseek_v4/nvidia/model.py:874` first layer → `mhc_pre_tilelang`; `nvidia/model.py:888` subsequent layers → `mhc_fused_post_pre_tilelang`).

For V4-Flash B200 (`hc_mult=4`, `dim/hidden_size=4096`):
- `hc_mult * H = 4 * 4096 = 16384` (SHAPE_K)
- `mix_hc = (2 + hc_mult) * hc_mult = 24` (SHAPE_N)
- `T = num_tokens` (M, runtime)
- `n_splits` chosen by `compute_num_split(64, hc_mult*H, ceil(T/64))` at tilelang.py:172, typically 1 for small T and a small split-K factor for large T.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `x` (TMA-mapped as `tensor_map_a`) | `[M = T, SHAPE_K = hc_mult * H]` (V4-Flash: `[T, 16384]`) | bf16 | row-major, contiguous; TMA descriptor built host-side with K-major (`kMajorA = K`). Per-stage SMEM tile: `[BLOCK_M, BLOCK_K] = [BLOCK_M, BLOCK_K]` bf16 (swizzle `kSwizzleAMode = min(BLOCK_K*2, 128)`) | Flattened HC residual stream `(T, hc * H)` (`residual_flat.view(num_tokens, hc_mult * hidden_size)` from tilelang.py:193) |
| `fn` (TMA-mapped as `tensor_map_b`) | `[SHAPE_N = mix_hc, SHAPE_K]` (V4-Flash: `[24, 16384]`) | fp32 | row-major, contiguous; TMA descriptor K-major (`kMajorB = K`). Per-stage SMEM tile: `[BLOCK_N, BLOCK_K]` fp32 (swizzle `kSwizzleBMode = min(BLOCK_K*4, 128)`) | Learned HC mixing matrix (`hc_fn` parameter; per-layer `[mix_hc, hc_mult * H]` fp32 from the reference checkpoint) |
| `num_split` | scalar int | — | — | Split-K factor `kNumSplits`. Chosen by `compute_num_split(64, hc_mult*H, ceil(T/64))` at tilelang.py:172. Becomes a template/JIT-compile constant inside DeepGEMM. |
| `out` (TMA-mapped as `tensor_map_d`) | `[kNumSplits, M, SHAPE_N]` (V4-Flash typical: `[1..N, T, 24]`) | fp32 | pre-allocated; row-major. When `kNumSplits == 1` written via 2D TMA store (`SM90_TMA_STORE_2D`, line 257); when `kNumSplits > 1` written via 3D TMA store (line 259) where the leading dim is the K-split axis | Pre-allocated split-K GEMM output. Caller reduces across split-0 dim in the subsequent TileLang fuse kernel (`mhc_pre_big_fuse_tilelang`). |
| `sqr_sum` (raw global pointer, **not** TMA) | `[kNumSplits, M]` (V4-Flash typical: `[1..N, T]`) | fp32 | pre-allocated; row-major, contiguous on M. Written at line 339 by the cast-and-reduce warp group: `sqr_sum[m_offset + m_idx]` where `m_offset = shape_m * k_split_idx` | Per-token squared-sum of the bf16-cast residual (consumed by the downstream RMSNorm fuse: `rsqrt(sqr_sum/(hc*H) + eps)`) |

Internal compile-time configs (template params, picked by the DeepGEMM JIT host launcher in `_C`):
| Constant | Meaning | V4-Flash typical |
| --- | --- | --- |
| `SHAPE_N`, `SHAPE_K` | GEMM N, K dims baked at JIT time | 24, 16384 |
| `BLOCK_M`, `BLOCK_N`, `BLOCK_K` | Per-cluster tile | `BLOCK_M ∈ {64, 128}` (asserted at line 222), `BLOCK_N == mix_hc = 24`, `BLOCK_K = 64` for split-K dispatch in tilelang.py:170 |
| `kNumStages` | TMA load pipeline depth | DeepGEMM picks ≥ `kNumCastStages = 2` (asserted line 57) |
| `kNumCastStages` | Cast-warp ↔ MMA-warp ping-pong depth | hard-coded 2 (line 52) |
| `kSwizzleCDMode` | D smem swizzle (must equal `BLOCK_N * sizeof(float)`) | 96 (= 24 × 4) — asserted line 58 |
| `kSwizzleAMode`, `kSwizzleBMode` | A/B smem swizzles | `min(BLOCK_K*sizeof(bf16), 128) = 128`, `min(BLOCK_K*sizeof(fp32), 128) = 128` |
| `kNumMMAThreads` | MMA + epilogue thread count | hard 128 (asserts at lines 59, 229) |
| `kNumCastAndReduceThreads` | Cast/reduce warp-group thread count | hard 128 (assert line 270); requires `BLOCK_M == 64` (assert line 269) |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `out` (D) | `[kNumSplits, T, mix_hc]` (V4-Flash: `[ns, T, 24]`) | fp32 | TMA-written from swizzled SMEM (line 257/259). Within each `(k_split_idx, M-block)`, BLOCK_M × BLOCK_N tile is contiguous in fp32. Down-stream `mhc_pre_big_fuse_tilelang` reduces over `k_split_idx` to produce `mix[t, :mix_hc]`. | `out[k, t, :] = x[t, :].float() @ fn[:, :].T` (slice of K assigned to split `k`); summed over splits by the next kernel. |
| `sqr_sum` | `[kNumSplits, T]` (V4-Flash: `[ns, T]`) | fp32 | Direct global store via `sqr_sum[m_offset + m_idx] = reduced_sum` (line 339). `m_offset = shape_m * k_split_idx` partitions the buffer by split. Only `lane_idx % 4 == 0` and `m_idx < shape_m` write (line 338). | `sqr_sum[k, t] = sum over the k-th K-shard of (x[t, j_in_shard].float())**2`. Summed over splits in the fuse kernel to form the RMSNorm denominator. |

The wrapper expects D and sqr_sum **pre-allocated by the caller** (`gemm_out_mul`, `gemm_out_sqrsum` at tilelang.py:186-191) — DeepGEMM writes into the provided buffers.

## Grid / Block

- **Grid**: `grid_dim.x = ceil(M / BLOCK_M) * kNumSplits`. Decoded inside the kernel as `m_block_idx = blockIdx.x / kNumSplits`, `k_split_idx = blockIdx.x % kNumSplits` (lines 130-131). `blockIdx.y/z` unused. **Cluster**: 1×1×1 (1-SM allocator at line 122; the kernel is annotated `__launch_bounds__(kNumMMAThreads + kNumCastAndReduceThreads, 1)` so it runs as a single CTA per cluster member; cluster geometry is set by the launcher).
- **Block**: `kNumMMAThreads + kNumCastAndReduceThreads = 128 + 128 = 256` threads. Decomposed into 8 warps (warps 0-3 = MMA group, warps 4-7 = cast-and-reduce group) via `warp_idx < kNumMMAThreads / 32` at line 140.
- **Per-CTA roles** (warp-specialized):
  - **Warp 0**: TMA-load producer. Issues `tma::copy` for `smem_a[stage]` and `smem_b[stage]` then `arrive_and_expect_tx` on `full_barriers[stage]` (lines 142-159). One iteration per K-stage of this split: `num_total_stages = kNumKBlocksPerSplit + (k_split_idx < kRemainKBlocks)` (line 134).
  - **Warp 1**: UMMA issue. Waits on `full_cast_barriers[cast_stage]`, then for each `k ∈ [0, BLOCK_K/UMMA_K)` issues `umma_t::fma` with `cast_stage * BLOCK_K + k * UMMA_K` as A-offset in **tensor memory** (the TF32 A operand was written there by the cast warp group). After the final MMA, arrives on `tmem_full_barrier` (line 212).
  - **Warps 0-3 (after MMA done)**: Epilogue. `STSM`-style load tensor memory into registers (`SM100_TMEM_LOAD_32dp32b4x`, line 243), store to swizzled SMEM (`ptx::st_shared`), then `SM90_TMA_STORE_2D/3D` to global `D`.
  - **Warps 4-7**: Cast and reduce. Wait on `full_barriers[stage]`, `ldsm` A from SMEM, convert bf16 → tf32 (`__bfloat1622float2`) into **tensor memory** for the MMA, **and** ffma the same values into a per-thread float2 squared-sum accumulator (`__ffma2_rn(fp32x2_values[u][i], fp32x2_values[u][i], sum[u])`, line 316). After all K-stages of this split, `warp_reduce_sum<4>` (4-lane butterfly because 4 lanes per row in tmem layout F at `BLOCK_M=64`) and a single store per (warp, lane-quad) at line 339.
- **Autotune**: DeepGEMM JIT host-side autotuning (in `_C`). Template params (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `kNumStages`, `kNumSplits`, `kSwizzleCDMode`) are selected by the launcher; this kernel file is one of many specializations. Asserts inside the body constrain valid shapes (e.g., `BLOCK_M ∈ {64, 128}` at line 222 for the epilogue; `BLOCK_M == 64` at line 269 for the cast group; A/B swizzle must equal `BLOCK_K * sizeof(elt)` at lines 53-54).
- **PDL / grid dep**: `cudaGridDependencySynchronize()` at line 137 — programmatic dependent launch waits for the previous TileLang kernel (compressor or `mhc_post`) to clear its dependency. No `cudaGridDependencyLaunch` issued by this kernel; the next launch (`mhc_pre_big_fuse_tilelang`) calls `T.pdl_sync()/T.pdl_trigger()` to chain in.
- **Shared memory layout** (lines 86-101): `[D-tile | A0..A_{S-1} | B0..B_{S-1} | barriers (kNumStages*4 + 1) | tmem_ptr]`. Aligned to 1024 bytes via `__align__(1024) uint8_t smem_buffer[]` (line 66). Total `≈ BLOCK_M*BLOCK_N*4 + kNumStages*(BLOCK_M*BLOCK_K*2 + BLOCK_N*BLOCK_K*4) + (kNumStages*4 + 1)*sizeof(Barrier) + 4` bytes.
- **Tensor memory**: `kNumTmemCols = get_num_aligned_tmem_cols<BLOCK_K * kNumCastStages + BLOCK_N>()`, asserted in `[32, 512]` at line 105. Allocated by warp 2 (line 122), freed by warp 1 in the epilogue (line 267).

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:674-680` (the `Block.hc_pre` pre-block: `x.square().mean(-1)` for RMS and `F.linear(x, hc_fn)` for the HC mix projection). The kernel computes BOTH halves of `hc_pre`'s arithmetic in one launch: the GEMM `x @ fn.T` and the per-token `sum(x*x)`, sharing the same bf16-cast-from-fp32 traversal of `x` so K is streamed exactly once.

```python
# PyTorch-operator equivalent of one CTA's work for split k_split_idx and M-block m_block_idx:
#
# Inputs (caller view; tilelang.py:193 flattens residual to 2D):
#   x:        [T, hc*H]            bf16    (e.g. [T, 16384] on V4-Flash)
#   fn:       [mix_hc, hc*H]       fp32    (e.g. [24, 16384])
#   sqr_sum:  [num_split, T]       fp32    (pre-allocated)
#   out:      [num_split, T, mix_hc] fp32  (pre-allocated)
#
# kNumKBlocks = ceil(hc*H / BLOCK_K)
# Each split owns a contiguous slice of K-blocks:
k0     = k_split_idx * kNumKBlocksPerSplit + min(k_split_idx, kRemainKBlocks)
n_kb   = kNumKBlocksPerSplit + (k_split_idx < kRemainKBlocks)
K_slice = slice(k0 * BLOCK_K, (k0 + n_kb) * BLOCK_K)

m0, m1 = m_block_idx * BLOCK_M, (m_block_idx + 1) * BLOCK_M

# --- Fused per-CTA work ---
# 1. The bf16 A operand is also the squared-sum source.
x_bf16   = x[m0:m1, K_slice]                              # [BLOCK_M, n_kb*BLOCK_K] bf16
x_fp32   = x_bf16.float()                                 # TF32-representable subset of fp32
fn_fp32  = fn[:, K_slice]                                 # [mix_hc, n_kb*BLOCK_K] fp32

# 2. TF32 UMMA-accumulated partial GEMM (lines 188-209).
#    NOTE: the cast warp group writes x_fp32 into tensor memory *with TF32 representation
#    implied by SM100_MMA_TF32_TS* — the mantissa is truncated to 10 bits before the
#    multiply. This is the source of the ~1e-3 numerical drift vs the PyTorch fp32 ref.
out_partial = x_fp32 @ fn_fp32.T                          # [BLOCK_M, mix_hc] fp32
out[k_split_idx, m0:m1, :] = out_partial                  # TMA-store, 2D or 3D

# 3. Squared-sum is computed in the cast warp group from x_fp32 (line 316),
#    NOT from x_bf16 — the FFMA reads the same fp32 values it just wrote to tmem.
sqr_partial = (x_fp32 * x_fp32).sum(dim=-1)               # [BLOCK_M] fp32
sqr_sum[k_split_idx, m0:m1] = sqr_partial                 # warp-reduced (warp_reduce_sum<4>),
                                                          # then lane_idx%4 == 0 stores.

# Downstream fuse (mhc_pre_big_fuse_tilelang) reduces over k_split_idx:
#   sqrsum_total[t] = sqr_sum[:, t].sum()
#   mix[t, :]       = out[:, t, :].sum(0)
#   rsqrt           = torch.rsqrt(sqrsum_total / (hc*H) + rms_eps)
#   pre, post, comb = hc_split_sinkhorn(mix * rsqrt, hc_scale, hc_base, ...)
```

Fusion notes:
- **Single K-traversal for two outputs**: `x` is touched exactly once. The bf16 → fp32 conversion happens inside the cast warp group (line 315) and the result is used for both (a) the squared-sum FFMA (line 316) and (b) the tmem store that feeds the TF32 MMA (line 321). Doing GEMM and sqr-sum separately would require two reads of `x`.
- **TF32 precision contract**: A is bf16 in shared memory and **tf32** at the MMA. The cast-and-reduce warp group materializes A in tensor memory with the bit layout `SM100_MMA_TF32_TS` expects (TF32 = sign + 8-bit exp + 10-bit mantissa). The sqr-sum read uses the full fp32 value (no TF32 mantissa truncation), so the squared-sum is more precise than the GEMM contribution — this is intentional, the RMS denominator must be accurate.
- **Split-K**: K is partitioned across CTAs (not warps within a CTA). Each split writes its own `out[k_split_idx]` and `sqr_sum[k_split_idx]` plane — no atomic add. The split-K reduction (`out.sum(0)`, `sqr_sum.sum(0)`) is done by the next kernel (`mhc_pre_big_fuse_tilelang`), folded into the rsqrt + Sinkhorn pipeline.
- **Layout F (BLOCK_M=64) vs Layout D (BLOCK_M=128)**: The two valid `BLOCK_M`s correspond to two SM100 UMMA accumulator layouts. `BLOCK_M=128` skips the `__syncwarp()` at line 249 because all 32 lanes contribute STS; `BLOCK_M=64` uses only the first 16 lanes per warp (line 246) and the warp resyncs between bank-group iterations.
- **PDL chain**: `griddepcontrol.wait` at the very start (via `cudaGridDependencySynchronize`, line 137) waits for the prior kernel (the compressor / `mhc_post` predecessor) to flush. The kernel does NOT emit `griddepcontrol.launch_dependents`; the immediate downstream `mhc_pre_big_fuse_tilelang` uses `T.pdl_sync()` on its own grid.

## Config-dependent dispatch

- **Activation condition**: V4-Flash, NVIDIA, SM100. Gated by `use_deep_gemm = is_deep_gemm_supported()` in `mhc_pre_tilelang` (tilelang.py:167) — True iff `current_platform.has_device_capability((10, 0))` AND `deep_gemm._C` imported successfully (`vllm/utils/deep_gemm.py:185-233`). On B200 this is True by default.
- **Locked alternative pointer (SM90, H100)**: `vllm/third_party/deep_gemm/include/deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh:42-` — same I/O contract, identical Python entry (DeepGEMM JIT host picks SM90 vs SM100 by `cudaGetDeviceProperties().major`). Uses `SM90_MMA_TF32` warp-group MMA (`mma::sm90::*`), no tensor memory, no `tcgen05`. **Per user D5: NOT a separate spec — this is a one-line pointer.**
- **Locked alternative pointer (TileLang fallback)**: `_tilelang_hc_prenorm_gemm` (tilelang.py:21-87) is used when `is_deep_gemm_supported()` is False (e.g., A100, MI300, or DeepGEMM build skipped). Same I/O contract, lower performance. Not part of V4-Flash's SM100 hot path.
- **No Class B branch on V4-Flash**: `expert_dtype=fp4` + `moe_backend=deep_gemm_mega_moe` does not change this kernel's selection; the dispatch is purely platform-based.
- **Downstream consumer constraints**:
  - `gemm_out_mul` and `gemm_out_sqrsum` must be allocated by the caller with the exact shapes `[n_splits, T, mix_hc]` and `[n_splits, T]` (tilelang.py:186-191). DeepGEMM does NOT reshape — the kernel writes into the buffers as flat `(M, mix_hc)` or `(M,)` slices indexed by `m_offset = shape_m * k_split_idx`, so wrong allocation strides corrupt the next kernel's split-reduction.
  - `fn` must be contiguous fp32 with `fn.shape == (mix_hc, hc_mult * H)` — the TMA descriptor is built with K-major and that shape encoded as compile-time `SHAPE_N`, `SHAPE_K`. Wrong shape → JIT cache miss / recompile or out-of-bounds TMA.
  - `x` must be contiguous bf16 with `x.shape == (T, hc_mult * H)`; `T` can vary at runtime (passed as `shape_m`), but `hc_mult * H` is JIT-baked.
- **Hard preconditions** (from the kernel asserts in the source body):
  - `kNumCastStages <= kNumStages` (line 57)
  - `kSwizzleCDMode / sizeof(float) == BLOCK_N` (line 58)
  - `BLOCK_M ∈ {64, 128}` for the epilogue (line 222); `BLOCK_M == 64` for the cast-and-reduce path (line 269)
  - `kNumMMAThreads == kNumCastAndReduceThreads == 128` (lines 59, 270)
  - `BLOCK_K * sizeof(bf16) == kSwizzleAMode` (line 275)
  - UMMA shape constraint at lines 181-184: `(BLOCK_M, BLOCK_N)` ∈ a closed set; `BLOCK_N % 8 == 0` and `BLOCK_N ≤ 256`. V4-Flash's `BLOCK_N = 24` satisfies this.
- **Web-search confirmed**: DeepGEMM v2.5.0 (`vllm/third_party/deep_gemm/__init__.py:126`) is the upstream source; the `sm100_tf32_hc_prenorm_gemm` kernel was added specifically for DeepSeek V4-Flash hyper-connection pre-blocks. The host launcher's auto-tuner caches per-`(M, N, K, num_split)` JIT compilation in `~/.cache/deep_gemm/`.
