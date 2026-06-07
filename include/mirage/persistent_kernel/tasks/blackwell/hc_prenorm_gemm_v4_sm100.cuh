/* Copyright 2026 Mirage Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once
#include "tasks/common/common_header.cuh"

// V4-Flash hc_prenorm_gemm (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/hc_prenorm_gemm_tilelang.md
//
// Computes the *un-reduced split-K partials* of both
//   gemm_out_mul[s, t, j] = sum_{k in K_s} x[t, k].float() * fn[j, k]
//   gemm_out_sqrsum[s, t] = sum_{k in K_s} x[t, k].float()**2
// for s in [0, N_SPLITS). For the naive port we hard-code N_SPLITS = 1
// (the wrapper's default when DeepGEMM is the fallback path; DeepGEMM
// can pick larger n_splits but the consumer kernel transparently reduces
// over the leading split dim regardless of its size).
//
// Naive design (correctness only, no perf):
//   * One CTA per (token), grid = (num_tokens, 1, 1).
//   * NUM_THREADS = 256 (Blackwell default WORKER_NUM_THREADS).
//   * For each output index j in [0, HC_MULT3), accumulate the K-axis
//     dot product in fp32 with a thread-strided loop, then warp + cross-
//     warp reduce. Same loop also accumulates the per-token sum-of-
//     squares (only on j == 0 to avoid redundant work, matching the
//     "i_t == 0 writes sqrsum" pattern of the TileLang reference).
//   * Single CTA, single K-pass per output index. No TMA / no UMMA /
//     no warp-specialization / no split-K orchestration.
//
// Input/output dtypes match the vLLM contract (and the downstream
// mhc_pre_big_fuse_* consumer):
//   * x:           bf16 [num_tokens, K] (K = HC_MULT * HIDDEN = 16384)
//   * fn:          fp32 [HC_MULT3,  K]  (HC_MULT3 = HC_MULT * (2 + HC_MULT) = 24)
//   * gemm_out:    fp32 [N_SPLITS, num_tokens, HC_MULT3]
//   * gemm_sqrsum: fp32 [N_SPLITS, num_tokens]
//
// The TBGraph partitions `x`, `gemm_out`, `gemm_sqrsum` on the token
// dim (dim 0 of the bf16 [T,K] view; dim 1 of the [1, T, 24] gemm_out
// tensor, but the per-token contiguous row matches what the runtime
// preoffsets). `fn` is broadcast (no partition). The codegen offsets
// each CTA's input/output pointers to the per-token slice; the kernel
// itself does NOT index by token.

namespace kernel {

namespace hc_prenorm_gemm_v4_detail {

template <int NUM_THREADS>
__device__ __forceinline__ float warp_block_reduce_sum(float val,
                                                       float *smem) {
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
#pragma unroll
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    val += shfl_xor_sync(val, offset);
  }
  int lane = threadIdx.x % NUM_THREADS_PER_WARP;
  int warp = threadIdx.x / NUM_THREADS_PER_WARP;
  if (lane == 0) {
    smem[warp] = val;
  }
  __syncthreads();
  float out = (threadIdx.x < NUM_WARPS) ? smem[threadIdx.x] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
      out += shfl_xor_sync(out, offset);
    }
    if (lane == 0) {
      smem[0] = out;
    }
  }
  __syncthreads();
  return smem[0];
}

} // namespace hc_prenorm_gemm_v4_detail

// Shared naive impl. Used by:
//   - hc_prenorm_gemm_v4_sm100         (output rank 3: [N_SPLITS=1, T, HC_MULT3])
//   - hc_prenorm_gemm_block_m_v4_sm100 (same output, alias for catalog parity)
//   - tf32_hc_prenorm_gemm_v4_sm100    (same output, B200/sm_100a only)
//
// All three TileLang/DeepGEMM variants produce IDENTICAL outputs at
// n_splits=1; the only difference upstream is tiling strategy. For the
// naive port a single shared kernel covers all three.
template <int HC_MULT, int HIDDEN, int NUM_THREADS = 256>
__device__ __forceinline__ void hc_prenorm_gemm_v4_sm100_impl(
    void const *x_ptr,           // bf16 [K]  (this token's flattened residual)
    void const *fn_ptr,          // fp32 [HC_MULT3, K]
    void *gemm_out_ptr,          // fp32 [HC_MULT3] (this token's row in [1,T,HC_MULT3])
    void *gemm_sqrsum_ptr        // fp32 [1]        (this token's scalar in [1,T])
) {
  static_assert(HC_MULT > 0, "HC_MULT must be positive");
  static_assert(HIDDEN > 0, "HIDDEN must be positive");
  static_assert(NUM_THREADS > 0, "NUM_THREADS must be positive");
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a warp multiple");

  constexpr int K = HC_MULT * HIDDEN;                // flattened K
  constexpr int HC_MULT3 = HC_MULT * (2 + HC_MULT);  // 24 when HC_MULT=4

  using bf16 = type::bfloat16_t;

  bf16 const *__restrict__ x_in = static_cast<bf16 const *>(x_ptr);
  float const *__restrict__ fn = static_cast<float const *>(fn_ptr);
  float *__restrict__ gemm_out = static_cast<float *>(gemm_out_ptr);
  float *__restrict__ sqrsum_out = static_cast<float *>(gemm_sqrsum_ptr);

  extern __shared__ char smem[];
  float *reduce_smem = reinterpret_cast<float *>(smem);

  // --- pass: for each output j, dot(x, fn[j]) (fp32). Also accumulate
  // the per-token sum-of-squares on the j==0 pass to share the K read.
#pragma unroll 1
  for (int j = 0; j < HC_MULT3; ++j) {
    float dot_partial = 0.0f;
    float sqr_partial = 0.0f;

    float const *__restrict__ fn_row = fn + j * K;

    for (int k = threadIdx.x; k < K; k += NUM_THREADS) {
      float v = static_cast<float>(x_in[k]);
      dot_partial += v * fn_row[k];
      if (j == 0) {
        sqr_partial += v * v;
      }
    }

    float dot = hc_prenorm_gemm_v4_detail::warp_block_reduce_sum<NUM_THREADS>(
        dot_partial, reduce_smem);
    if (threadIdx.x == 0) {
      gemm_out[j] = dot;
    }
    __syncthreads();

    if (j == 0) {
      float sqr = hc_prenorm_gemm_v4_detail::warp_block_reduce_sum<NUM_THREADS>(
          sqr_partial, reduce_smem);
      if (threadIdx.x == 0) {
        sqrsum_out[0] = sqr;
      }
      __syncthreads();
    }
  }
}

} // namespace kernel
