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

// V4-Flash mhc_fused (NAIVE Blackwell SM100 impl, decode-regime
// fused mhc_post + hc_prenorm_gemm + sqrsum).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/mhc_fused_tilelang.md
//
// Per-token semantics (one CTA owns one token t):
//   For each hidden index h in [0, HIDDEN):
//     new_r[hc]  = post_mix[hc] * x_in[h]
//                + sum_i comb_mix[i, hc] * residual_in[i, h]    (HC-wide)
//     residual_out[hc, h] = bf16(new_r[hc])
//     sqr += sum_hc new_r[hc]^2
//     for n in [0, N_OUT):
//       gemm_out_mul[t, n] += sum_hc weight_t[n, hc, h] * new_r[hc]
//
// We expose split_k=1 (kept as a template parameter for forward-compat
// with a future tiled version). The N_OUT axis collapses to a per-token
// fp32 vector of length N_OUT = HC_MULT3 = HC * (2 + HC).
//
// Naive design (one CTA per token, NUM_THREADS=256, no TMA/UMMA/warp-spec):
//   * Grid: (num_tokens, 1, 1). Pointers pre-offset per token.
//   * Stage post_mix (HC fp32) and comb_mix (HC*HC fp32) into shared
//     memory; tiny constants needed by every thread.
//   * Each thread loops over hidden indices h in stride NUM_THREADS:
//       - load residual_in[:, h] and x_in[h]
//       - compute new_r[HC] in registers
//       - store residual_out[:, h] (bf16) and accumulate sqr
//       - FMA into per-thread acc[N_OUT] against weight_t[n, hc, h]
//   * Cross-warp reductions: sqr is a single fp32 scalar (one warp-block
//     reduce); acc[N_OUT] is N_OUT scalars (warp-block reduce each).
//   * Thread 0 writes gemm_out_mul[t, :] and gemm_out_sqrsum[t].
//
// Inputs (in TBGraph order, matches register_..._task() codegen):
//   input_ptrs[0] = comb_mix    fp32 [num_tokens, HC, HC]
//   input_ptrs[1] = residual_in bf16 [num_tokens, HC, HIDDEN]
//   input_ptrs[2] = post_mix    fp32 [num_tokens, HC]
//   input_ptrs[3] = x_in        bf16 [num_tokens, HIDDEN]
//   input_ptrs[4] = weight_t    fp32 [N_OUT, HC, HIDDEN]      (broadcast)
//
// Outputs:
//   output_ptrs[0] = gemm_out_mul    fp32 [SPLIT_K=1, num_tokens, N_OUT]
//   output_ptrs[1] = gemm_out_sqrsum fp32 [SPLIT_K=1, num_tokens]
//   output_ptrs[2] = residual_out    bf16 [num_tokens, HC, HIDDEN]

namespace kernel {

namespace mhc_fused_v4_detail {

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

} // namespace mhc_fused_v4_detail

template <int HC,
          int HIDDEN,
          int N_OUT,
          int SPLIT_K = 1,
          int NUM_THREADS = 256>
__device__ __forceinline__ void mhc_fused_v4_sm100_impl(
    void const *comb_mix_ptr,        // fp32 [HC, HC]    (per-token slice)
    void const *residual_in_ptr,     // bf16 [HC, HIDDEN]
    void const *post_mix_ptr,        // fp32 [HC]
    void const *x_in_ptr,            // bf16 [HIDDEN]
    void const *weight_t_ptr,        // fp32 [N_OUT, HC, HIDDEN] (broadcast)
    void *gemm_out_mul_ptr,          // fp32 [N_OUT]     (this token's row,
                                     //                   split-k slot 0)
    void *gemm_out_sqrsum_ptr,       // fp32 [1]         (this token's slot)
    void *residual_out_ptr) {        // bf16 [HC, HIDDEN]
  static_assert(HC > 0, "HC must be positive");
  static_assert(HIDDEN > 0, "HIDDEN must be positive");
  static_assert(N_OUT > 0, "N_OUT must be positive");
  static_assert(SPLIT_K == 1, "Naive impl only supports SPLIT_K=1");
  static_assert(NUM_THREADS > 0, "NUM_THREADS must be positive");
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a warp multiple");

  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;

  using bf16 = type::bfloat16_t;

  float const *__restrict__ a = static_cast<float const *>(comb_mix_ptr);
  bf16 const *__restrict__ b =
      static_cast<bf16 const *>(residual_in_ptr);
  float const *__restrict__ c = static_cast<float const *>(post_mix_ptr);
  bf16 const *__restrict__ d = static_cast<bf16 const *>(x_in_ptr);
  float const *__restrict__ wt =
      static_cast<float const *>(weight_t_ptr);

  float *__restrict__ gemm_mul =
      static_cast<float *>(gemm_out_mul_ptr);
  float *__restrict__ gemm_sqr =
      static_cast<float *>(gemm_out_sqrsum_ptr);
  bf16 *__restrict__ y = static_cast<bf16 *>(residual_out_ptr);

  // Shared mem layout:
  //   s_a:      HC*HC fp32 comb_mix
  //   s_c:      HC    fp32 post_mix
  //   s_reduce: NUM_WARPS fp32 slots for the warp-block reductions
  __shared__ float s_a[HC * HC];
  __shared__ float s_c[HC];
  __shared__ float s_reduce[NUM_WARPS];

  if (threadIdx.x < HC * HC) {
    s_a[threadIdx.x] = a[threadIdx.x];
  }
  if (threadIdx.x < HC) {
    s_c[threadIdx.x] = c[threadIdx.x];
  }
  __syncthreads();

  // Per-thread N_OUT accumulators and a single sqr accumulator.
  float acc[N_OUT];
#pragma unroll
  for (int n = 0; n < N_OUT; ++n) {
    acc[n] = 0.0f;
  }
  float sqr = 0.0f;

  // Main loop over hidden indices.
  for (int h = threadIdx.x; h < HIDDEN; h += NUM_THREADS) {
    float d_val = static_cast<float>(d[h]);

    // Read residual_in[:, h]: HC bf16 values.
    float r_local[HC];
#pragma unroll
    for (int i = 0; i < HC; ++i) {
      r_local[i] = static_cast<float>(b[i * HIDDEN + h]);
    }

    // Compute new_r[HC]: c[hc] * d_val + sum_i a[i, hc] * r_local[i].
    float new_r[HC];
#pragma unroll
    for (int hc = 0; hc < HC; ++hc) {
      float v = s_c[hc] * d_val;
#pragma unroll
      for (int i = 0; i < HC; ++i) {
        v += s_a[i * HC + hc] * r_local[i];
      }
      new_r[hc] = v;
    }

    // Store residual_out[:, h] bf16 + accumulate sqr.
#pragma unroll
    for (int hc = 0; hc < HC; ++hc) {
      y[hc * HIDDEN + h] = bf16(new_r[hc]);
      sqr += new_r[hc] * new_r[hc];
    }

    // Per-output-row FMA into acc[n]:
    //   acc[n] += sum_hc weight_t[n, hc, h] * new_r[hc].
#pragma unroll
    for (int n = 0; n < N_OUT; ++n) {
      float v = 0.0f;
#pragma unroll
      for (int hc = 0; hc < HC; ++hc) {
        v += wt[n * HC * HIDDEN + hc * HIDDEN + h] * new_r[hc];
      }
      acc[n] += v;
    }
  }

  // Reduce sqr across the block.
  float total_sqr =
      mhc_fused_v4_detail::warp_block_reduce_sum<NUM_THREADS>(sqr, s_reduce);

  if (threadIdx.x == 0) {
    gemm_sqr[0] = total_sqr;
  }

  // Reduce each of the N_OUT accumulators across the block.
  // We sequence one reduction at a time (no parallel groups -- naive).
#pragma unroll
  for (int n = 0; n < N_OUT; ++n) {
    float total =
        mhc_fused_v4_detail::warp_block_reduce_sum<NUM_THREADS>(acc[n],
                                                                s_reduce);
    if (threadIdx.x == 0) {
      gemm_mul[n] = total;
    }
  }
}

} // namespace kernel
