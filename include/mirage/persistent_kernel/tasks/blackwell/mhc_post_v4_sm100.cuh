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

// V4-Flash mhc_post (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/mhc_post_tilelang.md
// Reference: deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:684-687
// (Block.hc_post):
//   y = post.unsqueeze(-1) * x.unsqueeze(-2)
//       + sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
//
// Per token t, per output hc-stream i_hco, per hidden index i1_h:
//   x_out[t, i_hco, i1_h] = c[t, i_hco] * d[t, i1_h]
//                           + sum over i_hci of
//                               a[t, i_hci, i_hco] * b[t, i_hci, i1_h]
//
// Naive design (Blackwell, single CTA per token, no TMA/UMMA/warp-spec):
//   * Grid: (num_tokens, 1, 1). Each CTA processes one token's
//     (HC, HIDDEN) output.
//   * Block: 256 threads (Blackwell default WORKER_NUM_THREADS).
//   * fp32 accumulation, bf16 store.
//   * Strategy: each thread loops over hidden indices i1_h in stride
//     blockDim.x; for each hidden index it computes the HC-wide vector
//     new_r and writes HC bf16 lanes.
//
// Inputs (in TBGraph order, matches register_..._task() codegen):
//   input_ptrs[0] = comb_mix (a)     fp32 [num_tokens, HC, HC]
//   input_ptrs[1] = residual_in (b)  bf16 [num_tokens, HC, HIDDEN]
//   input_ptrs[2] = post_mix (c)     fp32 [num_tokens, HC]
//   input_ptrs[3] = x_in (d)         bf16 [num_tokens, HIDDEN]
//
// Outputs:
//   output_ptrs[0] = residual_out (x) bf16 [num_tokens, HC, HIDDEN]
//
// Pointers are pre-offset by the runtime per-token slice.

namespace kernel {

template <int HC, int HIDDEN, int NUM_THREADS = 256>
__device__ __forceinline__ void mhc_post_v4_sm100_impl(
    void const *comb_mix_ptr,    // fp32 [HC, HC]      (this token's slice)
    void const *residual_in_ptr, // bf16 [HC, HIDDEN]
    void const *post_mix_ptr,    // fp32 [HC]
    void const *x_in_ptr,        // bf16 [HIDDEN]
    void *residual_out_ptr) {    // bf16 [HC, HIDDEN]
  static_assert(HC > 0, "HC must be positive");
  static_assert(HIDDEN > 0, "HIDDEN must be positive");
  static_assert(NUM_THREADS > 0, "NUM_THREADS must be positive");

  using bf16 = type::bfloat16_t;

  float const *__restrict__ a = static_cast<float const *>(comb_mix_ptr);
  bf16 const *__restrict__ b =
      static_cast<bf16 const *>(residual_in_ptr);
  float const *__restrict__ c = static_cast<float const *>(post_mix_ptr);
  bf16 const *__restrict__ d = static_cast<bf16 const *>(x_in_ptr);
  bf16 *__restrict__ y = static_cast<bf16 *>(residual_out_ptr);

  // Stage comb_mix (HC*HC fp32) and post_mix (HC fp32) into shared mem so
  // every thread can read them cheaply. HC=4 in V4-Flash so this is tiny.
  __shared__ float s_a[HC * HC];
  __shared__ float s_c[HC];
  if (threadIdx.x < HC * HC) {
    s_a[threadIdx.x] = a[threadIdx.x];
  }
  if (threadIdx.x < HC) {
    s_c[threadIdx.x] = c[threadIdx.x];
  }
  __syncthreads();

  // Per-hidden index, compute the HC-wide output vector and store as bf16.
  for (int i1_h = threadIdx.x; i1_h < HIDDEN; i1_h += NUM_THREADS) {
    float d_val = static_cast<float>(d[i1_h]);

    // Read residual_in[:, i1_h] (the HC-wide vector at this hidden col).
    float b_local[HC];
#pragma unroll
    for (int i = 0; i < HC; ++i) {
      b_local[i] = static_cast<float>(b[i * HIDDEN + i1_h]);
    }

    // For each output hc-stream i_hco:
    //   new_r[i_hco] = c[i_hco] * d_val + sum_i a[i, i_hco] * b_local[i]
#pragma unroll
    for (int i_hco = 0; i_hco < HC; ++i_hco) {
      float acc = s_c[i_hco] * d_val;
#pragma unroll
      for (int i_hci = 0; i_hci < HC; ++i_hci) {
        acc += s_a[i_hci * HC + i_hco] * b_local[i_hci];
      }
      y[i_hco * HIDDEN + i1_h] = bf16(acc);
    }
  }
}

} // namespace kernel
