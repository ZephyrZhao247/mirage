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

// V4-Flash fused_moe_kernel (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/fused_moe_kernel.md
//
// Per-(M-block, N-block) GEMM tile for one of the FusedMoE L1 (gate+up)
// or L2 (down) launches.  This naive port supports BF16 W and BF16 A
// (the simplest correct path -- additional quant strategies in the
// vLLM kernel are FP8/INT8 W8A8 and INT8 W8A16).  Since V4-Flash's
// fast path is the MegaMoE FP4xFP8 fusion and the FusedMoE backend is
// used only as a fallback, the naive bf16 port is sufficient to satisfy
// the I/O contract; higher-precision quant strategies fall back to
// dequantising on the host or to the GPTQ/AWQ-style kernel.
//
// Naive design:
//   * One CTA per (pid_m, pid_n) tile.  Grid = (num_m_blocks,
//     num_n_blocks, 1).  pid_m = task_metadata.request_id, pid_n =
//     task_metadata.kv_idx.
//   * NUM_THREADS = 256.
//   * For each (m, n) lane in the BLOCK_M x BLOCK_N output tile:
//       - Look up offs_token = sorted_token_ids[pid_m*BLOCK_M + m].
//       - Determine a_row = offs_token / TOP_K.
//       - Determine n_col = pid_n * BLOCK_N + n.
//       - For k in [0, K), accumulate A[a_row, k] * B[expert_id, n_col, k]
//         in fp32.
//       - Optional multiply by topk_weights[offs_token] (router weight).
//       - Cast to bf16 and store to C[offs_token, n_col].
//   * If expert_ids[pid_m] == -1 we no-op (zeros are written by the
//     companion write_zeros_to_output kernel).
//
// I/O contract:
//   inputs:
//     [0] A                  bf16 [T,  K]   -- per-token activations
//     [1] B                  bf16 [E,  N, K]
//     [2] sorted_token_ids   int32 [EM]
//     [3] expert_ids         int32 [num_m_blocks]
//     [4] num_tokens_post_pad int32 [1]
//     [5] topk_weights       fp32 [T*TOP_K]  (always present; pass a
//                              dummy buffer of ones when MUL_ROUTED_WEIGHT
//                              is False)
//   outputs:
//     [0] C                  bf16 [T,  TOP_K, N]
//
// Template parameters (codegen-baked):
//   T_DIM, TOP_K, K_DIM, N_DIM, NUM_EXPERTS, BLOCK_M, BLOCK_N,
//   MUL_ROUTED_WEIGHT

namespace kernel {

template <int T_DIM,
          int TOP_K,
          int K_DIM,
          int N_DIM,
          int NUM_EXPERTS,
          int BLOCK_M,
          int BLOCK_N,
          bool MUL_ROUTED_WEIGHT,
          int NUM_THREADS = 256>
__device__ __forceinline__ void fused_moe_kernel_v4_sm100_impl(
    void const *a_ptr,
    void const *b_ptr,
    void const *sorted_token_ids_ptr,
    void const *expert_ids_ptr,
    void const *num_tokens_post_pad_ptr,
    void const *topk_weights_ptr,
    void *c_ptr,
    int pid_m,
    int pid_n) {
  using bf16 = type::bfloat16_t;

  int const num_valid_tokens = T_DIM * TOP_K;

  bf16 const *__restrict__ A = static_cast<bf16 const *>(a_ptr);
  bf16 const *__restrict__ B = static_cast<bf16 const *>(b_ptr);
  int const *__restrict__ sorted_token_ids =
      static_cast<int const *>(sorted_token_ids_ptr);
  int const *__restrict__ expert_ids =
      static_cast<int const *>(expert_ids_ptr);
  int const *__restrict__ num_tokens_post_pad =
      static_cast<int const *>(num_tokens_post_pad_ptr);
  float const *__restrict__ topk_weights =
      static_cast<float const *>(topk_weights_ptr);
  bf16 *__restrict__ C = static_cast<bf16 *>(c_ptr);

  int const num_tokens_post = num_tokens_post_pad[0];
  int const m_start = pid_m * BLOCK_M;
  if (m_start >= num_tokens_post) {
    return;
  }

  int const expert_id = expert_ids[pid_m];
  if (expert_id < 0 || expert_id >= NUM_EXPERTS) {
    return; // write_zeros_to_output handles the -1 branch.
  }

  int const n_start = pid_n * BLOCK_N;
  int const n_lim = n_start + BLOCK_N < N_DIM ? n_start + BLOCK_N : N_DIM;
  int const n_width = n_lim - n_start;
  if (n_width <= 0) {
    return;
  }

  int64_t const e_off =
      static_cast<int64_t>(expert_id) * static_cast<int64_t>(N_DIM) *
      static_cast<int64_t>(K_DIM);

  // Iterate over (m, n) lanes in the tile, one per thread (stride
  // NUM_THREADS).  Within each lane, do the K-axis dot product
  // sequentially in fp32.  This is O(BLOCK_M * BLOCK_N * K_DIM /
  // NUM_THREADS) work per CTA -- naive but correct.
  int const tile_lanes = BLOCK_M * n_width;
  for (int idx = threadIdx.x; idx < tile_lanes; idx += NUM_THREADS) {
    int const m = idx / n_width;
    int const n = idx % n_width;
    int const offs_token = sorted_token_ids[m_start + m];
    if (offs_token >= num_valid_tokens) {
      continue;
    }
    int const n_col = n_start + n;
    int const a_row = offs_token / TOP_K;

    int64_t const a_off =
        static_cast<int64_t>(a_row) * static_cast<int64_t>(K_DIM);
    int64_t const b_off =
        e_off + static_cast<int64_t>(n_col) * static_cast<int64_t>(K_DIM);

    float acc = 0.0f;
    for (int k = 0; k < K_DIM; ++k) {
      float a = static_cast<float>(A[a_off + k]);
      float b = static_cast<float>(B[b_off + k]);
      acc += a * b;
    }

    if (MUL_ROUTED_WEIGHT) {
      acc *= topk_weights[offs_token];
    }

    int64_t const c_off =
        static_cast<int64_t>(offs_token) * static_cast<int64_t>(N_DIM) +
        static_cast<int64_t>(n_col);
    C[c_off] = static_cast<bf16>(acc);
  }
}

} // namespace kernel
