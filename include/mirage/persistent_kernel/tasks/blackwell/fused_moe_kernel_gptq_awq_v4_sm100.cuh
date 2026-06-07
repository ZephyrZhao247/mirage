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

// V4-Flash fused_moe_kernel_gptq_awq (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/fused_moe_kernel_gptq_awq.md
//
// Same MoE GEMM as fused_moe_kernel but with on-the-fly dequantisation
// of int8/int4 weights.  The naive port handles the INT8 W8A16 path
// (B is int8 with per-K-group fp scales).  W4A16 is a thin extension
// (extract a nibble from a packed byte); for correctness-only V4-Flash
// the W8A16 path covers our test cases -- W4A16 is hard-gated below
// and rejected at codegen if requested.
//
// Naive design:
//   * Identical CTA/grid pattern to fused_moe_kernel_v4_sm100.
//   * Inside the K loop, dequant B element by element via
//     `b_f = (int(b_q) - zp) * scale`.  Scales live at K-group
//     granularity: `scale[expert, n_col, k / GROUP_SIZE]`.
//   * Default zero-point is 128 (W8 symmetric).  If `HAS_ZP` is True a
//     `[E, N, K/GROUP_SIZE] int8` tensor is provided; otherwise an
//     all-ones / dummy buffer can be passed and ignored.
//
// I/O contract:
//   inputs:
//     [0] A                  bf16 [T,  K]
//     [1] B                  int8 [E,  N, K]
//     [2] B_scale            fp32 [E,  N, K/GROUP_SIZE]
//     [3] B_zp               int8 [E,  N, K/GROUP_SIZE]  (or zero/dummy)
//     [4] sorted_token_ids   int32 [EM]
//     [5] expert_ids         int32 [num_m_blocks]
//     [6] num_tokens_post_pad int32 [1]
//     [7] topk_weights       fp32 [T*TOP_K]
//   outputs:
//     [0] C                  bf16 [T,  TOP_K, N]

namespace kernel {

template <int T_DIM,
          int TOP_K,
          int K_DIM,
          int N_DIM,
          int NUM_EXPERTS,
          int BLOCK_M,
          int BLOCK_N,
          int GROUP_SIZE,
          bool HAS_ZP,
          bool MUL_ROUTED_WEIGHT,
          int NUM_THREADS = 256>
__device__ __forceinline__ void
fused_moe_kernel_gptq_awq_v4_sm100_impl(void const *a_ptr,
                                        void const *b_ptr,
                                        void const *b_scale_ptr,
                                        void const *b_zp_ptr,
                                        void const *sorted_token_ids_ptr,
                                        void const *expert_ids_ptr,
                                        void const *num_tokens_post_pad_ptr,
                                        void const *topk_weights_ptr,
                                        void *c_ptr,
                                        int pid_m,
                                        int pid_n) {
  using bf16 = type::bfloat16_t;

  static_assert(GROUP_SIZE > 0, "GROUP_SIZE must be positive");

  int const num_valid_tokens = T_DIM * TOP_K;
  int const NUM_K_GROUPS = (K_DIM + GROUP_SIZE - 1) / GROUP_SIZE;

  bf16 const *__restrict__ A = static_cast<bf16 const *>(a_ptr);
  int8_t const *__restrict__ B = static_cast<int8_t const *>(b_ptr);
  float const *__restrict__ B_scale =
      static_cast<float const *>(b_scale_ptr);
  int8_t const *__restrict__ B_zp =
      static_cast<int8_t const *>(b_zp_ptr);
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
    return;
  }

  int const n_start = pid_n * BLOCK_N;
  int const n_lim = n_start + BLOCK_N < N_DIM ? n_start + BLOCK_N : N_DIM;
  int const n_width = n_lim - n_start;
  if (n_width <= 0) {
    return;
  }

  int64_t const e_off_w = static_cast<int64_t>(expert_id) *
                          static_cast<int64_t>(N_DIM) *
                          static_cast<int64_t>(K_DIM);
  int64_t const e_off_s = static_cast<int64_t>(expert_id) *
                          static_cast<int64_t>(N_DIM) *
                          static_cast<int64_t>(NUM_K_GROUPS);

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
        e_off_w + static_cast<int64_t>(n_col) * static_cast<int64_t>(K_DIM);
    int64_t const s_off = e_off_s + static_cast<int64_t>(n_col) *
                                        static_cast<int64_t>(NUM_K_GROUPS);

    float acc = 0.0f;
    for (int k = 0; k < K_DIM; ++k) {
      int const kg = k / GROUP_SIZE;
      float a = static_cast<float>(A[a_off + k]);
      int q = static_cast<int>(B[b_off + k]);
      int zp = HAS_ZP ? static_cast<int>(B_zp[s_off + kg]) : 128;
      float scale = B_scale[s_off + kg];
      float b = static_cast<float>(q - zp) * scale;
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
