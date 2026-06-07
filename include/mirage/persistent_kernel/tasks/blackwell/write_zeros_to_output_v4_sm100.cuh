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

// V4-Flash write_zeros_to_output (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/write_zeros_to_output.md
//
// In the vLLM fused MoE kernels this is a device-callable helper invoked
// from inside `fused_moe_kernel` when `expert_ids[pid_m] == -1` (the
// expert is not on the current TP rank).  Here we expose it as a
// standalone task that the MPK scheduler can dispatch to clear a single
// `(pid_m, pid_n)` C-tile in `[T, top_k, N]`.
//
// Naive design:
//   * One CTA per (pid_m, pid_n) tile.  Grid = (num_m_blocks,
//     num_n_blocks, 1).
//   * NUM_THREADS = 256 (Blackwell default).
//   * `pid_m = task_metadata.request_id`, `pid_n = task_metadata.kv_idx`
//     (16-bit fields are sufficient for the typical V4-Flash batch sizes
//     and N tile counts).
//   * For each (m, n) lane in the BLOCK_M x BLOCK_N tile, look up
//     `sorted_token_ids[pid_m * BLOCK_M + m]` and gate the store with
//     `offs_token < num_valid_tokens`.  Out-of-range tile lanes (off the
//     end of N) are skipped as well.
//
// Inputs (TBGraph order):
//   input_ptrs[0] = sorted_token_ids (int32) [EM]
//   input_ptrs[1] = expert_ids (int32) [num_m_blocks]
//   input_ptrs[2] = num_tokens_post_pad (int32) [1]
//
// Outputs:
//   output_ptrs[0] = c_out (bf16) [T, top_k, N]  -- in-place
//
// Constants from codegen:
//   T_DIM, TOP_K, N_DIM, BLOCK_M, BLOCK_N
//
// Behaviour:
//   * If `pid_m * BLOCK_M >= num_tokens_post_pad`: no-op.
//   * If `expert_ids[pid_m] != -1`: no-op (the GEMM kernel handles those).
//   * Else: write zeros to the (BLOCK_M, BLOCK_N) tile gated by
//     token_mask and the N-lane mask.

namespace kernel {

template <int T_DIM,
          int TOP_K,
          int N_DIM,
          int BLOCK_M,
          int BLOCK_N,
          int NUM_THREADS = 256>
__device__ __forceinline__ void write_zeros_to_output_v4_sm100_impl(
    void const *sorted_token_ids_ptr,
    void const *expert_ids_ptr,
    void const *num_tokens_post_pad_ptr,
    void *c_ptr,
    int pid_m,
    int pid_n) {
  using bf16 = type::bfloat16_t;

  int const num_valid_tokens = T_DIM * TOP_K;

  int const *__restrict__ sorted_token_ids =
      static_cast<int const *>(sorted_token_ids_ptr);
  int const *__restrict__ expert_ids =
      static_cast<int const *>(expert_ids_ptr);
  int const *__restrict__ num_tokens_post_pad =
      static_cast<int const *>(num_tokens_post_pad_ptr);
  bf16 *__restrict__ c_out = static_cast<bf16 *>(c_ptr);

  int const num_tokens_post = num_tokens_post_pad[0];
  int const m_start = pid_m * BLOCK_M;
  if (m_start >= num_tokens_post) {
    return;
  }

  int const expert_id = expert_ids[pid_m];
  if (expert_id != -1) {
    return;
  }

  int const n_start = pid_n * BLOCK_N;
  int const n_lim = n_start + BLOCK_N < N_DIM ? n_start + BLOCK_N : N_DIM;
  int const n_width = n_lim - n_start;
  if (n_width <= 0) {
    return;
  }

  int const tile_lanes = BLOCK_M * n_width;
  for (int idx = threadIdx.x; idx < tile_lanes; idx += NUM_THREADS) {
    int const m = idx / n_width;
    int const n = idx % n_width;
    int const offs_token = sorted_token_ids[m_start + m];
    if (offs_token >= num_valid_tokens) {
      continue;
    }
    int64_t const row = static_cast<int64_t>(offs_token);
    int64_t const col = static_cast<int64_t>(n_start + n);
    c_out[row * static_cast<int64_t>(N_DIM) + col] = static_cast<bf16>(0.0f);
  }
}

} // namespace kernel
