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

// V4-Flash combine_topk_swa_indices (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/combine_topk_swa_indices.md
//
// Concatenate per-token compressed-pool topk indices with the
// per-token SWA window indices into one flat per-token index row.
// Output is padded to `combined_topk = align_up(top_k + window_size, 128)`
// with `-1` (the caller pre-fills the buffer with -1; this kernel only
// writes the valid prefix).
//
//   topk_len_t = min((pos + 1) // COMPRESS_RATIO, TOP_K)
//   swa_len_t  = min(pos + 1, WINDOW_SIZE)
//   combined[t, :topk_len_t]                  = topk_indices[t, :topk_len_t] + M * batch
//   combined[t, topk_len_t : topk_len_t+swa_len_t]
//       = M*batch + N + (k + pos - swa_len_t + 1 - gather_start)
//   combined_lens[t] = topk_len_t + swa_len_t
//
// Naive design: one CTA per (batch, token). The caller's TBGraph
// partitions on the *token* dim (so this CTA writes a single
// token's combined row). The token's batch index is read from a
// per-token `token_to_batch` int32 vector (computed in Python from
// `query_start_loc`); this is simpler than recomputing the
// rebased-cumulative range in-kernel.

namespace kernel {

template <int NUM_THREADS = 256>
__device__ __forceinline__ void combine_topk_swa_indices_v4_sm100_impl(
    void const *topk_indices_ptr,    // int32 [top_k] -- this token's row
    void const *token_to_batch_ptr,  // int32 [1]    -- this token's batch idx
    void const *positions_ptr,       // int32 [1]    -- this token's abs seq pos
    void const *gather_start_ptr,    // int32 [1]    -- this token's gather_start
    void *combined_indices_ptr,      // int32 [combined_topk] -- this token's row
    void *combined_lens_ptr,         // int32 [1]    -- this token's len out
    int top_k,
    int compress_ratio,
    int window_size,
    int M,
    int N,
    int combined_topk
) {
  static_assert(NUM_THREADS > 0 && NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a positive warp multiple");

  int const *__restrict__ topk_in =
      static_cast<int const *>(topk_indices_ptr);
  int const *__restrict__ token_to_batch =
      static_cast<int const *>(token_to_batch_ptr);
  int const *__restrict__ positions =
      static_cast<int const *>(positions_ptr);
  int const *__restrict__ gather_start_in =
      static_cast<int const *>(gather_start_ptr);
  int *__restrict__ combined_out = static_cast<int *>(combined_indices_ptr);
  int *__restrict__ combined_lens_out =
      static_cast<int *>(combined_lens_ptr);

  int const batch_idx = token_to_batch[0];
  int const pos = positions[0];
  int const gather_start = gather_start_in[0];

  // Per-token valid lengths.
  int topk_len = 0;
  if (top_k > 0 && compress_ratio > 0) {
    topk_len = (pos + 1) / compress_ratio;
    if (topk_len > top_k) topk_len = top_k;
  }
  int swa_len = pos + 1;
  if (swa_len > window_size) swa_len = window_size;
  if (swa_len < 0) swa_len = 0;

  long long base_shift =
      static_cast<long long>(M) * static_cast<long long>(batch_idx);

  // === Topk portion ===
  for (int k = threadIdx.x; k < topk_len; k += NUM_THREADS) {
    int local = topk_in[k];
    int shifted = (local < 0) ? -1 : static_cast<int>(
                                          static_cast<long long>(local) +
                                          base_shift);
    combined_out[k] = shifted;
  }

  // === SWA portion ===
  // combined[t, topk_len + k] = M*batch + N + (k + pos - swa_len + 1 - gather_start)
  int swa_base = static_cast<int>(base_shift) + N + (pos - swa_len + 1 - gather_start);
  for (int k = threadIdx.x; k < swa_len; k += NUM_THREADS) {
    combined_out[topk_len + k] = swa_base + k;
  }

  // === Length write ===
  if (threadIdx.x == 0) {
    combined_lens_out[0] = topk_len + swa_len;
  }
  __syncthreads();
}

} // namespace kernel
