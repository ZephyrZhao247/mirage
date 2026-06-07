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

// V4-Flash compute_global_topk_indices_and_lens (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/compute_global_topk_indices_and_lens.md
//
// Per-token block-table lookup + valid-count for C4A decode:
//   for each token t:
//     for each topk lane k:
//       local = topk_indices[t, k]
//       if local < 0:           global = -1
//       else:
//         block_id = local // block_size
//         off      = local %  block_size
//         global   = block_table[req, block_id] * block_size + off
//         count++
//     topk_lens[t] = is_valid_token[t] ? count : 0
//
// Naive design: one CTA per token, NUM_THREADS = 256. Threads stripe
// across topk lanes; warp/block reduce for count.

namespace kernel {

namespace compute_global_topk_v4_detail {

template <int NUM_THREADS>
__device__ __forceinline__ int warp_block_reduce_sum_int(int val, int *smem) {
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
  int out = (threadIdx.x < NUM_WARPS) ? smem[threadIdx.x] : 0;
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

} // namespace compute_global_topk_v4_detail

template <int NUM_THREADS = 256>
__device__ __forceinline__ void compute_global_topk_indices_v4_sm100_impl(
    void const *topk_indices_ptr,      // int32 [topk] -- this token's row
    void const *token_to_req_ptr,      // int32 [1]    -- this token's req id
    void const *block_table_ptr,       // int32 [num_reqs, max_blocks_per_seq]
    void const *is_valid_token_ptr,    // uint8/int8 [1] -- this token's valid mask
    void *global_topk_indices_ptr,     // int32 [topk] -- this token's output row
    void *topk_lens_ptr,               // int32 [1]    -- this token's count
    int topk,
    int block_size,
    int max_blocks_per_seq
) {
  static_assert(NUM_THREADS > 0 && NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a positive warp multiple");

  int const *__restrict__ topk_in =
      static_cast<int const *>(topk_indices_ptr);
  int const *__restrict__ token_to_req =
      static_cast<int const *>(token_to_req_ptr);
  int const *__restrict__ block_table =
      static_cast<int const *>(block_table_ptr);
  unsigned char const *__restrict__ is_valid_in =
      static_cast<unsigned char const *>(is_valid_token_ptr);
  int *__restrict__ global_out = static_cast<int *>(global_topk_indices_ptr);
  int *__restrict__ topk_lens_out = static_cast<int *>(topk_lens_ptr);

  __shared__ int reduce_smem[NUM_THREADS / NUM_THREADS_PER_WARP];

  unsigned char is_valid_byte = is_valid_in[0];
  bool is_valid_token = (is_valid_byte != 0);
  int req_idx = token_to_req[0];
  // Each row of block_table is `max_blocks_per_seq` int32 entries.
  int const *bt_row =
      block_table +
      static_cast<long long>(req_idx) * static_cast<long long>(max_blocks_per_seq);

  int local_count = 0;
  for (int k = threadIdx.x; k < topk; k += NUM_THREADS) {
    int local_idx = topk_in[k];
    int slot_id;
    if (local_idx < 0) {
      slot_id = -1;
    } else {
      int block_in_seq = local_idx / block_size;
      int off = local_idx % block_size;
      int phys = bt_row[block_in_seq];
      slot_id = phys * block_size + off;
      local_count += 1;
    }
    global_out[k] = slot_id;
  }

  int total =
      compute_global_topk_v4_detail::warp_block_reduce_sum_int<NUM_THREADS>(
          local_count, reduce_smem);
  if (threadIdx.x == 0) {
    topk_lens_out[0] = is_valid_token ? total : 0;
  }
}

} // namespace kernel
