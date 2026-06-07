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

// V4-Flash moe_align_block_size (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/moe_align_block_size.md
//
// Whole-batch bin-by-expert + pad-to-block_size.  The vLLM reference uses
// CUB BlockScan with 1024 threads across two CTAs; this naive port runs
// EVERYTHING in a single CTA so it has a coherent whole-batch view --
// fine for the V4-Flash batch sizes we exercise in tests.
//
// Algorithm (single CTA, NUM_THREADS = 256):
//   1. Zero the per-expert count array `cumsum[num_experts]` in
//      shared / global memory.
//   2. Histogram pass: thread-strided over `topk_ids[0 .. T*top_k)`.
//      Use `atomicAdd` on a shared-memory `counts[]` array (size
//      NUM_EXPERTS).
//   3. Round each count up to `block_size`.
//   4. Sequential exclusive prefix-sum across NUM_EXPERTS in shared
//      memory (single warp, NUM_EXPERTS is at most 256 for V4-Flash).
//      Last lane writes `num_tokens_post_pad`.
//   5. Sentinel-fill `sorted_token_ids[0 .. EM)` with `T*top_k` (the
//      SENTINEL).  Stride by NUM_THREADS.
//   6. Bucket pass: thread-strided over `topk_ids[0 .. T*top_k)` again.
//      For each flat-index `i`, look up its (rounded) cumsum offset
//      `cur = atomicAdd(&offset[expert], 1)` and write
//      `sorted_token_ids[cur] = i`.
//   7. Fill `expert_ids[0 .. num_m_blocks)`: each thread `e` writes
//      `expert_ids[cumsum[e]/block_size .. cumsum[e+1]/block_size]` with
//      `e`.  Tail blocks (past the last valid) are filled with -1.
//
// I/O contract (matches the spec):
//   inputs:
//     [0] topk_ids                 int32 [T, top_k]
//   outputs:
//     [0] sorted_token_ids         int32 [EM]
//     [1] expert_ids               int32 [num_m_blocks]
//     [2] num_tokens_post_pad      int32 [1]
//
// Template parameters (codegen-baked):
//   T_DIM, TOP_K, NUM_EXPERTS, BLOCK_SIZE, EM, NUM_M_BLOCKS
//
// Grid: (1, 1, 1) -- single CTA only.

namespace kernel {

namespace moe_align_block_size_v4_detail {

// Sequential exclusive prefix scan over `arr[0..N)`.  Single thread
// (threadIdx.x == 0) -- N is at most a few hundred so this is fine.
template <int N>
__device__ __forceinline__ void serial_exclusive_scan(int *arr) {
  int running = 0;
  for (int i = 0; i < N; ++i) {
    int v = arr[i];
    arr[i] = running;
    running += v;
  }
  // Store the total at slot N (caller allocates N+1).
  arr[N] = running;
}

} // namespace moe_align_block_size_v4_detail

template <int T_DIM,
          int TOP_K,
          int NUM_EXPERTS,
          int BLOCK_SIZE,
          int EM,
          int NUM_M_BLOCKS,
          int NUM_THREADS = 256>
__device__ __forceinline__ void moe_align_block_size_v4_sm100_impl(
    void const *topk_ids_ptr,
    void *sorted_token_ids_ptr,
    void *expert_ids_ptr,
    void *num_tokens_post_pad_ptr) {
  static_assert(NUM_EXPERTS > 0 && NUM_EXPERTS <= 512,
                "NUM_EXPERTS out of range for the naive moe_align port");
  static_assert(BLOCK_SIZE > 0, "BLOCK_SIZE must be positive");

  int const NUMEL = T_DIM * TOP_K;
  int const SENTINEL = NUMEL;

  int const *__restrict__ topk_ids =
      static_cast<int const *>(topk_ids_ptr);
  int *__restrict__ sorted_token_ids =
      static_cast<int *>(sorted_token_ids_ptr);
  int *__restrict__ expert_ids =
      static_cast<int *>(expert_ids_ptr);
  int *__restrict__ num_tokens_post_pad =
      static_cast<int *>(num_tokens_post_pad_ptr);

  // Shared-memory layout:
  //   counts[NUM_EXPERTS + 1]  -- per-expert histogram (then padded count,
  //                                then turned into exclusive cumsum;
  //                                the +1 slot holds the total).
  //   offset[NUM_EXPERTS]      -- mutable rank cursor for the bucket pass.
  extern __shared__ char smem_raw[];
  int *counts = reinterpret_cast<int *>(smem_raw);
  int *offset = counts + (NUM_EXPERTS + 1);

  // 1. Zero counts.
  for (int e = threadIdx.x; e < NUM_EXPERTS + 1; e += NUM_THREADS) {
    counts[e] = 0;
  }
  __syncthreads();

  // 2. Histogram pass.
  for (int i = threadIdx.x; i < NUMEL; i += NUM_THREADS) {
    int e = topk_ids[i];
    if (e >= 0 && e < NUM_EXPERTS) {
      atomicAdd(&counts[e], 1);
    }
  }
  __syncthreads();

  // 3. Round counts up to BLOCK_SIZE; thread 0 builds the cumsum.
  if (threadIdx.x == 0) {
    for (int e = 0; e < NUM_EXPERTS; ++e) {
      int c = counts[e];
      int rounded = ((c + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
      counts[e] = rounded;
    }
    moe_align_block_size_v4_detail::serial_exclusive_scan<NUM_EXPERTS>(counts);
    num_tokens_post_pad[0] = counts[NUM_EXPERTS];
    for (int e = 0; e < NUM_EXPERTS; ++e) {
      offset[e] = counts[e];
    }
  }
  __syncthreads();

  // 5. Sentinel-fill sorted_token_ids.
  for (int i = threadIdx.x; i < EM; i += NUM_THREADS) {
    sorted_token_ids[i] = SENTINEL;
  }
  __syncthreads();

  // 6. Bucket pass.
  for (int i = threadIdx.x; i < NUMEL; i += NUM_THREADS) {
    int e = topk_ids[i];
    if (e >= 0 && e < NUM_EXPERTS) {
      int slot = atomicAdd(&offset[e], 1);
      if (slot < EM) {
        sorted_token_ids[slot] = i;
      }
    }
  }
  __syncthreads();

  // 7. Fill expert_ids: for each block index b in [0, NUM_M_BLOCKS), look up
  // which expert it belongs to via the cumsum.  Block b spans
  // [b*BLOCK_SIZE, (b+1)*BLOCK_SIZE) in sorted_token_ids; we find the
  // smallest expert e with cumsum[e+1] > b*BLOCK_SIZE.  Past the last
  // valid block we write -1.
  int const total_pad = num_tokens_post_pad[0];
  int const last_block = total_pad / BLOCK_SIZE;
  for (int b = threadIdx.x; b < NUM_M_BLOCKS; b += NUM_THREADS) {
    if (b >= last_block) {
      expert_ids[b] = -1;
      continue;
    }
    int target = b * BLOCK_SIZE;
    int found = -1;
    for (int e = 0; e < NUM_EXPERTS; ++e) {
      if (counts[e] <= target && target < counts[e + 1]) {
        found = e;
        break;
      }
    }
    expert_ids[b] = found;
  }
}

} // namespace kernel
