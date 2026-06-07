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

// V4-Flash save_partial_states (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/save_partial_states.md
// Triton ref:
//   vllm/models/deepseek_v4/common/ops/save_partial_states.py:48-102
//
// Per-CTA per-token operation (matches the Triton 1-D grid of shape
// (num_tokens,)):
//
//   slot_id = slot_mapping[t]
//   if slot_id < 0: return
//   block_idx     = slot_id / block_size
//   pos_in_block  = slot_id % block_size
//   pos           = positions[t]
//   ape_row       = pos % COMPRESS_RATIO
//   state_cache[block_idx, pos_in_block, 0          : HEAD_SIZE]   = kv[t]
//   state_cache[block_idx, pos_in_block, HEAD_SIZE  : 2*HEAD_SIZE] =
//       score[t] + ape[ape_row]
//
// Inputs are PARTITIONED on dim 0 (per-token slice) for:
//   kv, score, slot_mapping, positions
// and BROADCAST (full tensor) for:
//   ape, state_cache
//
// The runtime pre-offsets each per-token pointer. `state_cache` is
// addressed by the data-dependent `slot_id`, so we must NOT partition
// it on dim 0; we receive the full base.
//
// Naive design (correctness only):
//   * One CTA per token. grid = (num_tokens, 1, 1), block = (256, 1, 1).
//   * Thread-strided loop over the HEAD_SIZE dim. No smem staging.
//   * All math is fp32 (kv/score already fp32 per the spec; ape is fp32).
//
// dtypes (matches the spec):
//   * kv, score:    fp32 [num_tokens, HEAD_SIZE]
//   * ape:          fp32 [COMPRESS_RATIO, HEAD_SIZE]
//   * positions:    int64 [num_tokens]
//   * slot_mapping: int64 [num_tokens]
//   * state_cache:  fp32 [num_blocks, block_size, 2*HEAD_SIZE]  (in-place)

namespace kernel {

template <int HEAD_SIZE, int COMPRESS_RATIO, int BLOCK_SIZE,
          int NUM_THREADS = 256>
__device__ __forceinline__ void save_partial_states_v4_sm100_impl(
    void const *kv_ptr,           // fp32 [HEAD_SIZE]  (this token's row)
    void const *score_ptr,        // fp32 [HEAD_SIZE]  (this token's row)
    void const *ape_ptr,          // fp32 [COMPRESS_RATIO, HEAD_SIZE]
    void const *positions_ptr,    // int64 [1]         (this token's pos)
    void const *slot_mapping_ptr, // int64 [1]         (this token's slot)
    void *state_cache_ptr         // fp32 [num_blocks, BLOCK_SIZE, 2*HEAD_SIZE]
) {
  static_assert(HEAD_SIZE > 0, "HEAD_SIZE must be positive");
  static_assert(COMPRESS_RATIO > 0, "COMPRESS_RATIO must be positive");
  static_assert(BLOCK_SIZE > 0, "BLOCK_SIZE must be positive");

  float const *__restrict__ kv_in    = static_cast<float const *>(kv_ptr);
  float const *__restrict__ score_in = static_cast<float const *>(score_ptr);
  float const *__restrict__ ape_in   = static_cast<float const *>(ape_ptr);
  float *__restrict__ state_cache    = static_cast<float *>(state_cache_ptr);

  int64_t slot_id = *static_cast<int64_t const *>(slot_mapping_ptr);
  if (slot_id < 0) {
    return;
  }

  int64_t pos = *static_cast<int64_t const *>(positions_ptr);
  int ape_row = static_cast<int>(pos % static_cast<int64_t>(COMPRESS_RATIO));
  if (ape_row < 0) {
    ape_row += COMPRESS_RATIO; // defensive; positions are non-negative
  }

  // Slot layout: state_cache has shape [num_blocks, BLOCK_SIZE, 2*HEAD_SIZE].
  // slot_id is a flat index into [num_blocks * BLOCK_SIZE]; the last dim
  // is the per-slot kv/score pair.
  int64_t block_idx    = slot_id / static_cast<int64_t>(BLOCK_SIZE);
  int64_t pos_in_block = slot_id % static_cast<int64_t>(BLOCK_SIZE);
  int64_t slot_offset =
      (block_idx * static_cast<int64_t>(BLOCK_SIZE) + pos_in_block) *
      static_cast<int64_t>(2 * HEAD_SIZE);

  float *__restrict__ slot_base = state_cache + slot_offset;
  float const *__restrict__ ape_row_ptr = ape_in + ape_row * HEAD_SIZE;

  // Write kv half [0, HEAD_SIZE) (straight pass-through).
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    slot_base[i] = kv_in[i];
  }

  // Write score half [HEAD_SIZE, 2*HEAD_SIZE) (fused APE add).
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    slot_base[HEAD_SIZE + i] = score_in[i] + ape_row_ptr[i];
  }
}

} // namespace kernel
