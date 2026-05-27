/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// =============================================================================
// mla_v4_prefill_gather_sm100.cuh — DeepSeek V4-Flash MLA prefill KV gather.
//
// v1 (compress_ratio==0, SWA-only): paged-to-contiguous gather. For each KV
// row `kv_pos in [0, T_kv)`, the kernel copies `HEAD_DIM` bf16 elements from
// the paged SWA cache into a contiguous workspace buffer consumed by the
// `mla_v4_prefill_sm100` task:
//
//   page_id = page_table[kv_pos / PAGE_SIZE]
//   slot    = kv_pos % PAGE_SIZE
//   gathered_kv[kv_pos, :] = swa_cache[page_id, slot, :]
//
// The optional compressed-cache + indexer-topk inputs are reserved for v2
// follow-up (see docs/mpk/deepseek_v4/attention.md §4).
//
// Grid model:
//   - One CTA per KV row in v1 (kept simple; sibling V3 gather instead has
//     one CTA per request looping over pages). Each CTA derives its row
//     index from `task_desc->task_metadata.token_offset` (blockIdx-agnostic).
//
// Threading:
//   - NUM_THREADS lanes cooperate to copy a single HEAD_DIM bf16 row using
//     vectorized uint4 loads (8 bf16 per load). Lanes with
//     `threadIdx.x >= NUM_THREADS` exit early; the runtime issues
//     `__syncthreads()` around `_execute_task()` so the inactive lanes do
//     not break intra-task barriers (none are needed here).
// =============================================================================

#pragma once
#include "tasks/common/common_header.cuh"
#include <cstdint>
#include <cuda_bf16.h>

namespace kernel {

template <int HEAD_DIM, int PAGE_SIZE, int NUM_THREADS>
__device__ __forceinline__ void mla_v4_prefill_gather_sm100_task_impl(
    void const *__restrict__ swa_cache_ptr,           // bf16 [num_pages, PAGE_SIZE, HEAD_DIM]
    int const *__restrict__ paged_kv_indices_ptr,     // int32 [num_active_pages]
    int const *__restrict__ paged_kv_indptr_ptr,      // int32 [n_req + 1]
    int const *__restrict__ paged_kv_last_page_len_ptr, // int32 [n_req]
    void *__restrict__ gathered_kv_ptr,               // bf16 [T_kv_max, HEAD_DIM]
    int token_offset,
    int num_tokens_per_task,
    int num_tokens_total) {
  // Gate inactive lanes. The MPK worker launches WORKER_NUM_THREADS threads;
  // we only use the first NUM_THREADS.
  if (threadIdx.x >= NUM_THREADS) {
    return;
  }

  // Compile-time invariants.
  static_assert(HEAD_DIM % 8 == 0,
                "HEAD_DIM must be a multiple of 8 for uint4 (=8 bf16) loads");
  static_assert(PAGE_SIZE > 0, "PAGE_SIZE must be positive");

  using T = __nv_bfloat16;

  T const *swa_cache = reinterpret_cast<T const *>(swa_cache_ptr);
  T *gathered_kv = reinterpret_cast<T *>(gathered_kv_ptr);

  int const tid = threadIdx.x;
  int const kv_pos = token_offset;
  // Defensive bounds: V4-Flash gather is one CTA per KV row; skip OOB CTAs.
  if (kv_pos < 0 || kv_pos >= num_tokens_total) {
    return;
  }

  // For v1 (SWA-only, compress_ratio==0) the gather treats all active pages
  // as one flat ring keyed on absolute kv_pos. Per the spec, the page table
  // is the contiguous `paged_kv_indices_buffer` indexed by absolute slab
  // position. Note: per-request offsets (`paged_kv_indptr_buffer`,
  // `paged_kv_last_page_len_buffer`) are passed through but unused in v1's
  // flat-row scheme — they will be needed in v2 when the gather expands
  // into per-request compressed-cache + indexer-topk slabs. We accept them
  // here so the registered task signature stays stable.
  (void)paged_kv_indptr_ptr;
  (void)paged_kv_last_page_len_ptr;

  int const block_id = kv_pos / PAGE_SIZE;
  int const slot = kv_pos % PAGE_SIZE;
  int const page_id = paged_kv_indices_ptr[block_id];

  T const *src = swa_cache + (page_id * PAGE_SIZE + slot) * HEAD_DIM;
  T *dst = gathered_kv + kv_pos * HEAD_DIM;

  // Vectorized copy: HEAD_DIM / 8 uint4 loads, distributed across NUM_THREADS.
  constexpr int VEC = 8; // 8 bf16 per uint4 transaction
  for (int d = tid * VEC; d < HEAD_DIM; d += NUM_THREADS * VEC) {
    if (d + VEC <= HEAD_DIM) {
      *reinterpret_cast<uint4 *>(dst + d) =
          *reinterpret_cast<uint4 const *>(src + d);
    }
  }
  // Tail (only matters if HEAD_DIM % (NUM_THREADS * VEC) leaves a partial).
  // HEAD_DIM is required to be a multiple of 8 (uint4) above; the loop strides
  // by NUM_THREADS*8 so any remainder < NUM_THREADS*8 is still covered by the
  // staggered tids.
  (void)num_tokens_per_task;
}

} // namespace kernel
