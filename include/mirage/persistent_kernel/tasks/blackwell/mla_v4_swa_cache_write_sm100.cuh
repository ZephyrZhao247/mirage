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
// mla_v4_swa_cache_write_sm100.cuh — DeepSeek V4-Flash MLA sliding-window
// cache write-back.
//
// For each freshly computed K/V row (post-RMSNorm + per-token rotary +
// FP8 QAT), this task writes the row into the per-batch SWA ring at
// `[batch, pos % WINDOW_SIZE, :]`. v1: B=1, batch_ids defaulted to 0.
//
//     for t in [0, T):
//       pos  = positions[t]
//       slot = pos % WINDOW_SIZE
//       swa_cache[batch_ids[t], slot, :] = kv_in[t, :]
//
// Grid model:
//   - One CTA per token. token_offset = bid.x; num_tokens_per_task = 1.
// Threading:
//   - NUM_THREADS lanes vectorize the bf16 row copy with uint4 (8 bf16
//     per transaction). Lanes with threadIdx.x >= NUM_THREADS exit early.
// blockIdx-agnostic: token index comes from task_metadata.token_offset.
// =============================================================================

#pragma once
#include "tasks/common/common_header.cuh"
#include <cstdint>
#include <cuda_bf16.h>

namespace kernel {

template <int HEAD_DIM, int WINDOW_SIZE, int NUM_THREADS>
__device__ __forceinline__ void mla_v4_swa_cache_write_sm100_task_impl(
    void const *__restrict__ kv_in_ptr,        // bf16 [T, HEAD_DIM]
    int const *__restrict__ positions_ptr,     // int32 [T]
    int const *__restrict__ batch_ids_ptr,     // int32 [T] or nullptr (=> 0)
    void *__restrict__ swa_cache_ptr,          // bf16 [B, WINDOW_SIZE, HEAD_DIM]
    int token_offset,
    int num_tokens_per_task,
    int num_tokens_total) {
  if (threadIdx.x >= NUM_THREADS) {
    return;
  }
  static_assert(HEAD_DIM % 8 == 0,
                "HEAD_DIM must be a multiple of 8 for uint4 (=8 bf16) loads");
  static_assert(WINDOW_SIZE > 0, "WINDOW_SIZE must be positive");

  using T = __nv_bfloat16;
  T const *kv_in = reinterpret_cast<T const *>(kv_in_ptr);
  T *swa_cache = reinterpret_cast<T *>(swa_cache_ptr);

  int const tid = threadIdx.x;

  for (int local = 0; local < num_tokens_per_task; ++local) {
    int const t = token_offset + local;
    if (t < 0 || t >= num_tokens_total) {
      break;
    }
    int const pos = positions_ptr[t];
    int const slot = pos % WINDOW_SIZE;
    int const b = (batch_ids_ptr == nullptr) ? 0 : batch_ids_ptr[t];

    T const *src = kv_in + (long long)t * HEAD_DIM;
    T *dst = swa_cache + ((long long)b * WINDOW_SIZE + slot) * HEAD_DIM;

    // Vectorized copy: HEAD_DIM / 8 uint4 transactions striped across lanes.
    constexpr int VEC = 8;
    for (int d = tid * VEC; d < HEAD_DIM; d += NUM_THREADS * VEC) {
      if (d + VEC <= HEAD_DIM) {
        *reinterpret_cast<uint4 *>(dst + d) =
            *reinterpret_cast<uint4 const *>(src + d);
      }
    }
  }
}

} // namespace kernel
