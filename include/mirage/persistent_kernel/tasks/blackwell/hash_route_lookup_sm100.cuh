/* Copyright 2025 CMU
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
// hash_route_lookup_sm100.cuh — DeepSeek V4-Flash hash-based MoE expert
// routing for the early ``num_hash_layers`` layers (3 in V4-Flash).
//
// For ``layer_idx < num_hash_layers`` the expert indices for each token are
// looked up directly from a precomputed table shipped in the checkpoint:
//
//     expert_ids[t, k]   = tid2eid[input_ids[t] * K + k]      // k = 0..K-1
//     topk_weights[t, k] = 1.0f / K                           // uniform
//
// One CTA per token; ``token_offset`` comes from
// ``task_desc->task_metadata.token_offset`` (blockIdx-agnostic).
// =============================================================================

#pragma once
#include "tasks/common/common_header.cuh"

namespace kernel {

template <int K_TOPK>
__device__ __forceinline__ void hash_route_lookup_task_impl(
    void const *input_ids_ptr,   // int32 [num_tokens_total]
    void const *tid2eid_ptr,     // int32 [vocab_size, K_TOPK] row-major
    void *expert_ids_ptr,        // int32 [num_tokens_total, K_TOPK]
    void *topk_weights_ptr,      // fp32  [num_tokens_total, K_TOPK]
    int token_offset,
    int num_tokens_per_task,
    int num_tokens_total) {
  // NUM_THREADS for the kernel — only K_TOPK lanes do real work, but the
  // worker block may launch with up to WORKER_NUM_THREADS threads. Gate so
  // extra threads do no work (the runtime issues its own __syncthreads()
  // around _execute_task()).
  constexpr int NUM_THREADS = 32;
  if (threadIdx.x >= NUM_THREADS) {
    return;
  }

  int const *__restrict__ input_ids = static_cast<int const *>(input_ids_ptr);
  int const *__restrict__ tid2eid = static_cast<int const *>(tid2eid_ptr);
  int *__restrict__ expert_ids = static_cast<int *>(expert_ids_ptr);
  float *__restrict__ topk_weights = static_cast<float *>(topk_weights_ptr);

  constexpr float kUniformWeight = 1.0f / static_cast<float>(K_TOPK);

  // Each CTA handles a contiguous slice of ``num_tokens_per_task`` tokens
  // starting at ``token_offset``. v1: num_tokens_per_task = 1 in practice
  // (one CTA per token); the loop is written generically so larger slices
  // work without code changes.
  for (int local = 0; local < num_tokens_per_task; ++local) {
    int t = token_offset + local;
    if (t >= num_tokens_total) {
      break;
    }
    int tok = input_ids[t];
    // Per-token: K_TOPK threads each handle one (token, k) pair.
    if (threadIdx.x < K_TOPK) {
      int k = threadIdx.x;
      int e = tid2eid[tok * K_TOPK + k];
      expert_ids[t * K_TOPK + k] = e;
      topk_weights[t * K_TOPK + k] = kUniformWeight;
    }
  }
}

} // namespace kernel
