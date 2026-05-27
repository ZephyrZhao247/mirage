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
// indexer_score_topk_sm100.cuh — DeepSeek V4-Flash Indexer score + Top-K.
//
// Computes, per token t in [0, T):
//
//   for j in [0, S_max):
//     s_j = 0
//     for h in [0, INDEX_N_HEADS):
//       dot = q[t, h, :] @ kv_cache[t_batch, j, :]
//       s_j += relu(dot) * weights_proj[t, h]
//   # Mask: s > positions[t] // compress_ratio  -> -inf (sentinel -1 in output)
//   topk_indices[t, :] = topk(s, k=TOPK)
//
// V1 simplifications (per the C2 task description):
//   * q, kv_cache are bf16 (FP4 plumbing reserved for v2).
//   * kv_cache is treated as contiguous [S_max, INDEX_HEAD_DIM] (single
//     request) — paged-cache indptr is reserved for v2. The Python catalog
//     module rejects multi-batch inputs.
//   * weights_proj is fp32.
//   * Single fused score+topk per token: a streaming SMEM heap of size TOPK
//     selects the largest values as scores are produced. We use a binary
//     min-heap so the smallest of the top-K so far is at heap[0]; a new
//     score replaces it iff it's larger, then we sift down.
//   * One CTA per token. All NUM_THREADS lanes cooperate on the inner
//     head/head_dim dot products; heap maintenance is single-threaded
//     (thread 0) for simplicity. With TOPK=512 and S_max ~ a few thousand,
//     heap ops are O(log K) per insertion — acceptable for v1.
//
// blockIdx-agnostic: derives ``token_offset`` from ``task_metadata``.
//
// =============================================================================

#pragma once
#include "tasks/common/common_header.cuh"
#include <cuda_bf16.h>

namespace kernel {

namespace indexer_score_topk_detail {

// Block-wide sum reduction across the first NUM_THREADS lanes. Returns the
// same value to every thread (broadcast via smem slot 0).
template <int NUM_THREADS>
__device__ __forceinline__ float
block_reduce_sum(float local_val, float *reduce_smem) {
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a multiple of warp size");
#pragma unroll
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    local_val += shfl_xor_sync(local_val, offset);
  }
  int lane = threadIdx.x % NUM_THREADS_PER_WARP;
  int warp = threadIdx.x / NUM_THREADS_PER_WARP;
  if (lane == 0) {
    reduce_smem[warp] = local_val;
  }
  __syncthreads();
  float v = (threadIdx.x < NUM_WARPS) ? reduce_smem[threadIdx.x] : 0.f;
  if (warp == 0) {
#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
      v += shfl_xor_sync(v, offset);
    }
    if (threadIdx.x == 0) {
      reduce_smem[0] = v;
    }
  }
  __syncthreads();
  return reduce_smem[0];
}

// Min-heap sift-down: heap[root] just got demoted; restore heap order over
// [root, size). ``score`` and ``idx`` are parallel arrays.
__device__ __forceinline__ void
sift_down(float *score, int *idx, int root, int size) {
  while (true) {
    int left = 2 * root + 1;
    int right = 2 * root + 2;
    int smallest = root;
    if (left < size && score[left] < score[smallest]) {
      smallest = left;
    }
    if (right < size && score[right] < score[smallest]) {
      smallest = right;
    }
    if (smallest == root) {
      break;
    }
    float ts = score[root];
    int ti = idx[root];
    score[root] = score[smallest];
    idx[root] = idx[smallest];
    score[smallest] = ts;
    idx[smallest] = ti;
    root = smallest;
  }
}

// Sift-up after inserting at position ``pos``. Used during initial heap fill.
__device__ __forceinline__ void
sift_up(float *score, int *idx, int pos) {
  while (pos > 0) {
    int parent = (pos - 1) / 2;
    if (score[pos] < score[parent]) {
      float ts = score[pos];
      int ti = idx[pos];
      score[pos] = score[parent];
      idx[pos] = idx[parent];
      score[parent] = ts;
      idx[parent] = ti;
      pos = parent;
    } else {
      break;
    }
  }
}

} // namespace indexer_score_topk_detail

// =====================================================================
// indexer_score_topk_task_impl
//
// Inputs (pointers):
//   q_ptr            bf16  [num_tokens_total, INDEX_N_HEADS, INDEX_HEAD_DIM]
//   kv_cache_ptr     bf16  [S_max, INDEX_HEAD_DIM]   (contiguous; v1)
//   weights_ptr      fp32  [num_tokens_total, INDEX_N_HEADS]
//   positions_ptr    int32 [num_tokens_total]
//
// Output:
//   topk_indices_ptr int32 [num_tokens_total, TOPK]
//
// Runtime params:
//   token_offset       — which token this CTA handles.
//   num_tokens_total   — bound on the token axis.
//   s_max              — number of compressed-KV rows in the cache.
//   compress_ratio     — causal mask divisor: positions s > (pos / R) are
//                         masked (sentinel -1 in the output).
// =====================================================================
template <int INDEX_N_HEADS,
          int INDEX_HEAD_DIM,
          int TOPK,
          int NUM_THREADS = 256>
__device__ __forceinline__ void
indexer_score_topk_task_impl(void const *q_ptr,
                             void const *kv_cache_ptr,
                             void const *weights_ptr,
                             void const *positions_ptr,
                             void *topk_indices_ptr,
                             int token_offset,
                             int num_tokens_total,
                             int s_max,
                             int compress_ratio) {
  static_assert(INDEX_N_HEADS > 0, "INDEX_N_HEADS must be positive");
  static_assert(INDEX_HEAD_DIM > 0, "INDEX_HEAD_DIM must be positive");
  static_assert(TOPK > 0, "TOPK must be positive");
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a multiple of warp size");

  if (threadIdx.x >= NUM_THREADS) {
    return;
  }
  if (token_offset >= num_tokens_total) {
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;

  __nv_bfloat16 const *__restrict__ q_base =
      reinterpret_cast<__nv_bfloat16 const *>(q_ptr);
  __nv_bfloat16 const *__restrict__ kv_base =
      reinterpret_cast<__nv_bfloat16 const *>(kv_cache_ptr);
  float const *__restrict__ weights_base =
      reinterpret_cast<float const *>(weights_ptr);
  int const *__restrict__ positions =
      reinterpret_cast<int const *>(positions_ptr);
  int *__restrict__ topk_indices_base =
      reinterpret_cast<int *>(topk_indices_ptr);

  int const t = token_offset;
  int const pos = positions[t];
  // Causal mask: positions s in [0, valid_len) are scored; s >= valid_len
  // produce sentinel -1 in the output (matches the spec's
  // ``s > pos // compress_ratio`` mask — strict-greater means inclusive
  // upper bound ``valid_len = pos / R + 1``).
  int valid_len = (compress_ratio > 0) ? (pos / compress_ratio + 1) : s_max;
  if (valid_len < 0) {
    valid_len = 0;
  }
  if (valid_len > s_max) {
    valid_len = s_max;
  }

  __nv_bfloat16 const *__restrict__ q_t =
      q_base + static_cast<size_t>(t) *
                   static_cast<size_t>(INDEX_N_HEADS) *
                   static_cast<size_t>(INDEX_HEAD_DIM);
  float const *__restrict__ w_t = weights_base +
                                  static_cast<size_t>(t) *
                                      static_cast<size_t>(INDEX_N_HEADS);
  int *__restrict__ out_t =
      topk_indices_base + static_cast<size_t>(t) *
                              static_cast<size_t>(TOPK);

  // Shared workspace.
  // 1. ``reduce_smem`` — warp-level reduction buffer for the dot-product.
  // 2. ``heap_score`` / ``heap_idx`` — min-heap of the current top-TOPK.
  // 3. ``w_smem`` — staged per-head weights so the inner loop avoids
  //    repeated gmem loads of the same w[t, h] across all positions j.
  __shared__ float reduce_smem[NUM_WARPS > 1 ? NUM_WARPS : 1];
  __shared__ float heap_score[TOPK];
  __shared__ int heap_idx[TOPK];
  __shared__ float w_smem[INDEX_N_HEADS];
  __shared__ int heap_size_smem;

  // Stage per-head weights into smem.
  for (int h = threadIdx.x; h < INDEX_N_HEADS; h += NUM_THREADS) {
    w_smem[h] = w_t[h];
  }
  if (threadIdx.x == 0) {
    heap_size_smem = 0;
  }
  __syncthreads();

  // Walk all candidate positions in [0, valid_len). For each j, accumulate
  // the score s_j = sum_h relu(q[t,h,:] . kv[j,:]) * w_smem[h]. Heap is
  // owned by thread 0.
  for (int j = 0; j < valid_len; ++j) {
    __nv_bfloat16 const *__restrict__ kv_j =
        kv_base + static_cast<size_t>(j) *
                      static_cast<size_t>(INDEX_HEAD_DIM);

    float s_j = 0.f;
    for (int h = 0; h < INDEX_N_HEADS; ++h) {
      __nv_bfloat16 const *__restrict__ q_h = q_t + h * INDEX_HEAD_DIM;
      float local = 0.f;
      for (int d = threadIdx.x; d < INDEX_HEAD_DIM; d += NUM_THREADS) {
        float qv = __bfloat162float(q_h[d]);
        float kv = __bfloat162float(kv_j[d]);
        local += qv * kv;
      }
      float dot = indexer_score_topk_detail::block_reduce_sum<NUM_THREADS>(
          local, reduce_smem);
      // relu then per-head weight, per the official's
      // ``relu(score).sum_over_heads * weights.unsqueeze(-1)`` ordering
      // — although the official multiplies weights *after* sum, that is
      // numerically equivalent for non-negative relu output:
      //   sum_h relu(dot_h) * w_h
      //   == (sum_h relu(dot_h)) * w (only when w is shared across h;
      //   here w varies per-h so we fuse inside the head loop).
      // Wait: in the official, ``weights[t, h]`` *does* vary per-h
      // (``weights_proj`` output has shape [B, T, n_heads]). The
      // expression ``index_score.relu_() * weights.unsqueeze(-1)`` then
      // ``.sum(dim=2)`` over the head axis is exactly
      // ``sum_h relu(score[t,h,j]) * w[t,h]`` — which is what we compute.
      float val = fmaxf(dot, 0.f) * w_smem[h];
      s_j += val;
    }

    // Heap update: serial on thread 0 to keep the implementation simple.
    if (threadIdx.x == 0) {
      int size = heap_size_smem;
      if (size < TOPK) {
        heap_score[size] = s_j;
        heap_idx[size] = j;
        indexer_score_topk_detail::sift_up(heap_score, heap_idx, size);
        heap_size_smem = size + 1;
      } else if (s_j > heap_score[0]) {
        heap_score[0] = s_j;
        heap_idx[0] = j;
        indexer_score_topk_detail::sift_down(
            heap_score, heap_idx, 0, TOPK);
      }
    }
    __syncthreads();
  }

  // Emit: indices in heap slots [0, heap_size) are the top-K (unsorted —
  // matches the user's task description that exact order can vary among
  // ties). Slots [heap_size, TOPK) are filled with -1 sentinel.
  int const size = heap_size_smem;
  for (int k = threadIdx.x; k < TOPK; k += NUM_THREADS) {
    out_t[k] = (k < size) ? heap_idx[k] : -1;
  }
}

} // namespace kernel
