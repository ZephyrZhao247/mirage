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
// mla_v4_q_kv_rmsnorm_sm100.cuh — DeepSeek V4-Flash pre-attention
// joint Q-lora + KV-lora RMSNorm (port of vLLM's
// ``fused_qk_rmsnorm`` Triton kernel,
// ``deps/vllm/vllm/v1/attention/ops/deepseek_v4_ops/fused_qk_rmsnorm.py``).
//
// Math (per token row ``t``, applied independently to two streams
// ``q_lora`` (width ``Q_SIZE``) and ``kv_lora`` (width ``KV_SIZE``)):
//
//   x_f32     = x.float()                                    // bf16 -> f32
//   var       = sum(x_f32 * x_f32) / SIZE
//   rrms      = rsqrt(var + eps)
//   y_f32     = x_f32 * rrms * weight.float()
//   y_bf16    = bf16_cast(y_f32)
//
// Both streams share the per-token CTA: a single ``__syncthreads`` /
// reduction pass per stream, but the two streams reuse the same threads
// and shared workspace serially. We do Q first then KV (Q is the larger
// row); the per-CTA bookkeeping (read offsets via
// ``task_metadata.token_offset``) is shared.
//
// MPK convention:
//   * ``__device__ __forceinline__`` task impl.
//   * blockIdx-agnostic: the kernel derives ``token_offset`` and
//     ``num_tokens_per_task`` from ``task_desc->task_metadata``.
//   * v1: one CTA per token (``num_tokens_per_task == 1``); the loop
//     below is written generically so larger slices Just Work.
//   * NUM_THREADS is a template parameter; threadIdx.x >= NUM_THREADS
//     threads early-out so a worker block with WORKER_NUM_THREADS=256
//     can host a smaller-CTA task without wasted work.
//
// FP32 accumulators for the sqrsum reductions. The kernel currently
// requires both ``Q_SIZE`` and ``KV_SIZE`` to be multiples of
// ``NUM_THREADS`` (true for V4-Flash: 1024 % 128 = 0, 512 % 128 = 0).
// =============================================================================

#pragma once
#include "tasks/common/common_header.cuh"
#include <cuda_bf16.h>

namespace kernel {

namespace mla_v4_q_kv_rmsnorm_detail {

// Reduce ``local_sum`` across the first NUM_THREADS lanes of the CTA via
// warp shuffles + a small smem cross-warp reduction. Returns the same
// value to every thread (broadcast via smem slot 0). Caller is
// responsible for the surrounding __syncthreads().
template <int NUM_THREADS>
__device__ __forceinline__ float
block_reduce_sum(float local_sum, float *reduce_smem) {
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a multiple of warp size");
  // Warp-local reduce.
#pragma unroll
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    local_sum += shfl_xor_sync(local_sum, offset);
  }
  // Warp 0 collects per-warp sums.
  int lane = threadIdx.x % NUM_THREADS_PER_WARP;
  int warp = threadIdx.x / NUM_THREADS_PER_WARP;
  if (lane == 0) {
    reduce_smem[warp] = local_sum;
  }
  __syncthreads();
  // First warp reduces the per-warp partials.
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

// Per-stream RMSNorm on a single row (length SIZE). All NUM_THREADS
// lanes cooperate. ``row_in``, ``row_out`` point at the row's first
// element; ``weight`` is the broadcast per-channel weight.
template <int SIZE, int NUM_THREADS>
__device__ __forceinline__ void rmsnorm_one_row(
    __nv_bfloat16 const *__restrict__ row_in,
    __nv_bfloat16 const *__restrict__ weight,
    __nv_bfloat16 *__restrict__ row_out,
    float eps,
    float *reduce_smem) {
  // Phase 1: sum-of-squares (fp32).
  float local_sum = 0.f;
  for (int i = threadIdx.x; i < SIZE; i += NUM_THREADS) {
    float v = __bfloat162float(row_in[i]);
    local_sum += v * v;
  }
  float ssum = block_reduce_sum<NUM_THREADS>(local_sum, reduce_smem);
  float rrms = rsqrtf(ssum / static_cast<float>(SIZE) + eps);

  // Phase 2: scale + cast.
  for (int i = threadIdx.x; i < SIZE; i += NUM_THREADS) {
    float v = __bfloat162float(row_in[i]);
    float w = __bfloat162float(weight[i]);
    float y = v * rrms * w;
    row_out[i] = __float2bfloat16(y);
  }
}

} // namespace mla_v4_q_kv_rmsnorm_detail

template <int Q_SIZE, int KV_SIZE, int NUM_THREADS = 128>
__device__ __forceinline__ void mla_v4_q_kv_rmsnorm_task_impl(
    void const *q_in_ptr,        // bf16 [num_tokens_total, Q_SIZE]
    void const *kv_in_ptr,       // bf16 [num_tokens_total, KV_SIZE]
    void const *q_weight_ptr,    // bf16 [Q_SIZE]
    void const *kv_weight_ptr,   // bf16 [KV_SIZE]
    void *q_out_ptr,             // bf16 [num_tokens_total, Q_SIZE]
    void *kv_out_ptr,            // bf16 [num_tokens_total, KV_SIZE]
    int token_offset,
    int num_tokens_per_task,
    int num_tokens_total,
    float eps) {
  static_assert(Q_SIZE % NUM_THREADS == 0,
                "Q_SIZE must be a multiple of NUM_THREADS");
  static_assert(KV_SIZE % NUM_THREADS == 0,
                "KV_SIZE must be a multiple of NUM_THREADS");
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a multiple of warp size");

  // Worker thread blocks may have up to WORKER_NUM_THREADS threads. Only
  // the first NUM_THREADS lanes participate; gate the rest so they do
  // nothing (the runtime issues its own __syncthreads() around
  // _execute_task()).
  if (threadIdx.x >= NUM_THREADS) {
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
  __shared__ float reduce_smem[NUM_WARPS > 1 ? NUM_WARPS : 1];

  __nv_bfloat16 const *__restrict__ q_in =
      reinterpret_cast<__nv_bfloat16 const *>(q_in_ptr);
  __nv_bfloat16 const *__restrict__ kv_in =
      reinterpret_cast<__nv_bfloat16 const *>(kv_in_ptr);
  __nv_bfloat16 const *__restrict__ q_weight =
      reinterpret_cast<__nv_bfloat16 const *>(q_weight_ptr);
  __nv_bfloat16 const *__restrict__ kv_weight =
      reinterpret_cast<__nv_bfloat16 const *>(kv_weight_ptr);
  __nv_bfloat16 *__restrict__ q_out =
      reinterpret_cast<__nv_bfloat16 *>(q_out_ptr);
  __nv_bfloat16 *__restrict__ kv_out =
      reinterpret_cast<__nv_bfloat16 *>(kv_out_ptr);

  for (int slot = 0; slot < num_tokens_per_task; ++slot) {
    int t = token_offset + slot;
    if (t >= num_tokens_total) {
      break;
    }
    // Q stream.
    mla_v4_q_kv_rmsnorm_detail::rmsnorm_one_row<Q_SIZE, NUM_THREADS>(
        q_in + static_cast<size_t>(t) * Q_SIZE,
        q_weight,
        q_out + static_cast<size_t>(t) * Q_SIZE,
        eps,
        reduce_smem);
    __syncthreads();
    // KV stream — reuses the same reduce_smem after a __syncthreads().
    mla_v4_q_kv_rmsnorm_detail::rmsnorm_one_row<KV_SIZE, NUM_THREADS>(
        kv_in + static_cast<size_t>(t) * KV_SIZE,
        kv_weight,
        kv_out + static_cast<size_t>(t) * KV_SIZE,
        eps,
        reduce_smem);
    __syncthreads();
  }
}

} // namespace kernel
