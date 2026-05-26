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
#pragma once
#include "tasks/common/common_header.cuh"

// DeepSeek V4-Flash mhc_head — final HC collapse before the LM head.
//
// Math (per token n):
//   x[n, k]_f32   = residual[n, k / H, k % H]_f32          for k in [0, hc*H)
//   sqrsum[n]     = sum_k x[n,k]^2
//   rsqrt[n]      = rsqrt(sqrsum[n] / (hc*H) + rms_eps)
//   mixes[n, m]   = (sum_k x[n,k] * fn[m, k]) * rsqrt[n]   for m in [0, hc)
//   pre[n, m]     = sigmoid(mixes[n,m] * hc_scale[0] + hc_base[m]) + hc_eps
//   out[n, h]_bf  = sum_m pre[n,m] * residual[n, m, h]_f32 -> bf16
//                                                            for h in [0, H)
//
// Source: vLLM's `hc_head_fuse_tilelang` (deps/vllm/vllm/model_executor/layers/mhc.py:460-551),
//         generated CUDA reference at docs/mpk/deepseek_v4/_generated_cuda/hc_head_fuse_tilelang.cu.
//
// blockIdx-agnostic: pointer offsets are baked into input_ptrs/output_ptrs by
// the MPK runtime when the grid is partitioned along dim 0 of `residual` and
// dim 0 of `output`. Each CTA processes BATCH_SIZE tokens starting at its
// (already-offset) pointers.
//
// Grid: (num_tokens, 1, 1); each block handles BATCH_SIZE=1 token.
// Threads/CTA: blockDim.x (typically 128 or 256 worker threads).

namespace kernel {

template <typename T, int BATCH_SIZE, int HC, int H>
__device__ __forceinline__ void mhc_head_task_impl(
    void const *__restrict__ residual_ptr,
    void const *__restrict__ fn_ptr,
    void const *__restrict__ hc_scale_ptr,
    void const *__restrict__ hc_base_ptr,
    void *__restrict__ output_ptr,
    float rms_eps,
    float hc_eps) {
  constexpr int HC_H = HC * H;
  T const *__restrict__ residual = static_cast<T const *>(residual_ptr);
  float const *__restrict__ fn = static_cast<float const *>(fn_ptr);
  float const *__restrict__ hc_scale = static_cast<float const *>(hc_scale_ptr);
  float const *__restrict__ hc_base = static_cast<float const *>(hc_base_ptr);
  T *__restrict__ output = static_cast<T *>(output_ptr);

  int const tid = threadIdx.x;
  int const nthreads = blockDim.x;
  int const lane_id = tid & 31;
  int const warp_id = tid >> 5;
  int const num_warps = (nthreads + 31) >> 5;

  // Shared memory layout:
  //   pre_mix[0..HC-1]                : float, gated sigmoid mix
  //   smem_reduce[0..num_warps-1]     : float, per-warp partial reductions
  //   smem_mixes_warp[0..num_warps*HC-1] : float, per-warp mix partials
  __shared__ float smem_pre_mix[HC];
  __shared__ float smem_reduce[16]; // up to 16 warps (512 threads)
  __shared__ float smem_mixes_warp[16 * HC];

  for (int n = 0; n < BATCH_SIZE; ++n) {
    T const *__restrict__ res_n = residual + n * HC_H;
    T *__restrict__ out_n = output + n * H;

    // ------------------------------------------------------------------
    // Pass 1: compute sqrsum + mixes[0..HC-1].
    // ------------------------------------------------------------------
    float sqrsum_local = 0.0f;
    float mixes_local[HC];
#pragma unroll
    for (int m = 0; m < HC; ++m) {
      mixes_local[m] = 0.0f;
    }

    // Stride over the full hc*H elements.
    for (int k = tid; k < HC_H; k += nthreads) {
      float x = static_cast<float>(res_n[k]);
      sqrsum_local += x * x;
#pragma unroll
      for (int m = 0; m < HC; ++m) {
        // fn is laid out [HC, HC*H]; row m at offset m*HC_H.
        float w = fn[m * HC_H + k];
        mixes_local[m] += x * w;
      }
    }

    // Warp-reduce sqrsum.
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      sqrsum_local += __shfl_xor_sync(0xffffffff, sqrsum_local, offset);
    }
    if (lane_id == 0) {
      smem_reduce[warp_id] = sqrsum_local;
    }

    // Warp-reduce mixes[m] for each m.
#pragma unroll
    for (int m = 0; m < HC; ++m) {
      float v = mixes_local[m];
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, offset);
      }
      if (lane_id == 0) {
        smem_mixes_warp[warp_id * HC + m] = v;
      }
    }
    __syncthreads();

    // Cross-warp finalize using warp 0.
    if (warp_id == 0) {
      float sqr_total = (lane_id < num_warps) ? smem_reduce[lane_id] : 0.0f;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        sqr_total += __shfl_xor_sync(0xffffffff, sqr_total, offset);
      }
      if (lane_id == 0) {
        smem_reduce[0] = sqr_total;
      }

#pragma unroll
      for (int m = 0; m < HC; ++m) {
        float v = (lane_id < num_warps) ? smem_mixes_warp[lane_id * HC + m]
                                        : 0.0f;
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          v += __shfl_xor_sync(0xffffffff, v, offset);
        }
        if (lane_id == 0) {
          // Compute pre_mix immediately.
          float sqr_total_local = smem_reduce[0];
          float rsqrt_v =
              rsqrtf(sqr_total_local / static_cast<float>(HC_H) + rms_eps);
          float scaled = v * rsqrt_v * hc_scale[0] + hc_base[m];
          float sig = 1.0f / (1.0f + __expf(-scaled));
          smem_pre_mix[m] = sig + hc_eps;
        }
      }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Pass 2: out[h] = sum_m pre_mix[m] * residual[n, m, h]
    // ------------------------------------------------------------------
    // Load pre_mix into registers.
    float pre_mix_reg[HC];
#pragma unroll
    for (int m = 0; m < HC; ++m) {
      pre_mix_reg[m] = smem_pre_mix[m];
    }

    for (int h = tid; h < H; h += nthreads) {
      float acc = 0.0f;
#pragma unroll
      for (int m = 0; m < HC; ++m) {
        float r = static_cast<float>(res_n[m * H + h]);
        acc += pre_mix_reg[m] * r;
      }
      out_n[h] = static_cast<T>(acc);
    }
    __syncthreads();
  }
}

} // namespace kernel
