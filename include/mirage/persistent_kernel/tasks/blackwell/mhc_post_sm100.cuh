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

// mhc_post: K5 HC post-expand step for DeepSeek V4-Flash.
//
// Math (per token n):
//   out[n, hc_o, h] = post[n, hc_o] * x[n, h]
//                     + sum_{hc_i} comb[n, hc_i, hc_o] * residual[n, hc_i, h]
//
// Source: ports `mhc_post_tilelang` in deps/vllm/vllm/model_executor/layers/mhc.py
// (lines 359-408). Generated CUDA reference at
// docs/mpk/deepseek_v4/_generated_cuda/mhc_post_tilelang.cu.
//
// Layout:
//   - comb     : [N, hc, hc]   fp32   (input_ptrs[0]) — TileLang "a"
//   - residual : [N, hc, H]    bf16   (input_ptrs[1]) — TileLang "b"
//   - post     : [N, hc]       fp32   (input_ptrs[2]) — TileLang "c"
//   - x        : [N, H]        bf16   (input_ptrs[3]) — TileLang "d"
//   - out      : [N, hc, H]    bf16   (output_ptrs[0]) — TileLang "x"
//
// The runtime slices the row pointers per CTA via imap so each task sees the
// per-token sub-buffers directly. This kernel is blockIdx-agnostic.

namespace kernel {

template <typename T, int HC, int H, int H_BLK, int NUM_THREADS>
__device__ __forceinline__ void mhc_post_task_impl(void const *comb_ptr,
                                                   void const *residual_ptr,
                                                   void const *post_ptr,
                                                   void const *x_ptr,
                                                   void *out_ptr) {
  static_assert(H % H_BLK == 0, "H must be a multiple of H_BLK");
  constexpr int NUM_H_TILES = H / H_BLK;

  // The runtime worker CTA may have a larger blockDim.x than the NUM_THREADS
  // this task is parametrized for (e.g. WORKER_NUM_THREADS=256 on Blackwell).
  // Gate extra threads so they don't issue OOB loads/stores.
  if (threadIdx.x >= NUM_THREADS) {
    return;
  }

  float const *__restrict__ comb = static_cast<float const *>(comb_ptr);
  T const *__restrict__ residual = static_cast<T const *>(residual_ptr);
  float const *__restrict__ post = static_cast<float const *>(post_ptr);
  T const *__restrict__ x = static_cast<T const *>(x_ptr);
  T *__restrict__ out = static_cast<T *>(out_ptr);

  // Per-token-resident registers (broadcast across the CTA via thread-local
  // copies — `comb` is hc*hc fp32 and `post` is hc fp32, total HC*(HC+1) =
  // 20 floats for hc=4. Cheap enough to materialize per thread.)
  float c_local[HC];   // post[n, :]
  float a_local[HC * HC]; // comb[n, :, :], flattened row-major (hc_in, hc_out)

#pragma unroll
  for (int i = 0; i < HC; ++i) {
    c_local[i] = post[i];
  }
#pragma unroll
  for (int i = 0; i < HC * HC; ++i) {
    a_local[i] = comb[i];
  }

  // Pipeline over the hidden dimension. Each h-tile processes H_BLK columns
  // for all HC output channels. Threads partition the H_BLK dimension.
  // h_local stride: each thread handles ELEMS_PER_THREAD = H_BLK / NUM_THREADS
  // contiguous columns per tile.
  static_assert(H_BLK % NUM_THREADS == 0,
                "H_BLK must be a multiple of NUM_THREADS");
  constexpr int ELEMS_PER_THREAD = H_BLK / NUM_THREADS;

  for (int tile = 0; tile < NUM_H_TILES; ++tile) {
    int h_offset = tile * H_BLK;

    // Load b = residual[n, :, h_offset : h_offset + H_BLK] into per-thread
    // registers (fp32 cast). Layout in registers: b_local[hc_i][e].
    float b_local[HC][ELEMS_PER_THREAD];
#pragma unroll
    for (int hc_i = 0; hc_i < HC; ++hc_i) {
#pragma unroll
      for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        int h = h_offset + e * NUM_THREADS + threadIdx.x;
        b_local[hc_i][e] =
            static_cast<float>(residual[hc_i * H + h]);
      }
    }

    // Load d = x[n, h_offset : h_offset + H_BLK] (fp32 cast).
    float d_local[ELEMS_PER_THREAD];
#pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
      int h = h_offset + e * NUM_THREADS + threadIdx.x;
      d_local[e] = static_cast<float>(x[h]);
    }

    // Compute and store: out[n, hc_o, h] = c[hc_o] * d[h]
    //                                    + sum_hc_i a[hc_i, hc_o] * b[hc_i, h]
#pragma unroll
    for (int hc_o = 0; hc_o < HC; ++hc_o) {
#pragma unroll
      for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        float acc = c_local[hc_o] * d_local[e];
#pragma unroll
        for (int hc_i = 0; hc_i < HC; ++hc_i) {
          acc += a_local[hc_i * HC + hc_o] * b_local[hc_i][e];
        }
        int h = h_offset + e * NUM_THREADS + threadIdx.x;
        out[hc_o * H + h] = static_cast<T>(acc);
      }
    }
  }
}

} // namespace kernel
