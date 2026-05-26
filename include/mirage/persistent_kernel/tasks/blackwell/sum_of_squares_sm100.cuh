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

// Per-row sum-of-squares reducer **and** narrow-output GEMM combined.
//
// This task is the v1 implementation of DeepSeek V4-Flash's mHC pre-norm
// GEMM (`mhc_prenorm_gemm`). See `docs/mpk/deepseek_v4/hc.md` §1.7 ("v1
// fallback — single new .cuh that implements the squared-sum and the matmul
// together with FP32 accumulators in registers, no tcgen05, no warp-spec").
//
// Math (per row n):
//   gemm_out_sqrsum[n]     = sum_{k=0..REDUCTION_SIZE-1} float(input[n,k])^2
//   gemm_out_mul[n, j]_bf16 = sum_{k=0..REDUCTION_SIZE-1}
//                              float(input[n,k]) * float(fn[j,k])
//                              for j in [0, OUTPUT_SIZE)
// where OUTPUT_SIZE = hc3 = (2+hc)*hc; typically 24 at production dims.
//
// The narrow N axis (OUTPUT_SIZE <= 64) is what disqualifies the
// MMA-based `linear_sm100_mpk` path here — its MMA tile is N=16 and the
// tile-and-grid math assumes N is a multiple of 64. This naive kernel
// instead keeps an `OUTPUT_SIZE`-length register array per thread and
// fans out across the K axis.
//
// Grid: (BATCH_SIZE, 1, 1). One CTA per row of the residual.
// Block: (NUM_THREADS, 1, 1) — typically (256, 1, 1) on Blackwell.
//
// blockIdx-agnostic per `/add-mpk-task`: the row index is encoded by the
// runtime via per-task input/output pointer offsets — the kernel reads
// `task_desc->input_ptrs[]` / `output_ptrs[]` directly. The two outputs
// (sqrsum scalar + mul row) and two inputs (residual row + full fn
// matrix) are passed via the standard MPK input/output_ptrs arrays.
//
// I/O contract:
//   input_ptrs[0]  -> bf16 residual_row of shape [REDUCTION_SIZE]
//   input_ptrs[1]  -> bf16 fn          of shape [OUTPUT_SIZE, REDUCTION_SIZE]
//                     (shared across all tokens; row-major)
//   output_ptrs[0] -> bf16 gemm_out_mul_row of shape [OUTPUT_SIZE]
//                     (bf16 in v1 because we reuse the existing dtype path;
//                      v2 will produce fp32 — see hc.md §1.4 v2 note)
//   output_ptrs[1] -> fp32 gemm_out_sqrsum_scalar of shape [1]

namespace kernel {

template <typename T_IN,
          typename T_OUT_MUL,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int NUM_THREADS>
__device__ __forceinline__ void
sum_of_squares_sm100_task_impl(void const *input_ptr,
                               void const *fn_ptr,
                               void *out_mul_ptr,
                               void *out_sqrsum_ptr) {
  T_IN const *__restrict__ x = static_cast<T_IN const *>(input_ptr);
  T_IN const *__restrict__ fn = static_cast<T_IN const *>(fn_ptr);
  T_OUT_MUL *__restrict__ out_mul = static_cast<T_OUT_MUL *>(out_mul_ptr);
  float *__restrict__ out_sqrsum = static_cast<float *>(out_sqrsum_ptr);

  constexpr int WARP_SIZE = 32;
  static_assert(NUM_THREADS % WARP_SIZE == 0,
                "NUM_THREADS must be a multiple of warp size");
  constexpr int NUM_WARPS_LOCAL = NUM_THREADS / WARP_SIZE;

  __shared__ float warp_sums[NUM_WARPS_LOCAL];
  // One [NUM_WARPS_LOCAL, OUTPUT_SIZE] scratch for the per-warp partial-mul.
  __shared__ float warp_mul[NUM_WARPS_LOCAL * OUTPUT_SIZE];

  // Per-thread state.
  float local_sqrsum = 0.0f;
  float local_mul[OUTPUT_SIZE];
#pragma unroll
  for (int j = 0; j < OUTPUT_SIZE; ++j) {
    local_mul[j] = 0.0f;
  }

#pragma unroll 1
  for (int k = threadIdx.x; k < REDUCTION_SIZE; k += NUM_THREADS) {
    float xv = static_cast<float>(x[k]);
    local_sqrsum += xv * xv;
#pragma unroll
    for (int j = 0; j < OUTPUT_SIZE; ++j) {
      float w = static_cast<float>(fn[j * REDUCTION_SIZE + k]);
      local_mul[j] += xv * w;
    }
  }

  // ---- sqrsum reduction (warp-shuffle then cross-warp via smem) ----
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
    local_sqrsum += __shfl_xor_sync(0xffffffff, local_sqrsum, offset);
  }

  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;
  if (lane_id == 0) {
    warp_sums[warp_id] = local_sqrsum;
  }

  // ---- mul reduction: intra-warp warp-shuffle for each j, then write
  //      per-warp partials to smem ----
#pragma unroll
  for (int j = 0; j < OUTPUT_SIZE; ++j) {
    float v = local_mul[j];
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    if (lane_id == 0) {
      warp_mul[warp_id * OUTPUT_SIZE + j] = v;
    }
  }

  __syncthreads();

  // ---- finalize sqrsum in warp 0 ----
  if (warp_id == 0) {
    float v = (lane_id < NUM_WARPS_LOCAL) ? warp_sums[lane_id] : 0.0f;
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
      v += __shfl_xor_sync(0xffffffff, v, offset);
    }
    if (lane_id == 0) {
      out_sqrsum[0] = v;
    }
  }

  // ---- finalize mul: assign each output element to one thread to do the
  //      cross-warp sum and the final bf16 cast. OUTPUT_SIZE is small
  //      (<= 64 at production dims), so we use threadIdx.x for the j-fan-out
  //      via a simple stride loop. ----
  for (int j = threadIdx.x; j < OUTPUT_SIZE; j += NUM_THREADS) {
    float v = 0.0f;
#pragma unroll
    for (int w = 0; w < NUM_WARPS_LOCAL; ++w) {
      v += warp_mul[w * OUTPUT_SIZE + j];
    }
    out_mul[j] = static_cast<T_OUT_MUL>(v);
  }
}

} // namespace kernel
