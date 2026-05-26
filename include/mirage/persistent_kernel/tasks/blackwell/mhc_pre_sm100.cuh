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
// mhc_pre_sm100.cuh — DeepSeek V4-Flash mHC "pre" block (port of vLLM's
// `mhc_pre_big_fuse_tilelang` from `deps/vllm/vllm/model_executor/layers/mhc.py`,
// lines 41-178).
//
// Math (per token n):
//   sqrsum_total = sum_s gemm_out_sqrsum[s, n]              (fp32)
//   rms          = rsqrt(sqrsum_total / (HC * H) + rms_eps) (fp32)
//   mixes[j]     = (sum_s gemm_out_mul[s, n, j]) * rms       for j ∈ [0, HC3)
//   pre[j]       = sigmoid(mixes[j]      * hc_scale[0] + hc_base[j])      + hc_pre_eps   for j ∈ [0, HC)
//   post[j]      = sigmoid(mixes[j+HC]   * hc_scale[1] + hc_base[j+HC]) * 2.0            for j ∈ [0, HC)
//   comb_logit[j,k]  = mixes[j*HC+k+2*HC] * hc_scale[2] + hc_base[j*HC+k+2*HC]            for j,k ∈ [0, HC)
//   comb = row_softmax(comb_logit) + eps; comb = comb / (col_sum + eps);
//   for 19 more iters: row-normalize then col-normalize (each with +eps in denominator).
//   layer_input[n, h] = sum_j pre[j] * residual[n, j, h]   (cast bf16)
//
// Outputs:
//   post_mix    [N, HC]      fp32
//   comb_mix    [N, HC*HC]   fp32 (flat layout matches TileLang and Python view)
//   layer_input [N, H]       bf16
//
// MPK convention:
//   - Function is __device__ and called from `_execute_task` dispatcher.
//   - Kernel is blockIdx-agnostic: derives `token_offset` and
//     `num_tokens_per_task` from `task_desc->task_metadata`. We do NOT use
//     `blockIdx.x` for routing. With grid_dim=(num_tokens,) and
//     num_tokens_per_task=1, each CTA processes exactly one token.
//   - Implementation is v1 / naive: serial per-token (no warp specialization,
//     no TMA, no async copy). FP32 accumulators throughout.
//
// Grid design:
//   Natural grid = (num_tokens,) per vLLM. MPK runtime emits N tasks per
//   invocation (one per token) and dispatches each to a free worker. Each
//   CTA reads `n = token_offset` and processes [n, n + num_tokens_per_task).
//   v1 uses num_tokens_per_task=1; the loop is written generically so larger
//   slices work without code changes.
//
// Threads-per-CTA: 128 (4 warps). vLLM uses 96 with warp specialization (warp0
// does Sinkhorn, warps 1-2 do head-reduce). We do them serially with all 128
// threads, which gives us the full warp-group for the head-reduce inner loop
// (the dominant cost at H=4096) and a single warp for the 4x4 Sinkhorn (tiny).
// =============================================================================

#pragma once
#include "tasks/common/common_header.cuh"
#include <math_constants.h>
#include <cuda_bf16.h>

namespace kernel {

template <int HC, int H, int HC3, int N_SPLITS, int SINKHORN_ITERS>
__device__ __forceinline__ void mhc_pre_task_impl(
    void const *gemm_out_mul_ptr,    // fp32 [N_SPLITS, N, HC3]
    void const *gemm_out_sqrsum_ptr, // fp32 [N_SPLITS, N]
    void const *hc_scale_ptr,        // fp32 [3]
    void const *hc_base_ptr,         // fp32 [HC3]
    void const *residual_ptr,        // bf16 [N, HC, H]
    void *post_mix_ptr,              // fp32 [N, HC]
    void *comb_mix_ptr,              // fp32 [N, HC*HC]
    void *layer_input_ptr,           // bf16 [N, H]
    int token_offset,
    int num_tokens_per_task,
    int num_tokens_total,
    float rms_eps,
    float hc_pre_eps,
    float hc_sinkhorn_eps,
    float hc_post_mult_value) {
  // Compile-time invariants.
  static_assert(HC3 == HC * (HC + 2), "HC3 must equal HC * (HC + 2)");
  static_assert(HC == 4, "v1 bakes in HC=4 (matches Flash config)");

  const float *gemm_out_mul = reinterpret_cast<const float *>(gemm_out_mul_ptr);
  const float *gemm_out_sqrsum =
      reinterpret_cast<const float *>(gemm_out_sqrsum_ptr);
  const float *hc_scale = reinterpret_cast<const float *>(hc_scale_ptr);
  const float *hc_base = reinterpret_cast<const float *>(hc_base_ptr);
  const __nv_bfloat16 *residual =
      reinterpret_cast<const __nv_bfloat16 *>(residual_ptr);
  float *post_mix = reinterpret_cast<float *>(post_mix_ptr);
  float *comb_mix = reinterpret_cast<float *>(comb_mix_ptr);
  __nv_bfloat16 *layer_input =
      reinterpret_cast<__nv_bfloat16 *>(layer_input_ptr);

  // Shared workspace:
  //   mixes_smem[HC3]  : per-mix accumulator + rms-scaled output (fp32)
  //   pre_smem[HC]     : pre-mix after sigmoid (fp32)
  //   rms_smem[1]      : broadcast slot for rms (fp32)
  __shared__ float mixes_smem[HC3];
  __shared__ float pre_smem[HC];
  __shared__ float rms_smem;

  const int tid = threadIdx.x;
  const int blk_threads = blockDim.x;
  const float inv_hc_H = 1.0f / static_cast<float>(HC * H);

  for (int slot = 0; slot < num_tokens_per_task; ++slot) {
    int n = token_offset + slot;
    if (n >= num_tokens_total) {
      break;
    }

    // ---------------------------------------------------------------------
    // Phase A: reduce across splits + rsqrt + mix scaling.
    // ---------------------------------------------------------------------
    // Step 1: thread 0 reduces sqrsum across splits and computes rms.
    if (tid == 0) {
      float sqrsum = 0.f;
#pragma unroll
      for (int s = 0; s < N_SPLITS; ++s) {
        sqrsum += gemm_out_sqrsum[static_cast<size_t>(s) * num_tokens_total + n];
      }
      rms_smem = rsqrtf(sqrsum * inv_hc_H + rms_eps);
    }
    __syncthreads();
    float rms = rms_smem;

    // Step 2: each thread handles a stripe of j ∈ [0, HC3) (here HC3=24, blk=128).
    for (int j = tid; j < HC3; j += blk_threads) {
      float mix = 0.f;
#pragma unroll
      for (int s = 0; s < N_SPLITS; ++s) {
        // gemm_out_mul is laid out [splits, N, HC3] row-major.
        size_t idx = (static_cast<size_t>(s) * num_tokens_total + n) *
                         static_cast<size_t>(HC3) +
                     j;
        mix += gemm_out_mul[idx];
      }
      mixes_smem[j] = mix * rms;
    }
    __syncthreads();

    // ---------------------------------------------------------------------
    // Phase B: pre / post / comb-logit (warp 0 does Sinkhorn serially).
    // ---------------------------------------------------------------------
    // Write post_mix[n, j] for j ∈ [0, HC). HC=4 fits in a single warp.
    if (tid < HC) {
      float v = mixes_smem[tid + HC] * hc_scale[1] + hc_base[tid + HC];
      float s = 1.0f / (1.0f + __expf(-v));
      post_mix[static_cast<size_t>(n) * HC + tid] = s * hc_post_mult_value;
    }
    // Compute pre_mix[j] for j ∈ [0, HC), store to smem for Phase C consumers.
    if (tid < HC) {
      float v = mixes_smem[tid] * hc_scale[0] + hc_base[tid];
      pre_smem[tid] = 1.0f / (1.0f + __expf(-v)) + hc_pre_eps;
    }

    // Sinkhorn on a 4x4 matrix done by thread 0 (serial; only 20 iters × 16 ops).
    // Result lands in shared memory (cm_smem) for cooperative store afterwards.
    __shared__ float cm_smem[HC * HC];
    if (tid == 0) {
      float cm[HC * HC];
#pragma unroll
      for (int j = 0; j < HC; ++j) {
#pragma unroll
        for (int k = 0; k < HC; ++k) {
          cm[j * HC + k] =
              mixes_smem[j * HC + k + 2 * HC] * hc_scale[2] +
              hc_base[j * HC + k + 2 * HC];
        }
      }

      // Initial row-softmax + eps.
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        float rmax = -CUDART_INF_F;
#pragma unroll
        for (int k = 0; k < HC; ++k) {
          rmax = fmaxf(rmax, cm[j * HC + k]);
        }
        float rsum = 0.f;
#pragma unroll
        for (int k = 0; k < HC; ++k) {
          float e = __expf(cm[j * HC + k] - rmax);
          cm[j * HC + k] = e;
          rsum += e;
        }
        float inv = 1.0f / rsum;
#pragma unroll
        for (int k = 0; k < HC; ++k) {
          cm[j * HC + k] = cm[j * HC + k] * inv + hc_sinkhorn_eps;
        }
      }

      // Initial col-normalize.
      {
        float csum[HC];
#pragma unroll
        for (int k = 0; k < HC; ++k) {
          float s = 0.f;
#pragma unroll
          for (int j = 0; j < HC; ++j) {
            s += cm[j * HC + k];
          }
          csum[k] = s;
        }
#pragma unroll
        for (int j = 0; j < HC; ++j) {
#pragma unroll
          for (int k = 0; k < HC; ++k) {
            cm[j * HC + k] = cm[j * HC + k] / (csum[k] + hc_sinkhorn_eps);
          }
        }
      }

      // Sinkhorn iterations 2..SINKHORN_ITERS.
#pragma unroll
      for (int iter = 0; iter < SINKHORN_ITERS - 1; ++iter) {
        // Row-normalize.
#pragma unroll
        for (int j = 0; j < HC; ++j) {
          float rsum = 0.f;
#pragma unroll
          for (int k = 0; k < HC; ++k) {
            rsum += cm[j * HC + k];
          }
          float denom = rsum + hc_sinkhorn_eps;
          float inv = 1.0f / denom;
#pragma unroll
          for (int k = 0; k < HC; ++k) {
            cm[j * HC + k] *= inv;
          }
        }
        // Col-normalize.
        float csum[HC];
#pragma unroll
        for (int k = 0; k < HC; ++k) {
          float s = 0.f;
#pragma unroll
          for (int j = 0; j < HC; ++j) {
            s += cm[j * HC + k];
          }
          csum[k] = s;
        }
#pragma unroll
        for (int j = 0; j < HC; ++j) {
#pragma unroll
          for (int k = 0; k < HC; ++k) {
            cm[j * HC + k] = cm[j * HC + k] / (csum[k] + hc_sinkhorn_eps);
          }
        }
      }

      // Write to shared, then to global below (all threads cooperate).
#pragma unroll
      for (int j = 0; j < HC * HC; ++j) {
        cm_smem[j] = cm[j];
      }
    }
    __syncthreads();

    // Cooperatively flush comb_mix[n, :, :] (HC*HC = 16 fp32 writes).
    if (tid < HC * HC) {
      comb_mix[static_cast<size_t>(n) * (HC * HC) + tid] = cm_smem[tid];
    }

    // ---------------------------------------------------------------------
    // Phase C: head reduction layer_input[n, h] = sum_j pre[j] * residual[n,j,h].
    // ---------------------------------------------------------------------
    // All threads stride over h ∈ [0, H). FP32 accumulator, bf16 store.
    // residual layout is row-major [N, HC, H] so residual[n, j, h] is at
    // offset n*HC*H + j*H + h.
    size_t residual_base = static_cast<size_t>(n) * HC * H;
    size_t layer_input_base = static_cast<size_t>(n) * H;
    for (int h = tid; h < H; h += blk_threads) {
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < HC; ++j) {
        float r = __bfloat162float(residual[residual_base + j * H + h]);
        acc += pre_smem[j] * r;
      }
      layer_input[layer_input_base + h] = __float2bfloat16(acc);
    }
    __syncthreads();
  } // for slot
}

} // namespace kernel
