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

// V4-Flash hc_head_fuse (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/hc_head_fuse_tilelang.md
// vLLM reference: vllm/model_executor/kernels/mhc/tilelang_kernels.py:718-812
// Model reference: ParallelHead.hc_head in
//   deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:729-736
//
// This is the **terminal mHC kernel** -- collapses
// [T, HC_MULT, HIDDEN] -> [T, HIDDEN] for the bf16 hidden state fed
// into the final RMSNorm + lm_head. Fuses:
//   1. Flatten residual to [T, HC_MULT*HIDDEN] and compute per-token
//      sum-of-squares + the hc_mult dot-products mixes[m] =
//      sum_{c,h} residual[t,c,h] * fn[m, c*HIDDEN + h].
//   2. rsqrt = rsqrt(sqrsum / (HC_MULT*HIDDEN) + rms_eps).
//   3. pre_mix[m] = sigmoid(mixes[m] * rsqrt * hc_scale[0] + hc_base[m])
//                   + hc_eps.
//   4. out[t, h] = sum_c pre_mix[c] * residual[t, c, h], stored bf16.
//
// Naive design (correctness only, no perf):
//   * One CTA per token, grid = (num_tokens, 1, 1).
//   * NUM_THREADS = 256 (Blackwell default WORKER_NUM_THREADS).
//   * Pass 1: thread-strided sum over the full [HC_MULT, HIDDEN] tile.
//     - One pass over c in [0, HC_MULT) (sequential) accumulates the
//       sum-of-squares (cross-c, cross-h) and the HC_MULT projections
//       (mixes[m] = sum_{c,h} x[c,h] * fn[m, c*HIDDEN + h]).
//   * After pass 1: warp + cross-warp reduce the (1 + HC_MULT) fp32
//     accumulators, compute rsqrt + pre_mix in thread 0, stash in smem.
//   * Pass 2: thread-strided over hidden h. For each h compute
//     sum_c pre_mix[c] * residual[t,c,h] in fp32, cast bf16, store.
//   * No TMA / no UMMA / no warp-specialization / no pipelining.
//
// Tensor contract (matches the spec; consumer of mhc_post output):
//   * residual: bf16 [num_tokens, HC_MULT, HIDDEN] (row-major)
//   * fn:       fp32 [HC_MULT, HC_MULT*HIDDEN]    (hc_head_fn weights)
//   * hc_scale: fp32 [1]                          (scalar scale)
//   * hc_base:  fp32 [HC_MULT]                    (per-mix bias)
//   * out:      bf16 [num_tokens, HIDDEN]         (pre-norm collapse)
//
// The TBGraph partitions residual + out on the token dim; fn / hc_scale
// / hc_base are broadcast (no partition). The runtime preoffsets the
// per-token slices for residual + out, so this kernel sees a single
// token's worth of work and indexes locally.

namespace kernel {

namespace hc_head_fuse_v4_detail {

template <int NUM_THREADS>
__device__ __forceinline__ float warp_block_reduce_sum(float val,
                                                       float *smem) {
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
#pragma unroll
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    val += shfl_xor_sync(val, offset);
  }
  int lane = threadIdx.x % NUM_THREADS_PER_WARP;
  int warp = threadIdx.x / NUM_THREADS_PER_WARP;
  if (lane == 0) {
    smem[warp] = val;
  }
  __syncthreads();
  float out = (threadIdx.x < NUM_WARPS) ? smem[threadIdx.x] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
      out += shfl_xor_sync(out, offset);
    }
    if (lane == 0) {
      smem[0] = out;
    }
  }
  __syncthreads();
  return smem[0];
}

__device__ __forceinline__ float sigmoidf(float x) {
  return 1.0f / (1.0f + __expf(-x));
}

} // namespace hc_head_fuse_v4_detail

template <int HC_MULT, int HIDDEN, int NUM_THREADS = 256>
__device__ __forceinline__ void hc_head_fuse_v4_sm100_impl(
    void const *residual_ptr,  // bf16 [HC_MULT, HIDDEN] (this token's tile)
    void const *fn_ptr,        // fp32 [HC_MULT, HC_MULT*HIDDEN]
    void const *hc_scale_ptr,  // fp32 [1]
    void const *hc_base_ptr,   // fp32 [HC_MULT]
    void *out_ptr,             // bf16 [HIDDEN] (this token's row)
    float rms_eps,
    float hc_eps) {
  static_assert(HC_MULT > 0, "HC_MULT must be positive");
  static_assert(HIDDEN > 0, "HIDDEN must be positive");
  static_assert(NUM_THREADS > 0, "NUM_THREADS must be positive");
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a warp multiple");

  using bf16 = type::bfloat16_t;
  namespace dt = hc_head_fuse_v4_detail;

  constexpr int K = HC_MULT * HIDDEN;
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;

  bf16 const *__restrict__ residual =
      static_cast<bf16 const *>(residual_ptr);
  float const *__restrict__ fn = static_cast<float const *>(fn_ptr);
  float const *__restrict__ hc_scale =
      static_cast<float const *>(hc_scale_ptr);
  float const *__restrict__ hc_base =
      static_cast<float const *>(hc_base_ptr);
  bf16 *__restrict__ out = static_cast<bf16 *>(out_ptr);

  // Shared mem layout:
  //   [0 .. NUM_WARPS)         : cross-warp reducer scratch (fp32)
  //   [NUM_WARPS .. NUM_WARPS+HC_MULT) : pre_mix[m] broadcast (fp32)
  extern __shared__ char smem[];
  float *reduce_smem = reinterpret_cast<float *>(smem);
  float *pre_mix_smem = reduce_smem + NUM_WARPS;

  // ---- Pass 1: sqrsum + HC_MULT projections.
  // Stream c=0..HC_MULT-1 sequentially; for each (c, h) thread reads
  // x_val once and contributes to (a) sqrsum, (b) each of the HC_MULT
  // projections (mixes[m] += x_val * fn[m, c*HIDDEN + h]). This is the
  // naive O(HC_MULT^2 * HIDDEN) pattern; performance is not a goal.

  float sqr_partial = 0.0f;
  float mixes_partial[HC_MULT];
#pragma unroll
  for (int m = 0; m < HC_MULT; ++m) {
    mixes_partial[m] = 0.0f;
  }

#pragma unroll 1
  for (int c = 0; c < HC_MULT; ++c) {
    bf16 const *__restrict__ x_chan = residual + c * HIDDEN;
    // Pre-compute the HC_MULT row offsets into fn for this channel.
    // fn is laid out [HC_MULT, HC_MULT * HIDDEN] with the K dim being
    // the (c, h) concatenation: fn[m, c*HIDDEN + h].
    for (int h = threadIdx.x; h < HIDDEN; h += NUM_THREADS) {
      float v = static_cast<float>(x_chan[h]);
      sqr_partial += v * v;
#pragma unroll
      for (int m = 0; m < HC_MULT; ++m) {
        float fn_v = fn[m * K + c * HIDDEN + h];
        mixes_partial[m] += v * fn_v;
      }
    }
  }

  // Reduce sqrsum across the CTA.
  float sqrsum = dt::warp_block_reduce_sum<NUM_THREADS>(sqr_partial, reduce_smem);

  // Reduce each mixes[m] across the CTA. Reuse the same reduce_smem
  // scratch (the helper already syncs internally).
  float mixes[HC_MULT];
#pragma unroll
  for (int m = 0; m < HC_MULT; ++m) {
    mixes[m] = dt::warp_block_reduce_sum<NUM_THREADS>(mixes_partial[m],
                                                      reduce_smem);
  }

  // Compute the per-token rsqrt + sigmoid-gated pre_mix in thread 0 and
  // broadcast via shared memory.
  if (threadIdx.x == 0) {
    float rsqrt = rsqrtf(sqrsum / static_cast<float>(K) + rms_eps);
    float scale = hc_scale[0];
#pragma unroll
    for (int m = 0; m < HC_MULT; ++m) {
      float logit = mixes[m] * rsqrt * scale + hc_base[m];
      pre_mix_smem[m] = dt::sigmoidf(logit) + hc_eps;
    }
  }
  __syncthreads();

  // ---- Pass 2: out[t, h] = sum_c pre_mix[c] * residual[t, c, h].
  // Re-read residual from gmem (the spec calls out two reads of residual
  // as intentional -- staging the full HC_MULT*HIDDEN tile in smem
  // exceeds budget on H=4096 / HC_MULT=4 / 8 KiB+).
  for (int h = threadIdx.x; h < HIDDEN; h += NUM_THREADS) {
    float acc = 0.0f;
#pragma unroll
    for (int c = 0; c < HC_MULT; ++c) {
      float v = static_cast<float>(residual[c * HIDDEN + h]);
      acc += pre_mix_smem[c] * v;
    }
    out[h] = bf16(acc);
  }
}

} // namespace kernel
