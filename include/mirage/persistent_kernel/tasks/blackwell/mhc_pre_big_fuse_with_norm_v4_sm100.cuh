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
#include "mhc_pre_big_fuse_v4_sm100.cuh" // for sinkhorn_inplace + HCMult3

// V4-Flash mhc_pre_big_fuse_with_norm (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/mhc_pre_big_fuse_with_norm_tilelang.md
//
// Same as mhc_pre_big_fuse but additionally fuses a second-pass RMSNorm
// (with learnable norm_weight) on top of the pre-mix-weighted-sum
// output. The fused RMS norm denominator is recomputed in-kernel
// (it would differ from gemm_out_sqrsum because y = sum_h pre_mix * residual,
// so its squared sum is not the same as residual.flatten().pow(2).sum()).
//
// Per the spec Math:
//   y      = sum_h pre_mix[h] * residual[t, h, :]   # fp32 [H]
//   y_bf16 = bf16(y)                                 # round to bf16
//   sumsq  = sum(y_bf16.float()**2)                  # fp32 scalar
//   rsqrt_norm  = rsqrt(sumsq / H + norm_eps)
//   layer_input = bf16(y_bf16.float() * rsqrt_norm * norm_weight.float())
//
// Note the deliberate bf16 round-trip on y before the second-pass
// squared-sum -- this matches the reference graph's
// `attn_norm(y.to(bf16))` precision.
//
// Naive design (one CTA per token, NUM_THREADS=256, no TMA/UMMA/warp-spec):
//   * Identical pre-amble (steps 1..3) to the no-norm variant.
//   * Step 4 is split into two passes:
//       (4a) compute pre-mix-weighted sum, round to bf16, stash in
//            shared output buffer, accumulate squared sum in registers.
//       (4b) warp-block reduce the squared sum, compute rsqrt_norm,
//            then write layer_input[i] = bf16(out[i].float() * rsqrt_norm *
//                                            norm_weight[i].float()).
//
// Inputs (in TBGraph order, matches register_..._task() codegen):
//   input_ptrs[0] = gemm_out_mul    fp32 [N_SPLITS, num_tokens, HC_MULT3]
//   input_ptrs[1] = gemm_out_sqrsum fp32 [N_SPLITS, num_tokens]
//   input_ptrs[2] = hc_scale        fp32 [3]                 (broadcast)
//   input_ptrs[3] = hc_base         fp32 [HC_MULT3]          (broadcast)
//   input_ptrs[4] = residual        bf16 [num_tokens, HC_MULT, HIDDEN]
//   input_ptrs[5] = norm_weight     bf16 [HIDDEN]            (broadcast)
//
// Outputs:
//   output_ptrs[0] = post_mix    fp32 [num_tokens, HC_MULT]
//   output_ptrs[1] = comb_mix    fp32 [num_tokens, HC_MULT*HC_MULT]
//   output_ptrs[2] = layer_input bf16 [num_tokens, HIDDEN]    (RMSNorm'd)

namespace kernel {

namespace mhc_pre_big_fuse_with_norm_v4_detail {

template <int NUM_THREADS>
__device__ __forceinline__ float warp_block_reduce_sum(float val,
                                                       float *smem) {
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
  // Intra-warp reduction.
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
  // First warp combines the per-warp partial sums.
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

} // namespace mhc_pre_big_fuse_with_norm_v4_detail

// TOTAL_TOKENS is needed because the gemm_out_mul / gemm_out_sqrsum
// parents are [N_SPLITS, T, HC_MULT3] / [N_SPLITS, T] (row-major), so
// the stride between split-rows in the per-token-offset pointer is
// TOTAL_TOKENS * HC_MULT3 / TOTAL_TOKENS respectively.
template <int HIDDEN,
          int HC_MULT,
          int N_SPLITS,
          int TOTAL_TOKENS,
          int SINKHORN_REPEAT = 20,
          int NUM_THREADS = 256>
__device__ __forceinline__ void mhc_pre_big_fuse_with_norm_v4_sm100_impl(
    void const *gemm_out_mul_ptr,    // fp32 [N_SPLITS, T, HC_MULT3] preoffset
    void const *gemm_out_sqrsum_ptr, // fp32 [N_SPLITS, T]            preoffset
    void const *hc_scale_ptr,        // fp32 [3]
    void const *hc_base_ptr,         // fp32 [HC_MULT3]
    void const *residual_ptr,        // bf16 [HC_MULT, HIDDEN]
    void const *norm_weight_ptr,     // bf16 [HIDDEN]
    void *post_mix_ptr,              // fp32 [HC_MULT]
    void *comb_mix_ptr,              // fp32 [HC_MULT*HC_MULT]
    void *layer_input_ptr,           // bf16 [HIDDEN]
    float rms_eps,
    float hc_pre_eps,
    float hc_sinkhorn_eps,
    float hc_post_alpha,
    float norm_eps) {
  static_assert(HIDDEN > 0, "HIDDEN must be positive");
  static_assert(HC_MULT > 0, "HC_MULT must be positive");
  static_assert(N_SPLITS > 0, "N_SPLITS must be positive");
  static_assert(TOTAL_TOKENS > 0, "TOTAL_TOKENS must be positive");
  static_assert(NUM_THREADS > 0, "NUM_THREADS must be positive");
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a warp multiple");

  constexpr int HC_MULT3 =
      mhc_pre_big_fuse_v4_detail::HCMult3<HC_MULT>::value;
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;

  using bf16 = type::bfloat16_t;

  float const *__restrict__ gemm_mul =
      static_cast<float const *>(gemm_out_mul_ptr);
  float const *__restrict__ gemm_sqr =
      static_cast<float const *>(gemm_out_sqrsum_ptr);
  float const *__restrict__ hc_scale =
      static_cast<float const *>(hc_scale_ptr);
  float const *__restrict__ hc_base =
      static_cast<float const *>(hc_base_ptr);
  bf16 const *__restrict__ residual =
      static_cast<bf16 const *>(residual_ptr);
  bf16 const *__restrict__ norm_w =
      static_cast<bf16 const *>(norm_weight_ptr);

  float *__restrict__ post_mix = static_cast<float *>(post_mix_ptr);
  float *__restrict__ comb_mix = static_cast<float *>(comb_mix_ptr);
  bf16 *__restrict__ layer_input = static_cast<bf16 *>(layer_input_ptr);

  // Shared memory layout:
  //   s_rsqrt:    rsqrt for the pre-mix path
  //   s_mixes:    HC_MULT3 reduced+rsqrt'd mixes
  //   s_pre_mix:  HC_MULT pre-mix weights (sigmoid+eps)
  //   s_y_bf16:   HIDDEN bf16 staged y after pre-mix-weighted sum (for
  //               pass-2 RMSNorm). 8 KiB at HIDDEN=4096.
  //   s_reduce:   NUM_WARPS fp32 slots for the cross-warp sumsq combine.
  __shared__ float s_rsqrt;
  __shared__ float s_mixes[HC_MULT3];
  __shared__ float s_pre_mix[HC_MULT];
  __shared__ bf16 s_y_bf16[HIDDEN];
  __shared__ float s_reduce[NUM_WARPS];

  // ---- Step 1: reduce sqrsum partials ------------------------------
  // Stride between split-rows in [N_SPLITS, T] = TOTAL_TOKENS.
  if (threadIdx.x == 0) {
    float sq = 0.0f;
#pragma unroll 1
    for (int s = 0; s < N_SPLITS; ++s) {
      sq += gemm_sqr[s * TOTAL_TOKENS];
    }
    s_rsqrt = rsqrtf(
        sq / static_cast<float>(HC_MULT * HIDDEN) + rms_eps);
  }
  __syncthreads();

  float rsqrt = s_rsqrt;

  // ---- Step 2: reduce gemm_out_mul partials and apply rsqrt --------
  // Stride between split-rows in [N_SPLITS, T, HC_MULT3] =
  // TOTAL_TOKENS * HC_MULT3.
  if (threadIdx.x < HC_MULT3) {
    int col = threadIdx.x;
    float acc = 0.0f;
#pragma unroll 1
    for (int s = 0; s < N_SPLITS; ++s) {
      acc += gemm_mul[s * TOTAL_TOKENS * HC_MULT3 + col];
    }
    s_mixes[col] = acc * rsqrt;
  }
  __syncthreads();

  // ---- Step 3 (thread 0): post_mix, Sinkhorn comb_mix, pre_mix ----
  if (threadIdx.x == 0) {
    float scale0 = hc_scale[0];
    float scale1 = hc_scale[1];
    float scale2 = hc_scale[2];

#pragma unroll
    for (int h = 0; h < HC_MULT; ++h) {
      float logit = s_mixes[HC_MULT + h] * scale1 + hc_base[HC_MULT + h];
      float sig = 1.0f / (1.0f + expf(-logit));
      post_mix[h] = sig * hc_post_alpha;
    }

#pragma unroll
    for (int h = 0; h < HC_MULT; ++h) {
      float logit = s_mixes[h] * scale0 + hc_base[h];
      float sig = 1.0f / (1.0f + expf(-logit));
      s_pre_mix[h] = sig + hc_pre_eps;
    }

    float cm[HC_MULT * HC_MULT];
#pragma unroll
    for (int i = 0; i < HC_MULT * HC_MULT; ++i) {
      cm[i] = s_mixes[2 * HC_MULT + i] * scale2 + hc_base[2 * HC_MULT + i];
    }
    mhc_pre_big_fuse_v4_detail::sinkhorn_inplace<HC_MULT, SINKHORN_REPEAT>(
        cm, hc_sinkhorn_eps);

#pragma unroll
    for (int i = 0; i < HC_MULT * HC_MULT; ++i) {
      comb_mix[i] = cm[i];
    }
  }
  __syncthreads();

  // ---- Step 4a: pre-mix-weighted sum (bf16 round) + sumsq partial ---
  float partial_sumsq = 0.0f;
  for (int i = threadIdx.x; i < HIDDEN; i += NUM_THREADS) {
    float acc = 0.0f;
#pragma unroll
    for (int h = 0; h < HC_MULT; ++h) {
      float r = static_cast<float>(residual[h * HIDDEN + i]);
      acc += s_pre_mix[h] * r;
    }
    bf16 y_bf16 = bf16(acc);
    s_y_bf16[i] = y_bf16;
    // The reference recomputes sumsq from the bf16-rounded y, NOT the
    // fp32 acc -- this is the deliberate precision pin in the spec.
    float y_round = static_cast<float>(y_bf16);
    partial_sumsq += y_round * y_round;
  }
  // Ensure s_y_bf16 is visible before pass 2 starts reading it.
  // (warp_block_reduce_sum issues __syncthreads internally as well.)
  float sumsq_y =
      mhc_pre_big_fuse_with_norm_v4_detail::warp_block_reduce_sum<NUM_THREADS>(
          partial_sumsq, s_reduce);

  float rsqrt_norm =
      rsqrtf(sumsq_y / static_cast<float>(HIDDEN) + norm_eps);

  // ---- Step 4b: apply RMSNorm gamma, bf16 store --------------------
  for (int i = threadIdx.x; i < HIDDEN; i += NUM_THREADS) {
    float y = static_cast<float>(s_y_bf16[i]);
    float w = static_cast<float>(norm_w[i]);
    layer_input[i] = bf16(y * rsqrt_norm * w);
  }
}

} // namespace kernel
