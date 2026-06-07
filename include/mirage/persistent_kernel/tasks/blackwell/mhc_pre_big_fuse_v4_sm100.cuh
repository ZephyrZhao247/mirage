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

// V4-Flash mhc_pre_big_fuse (NAIVE Blackwell SM100 impl, no fused norm).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/mhc_pre_big_fuse_tilelang.md
// Reference Math (per spec Math section, one token t):
//
//   sqrsum = sum_s gemm_out_sqrsum[s, t]                # fp32 scalar
//   rsqrt  = rsqrtf(sqrsum / (HC_MULT * HIDDEN) + rms_eps)
//   mixes  = sum_s gemm_out_mul[s, t, :] * rsqrt        # fp32 [HC_MULT3]
//
//   pre_logits  = mixes[0     : HC_MULT]
//   post_logits = mixes[HC_MULT : 2*HC_MULT]
//   comb_logits = mixes[2*HC_MULT : HC_MULT3].view(HC_MULT, HC_MULT)
//
//   post_mix[t] = sigmoid(post_logits * hc_scale[1] + hc_base[4:8]) * post_alpha
//   comb_mix[t] = Sinkhorn(comb_logits * hc_scale[2] + hc_base[8:24], R=20)
//   pre_mix     = sigmoid(pre_logits * hc_scale[0] + hc_base[0:4]) + hc_pre_eps
//   layer_input[t] = sum_h pre_mix[h] * residual[t, h, :]  (bf16 out, no RMSNorm)
//
// Naive design (one CTA per token, NUM_THREADS=256, no TMA/UMMA/warp-spec):
//   * Grid: (num_tokens, 1, 1). Block: (NUM_THREADS, 1, 1).
//   * Pointers are pre-offset to this CTA's token.
//   * Step 1: thread 0..N_SPLITS-1 reduce gemm_out_sqrsum partials in
//     shared memory; thread 0 computes rsqrt and stores in s_rsqrt.
//   * Step 2: HC_MULT3 threads (0..23) each reduce a column of
//     gemm_out_mul over the split-k dim, multiply by rsqrt, store to
//     s_mixes[HC_MULT3].
//   * Step 3: thread 0 computes post_mix and the Sinkhorn-normalized
//     comb_mix in registers/shared, writes them to global out.
//   * Step 4: all threads cooperatively compute layer_input[i] across
//     hidden dim via per-row pre_mix-weighted sum, with bf16 store.
//
// Inputs (in TBGraph order, matches register_..._task() codegen):
//   input_ptrs[0] = gemm_out_mul    fp32 [N_SPLITS, num_tokens, HC_MULT3]
//   input_ptrs[1] = gemm_out_sqrsum fp32 [N_SPLITS, num_tokens]
//   input_ptrs[2] = hc_scale        fp32 [3]                 (broadcast)
//   input_ptrs[3] = hc_base         fp32 [HC_MULT3]          (broadcast)
//   input_ptrs[4] = residual        bf16 [num_tokens, HC_MULT, HIDDEN]
//
// Outputs:
//   output_ptrs[0] = post_mix    fp32 [num_tokens, HC_MULT]
//   output_ptrs[1] = comb_mix    fp32 [num_tokens, HC_MULT*HC_MULT]
//   output_ptrs[2] = layer_input bf16 [num_tokens, HIDDEN]
//
// Note: gemm_out_mul / gemm_out_sqrsum partition on dim 1 (the token
//       axis), so the runtime pointer offset already points to this
//       token's [N_SPLITS, HC_MULT3] / [N_SPLITS] slice with the
//       leading-N_SPLITS stride preserved.

namespace kernel {

namespace mhc_pre_big_fuse_v4_detail {

// HC_MULT3 = HC_MULT * (2 + HC_MULT)  (e.g., 24 for HC_MULT=4)
template <int HC_MULT>
struct HCMult3 {
  static constexpr int value = HC_MULT * (2 + HC_MULT);
};

// Sinkhorn priming + iteration, all in registers for HC_MULT=4 (16
// fp32 elements). Performs the same sequence as the spec's PyTorch
// reference (one priming round of row-softmax+eps, col-normalize,
// then SINKHORN_REPEAT-1 (row, col) normalizations).
template <int HC, int SINKHORN_REPEAT>
__device__ __forceinline__ void sinkhorn_inplace(float cm[HC * HC],
                                                  float eps) {
  // ---- priming: row-softmax per row, then + eps, then col-normalize ----
#pragma unroll
  for (int r = 0; r < HC; ++r) {
    float mx = cm[r * HC];
#pragma unroll
    for (int c = 1; c < HC; ++c) {
      float v = cm[r * HC + c];
      if (v > mx) {
        mx = v;
      }
    }
    float sum = 0.0f;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
      float v = expf(cm[r * HC + c] - mx);
      cm[r * HC + c] = v;
      sum += v;
    }
    float inv = 1.0f / sum;
#pragma unroll
    for (int c = 0; c < HC; ++c) {
      cm[r * HC + c] = cm[r * HC + c] * inv + eps;
    }
  }
  // col-normalize
#pragma unroll
  for (int c = 0; c < HC; ++c) {
    float s = 0.0f;
#pragma unroll
    for (int r = 0; r < HC; ++r) {
      s += cm[r * HC + c];
    }
    float inv = 1.0f / (s + eps);
#pragma unroll
    for (int r = 0; r < HC; ++r) {
      cm[r * HC + c] *= inv;
    }
  }
  // ---- (SINKHORN_REPEAT - 1) full iterations ----
#pragma unroll 1
  for (int it = 0; it < SINKHORN_REPEAT - 1; ++it) {
    // row-normalize
#pragma unroll
    for (int r = 0; r < HC; ++r) {
      float s = 0.0f;
#pragma unroll
      for (int c = 0; c < HC; ++c) {
        s += cm[r * HC + c];
      }
      float inv = 1.0f / (s + eps);
#pragma unroll
      for (int c = 0; c < HC; ++c) {
        cm[r * HC + c] *= inv;
      }
    }
    // col-normalize
#pragma unroll
    for (int c = 0; c < HC; ++c) {
      float s = 0.0f;
#pragma unroll
      for (int r = 0; r < HC; ++r) {
        s += cm[r * HC + c];
      }
      float inv = 1.0f / (s + eps);
#pragma unroll
      for (int r = 0; r < HC; ++r) {
        cm[r * HC + c] *= inv;
      }
    }
  }
}

} // namespace mhc_pre_big_fuse_v4_detail

// TOTAL_TOKENS is the [num_tokens] dim of the parent gemm_out_mul /
// gemm_out_sqrsum tensors. We need it (in addition to N_SPLITS) because
// the parent layout is [N_SPLITS, num_tokens, HC_MULT3] (row-major), so
// after the runtime pre-offsets the pointer to this token's slot, the
// stride to the next split's row is TOTAL_TOKENS * HC_MULT3 (gemm_out_mul)
// / TOTAL_TOKENS (gemm_out_sqrsum), NOT HC_MULT3 / 1.
template <int HIDDEN,
          int HC_MULT,
          int N_SPLITS,
          int TOTAL_TOKENS,
          int SINKHORN_REPEAT = 20,
          int NUM_THREADS = 256>
__device__ __forceinline__ void mhc_pre_big_fuse_v4_sm100_impl(
    void const *gemm_out_mul_ptr,    // fp32 [N_SPLITS, T, HC_MULT3] preoffset
    void const *gemm_out_sqrsum_ptr, // fp32 [N_SPLITS, T]            preoffset
    void const *hc_scale_ptr,        // fp32 [3]
    void const *hc_base_ptr,         // fp32 [HC_MULT3]
    void const *residual_ptr,        // bf16 [HC_MULT, HIDDEN]
    void *post_mix_ptr,              // fp32 [HC_MULT]
    void *comb_mix_ptr,              // fp32 [HC_MULT*HC_MULT]
    void *layer_input_ptr,           // bf16 [HIDDEN]
    float rms_eps,
    float hc_pre_eps,
    float hc_sinkhorn_eps,
    float hc_post_alpha) {
  static_assert(HIDDEN > 0, "HIDDEN must be positive");
  static_assert(HC_MULT > 0, "HC_MULT must be positive");
  static_assert(N_SPLITS > 0, "N_SPLITS must be positive");
  static_assert(TOTAL_TOKENS > 0, "TOTAL_TOKENS must be positive");
  static_assert(NUM_THREADS > 0, "NUM_THREADS must be positive");

  constexpr int HC_MULT3 =
      mhc_pre_big_fuse_v4_detail::HCMult3<HC_MULT>::value;

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

  float *__restrict__ post_mix = static_cast<float *>(post_mix_ptr);
  float *__restrict__ comb_mix = static_cast<float *>(comb_mix_ptr);
  bf16 *__restrict__ layer_input = static_cast<bf16 *>(layer_input_ptr);

  // Shared memory: one rsqrt scalar, one mixes vector, one pre_mix vector
  // (so the hidden-pass can read pre_mix without re-reading hc_scale).
  __shared__ float s_rsqrt;
  __shared__ float s_mixes[HC_MULT3];
  __shared__ float s_pre_mix[HC_MULT];

  // ---- Step 1: reduce sqrsum partials ------------------------------
  // Thread 0 does the (small, N_SPLITS-element) reduction. The stride
  // between split-rows is TOTAL_TOKENS (sqrsum is [N_SPLITS, T]).
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
  // HC_MULT3 threads cooperate (one per column of the 24-wide mixes).
  // The stride between split-rows for gemm_out_mul is TOTAL_TOKENS *
  // HC_MULT3 (parent shape [N_SPLITS, T, HC_MULT3]).
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

    // post_mix
#pragma unroll
    for (int h = 0; h < HC_MULT; ++h) {
      float logit = s_mixes[HC_MULT + h] * scale1 + hc_base[HC_MULT + h];
      float sig = 1.0f / (1.0f + expf(-logit));
      post_mix[h] = sig * hc_post_alpha;
    }

    // pre_mix (cache in shared memory for step 4)
#pragma unroll
    for (int h = 0; h < HC_MULT; ++h) {
      float logit = s_mixes[h] * scale0 + hc_base[h];
      float sig = 1.0f / (1.0f + expf(-logit));
      s_pre_mix[h] = sig + hc_pre_eps;
    }

    // comb_mix: build cm[HC_MULT*HC_MULT] register array.
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

  // ---- Step 4: pre-mix-weighted sum across hc, bf16 store ----------
  // Each thread handles a stride-NUM_THREADS slice of HIDDEN.
  for (int i = threadIdx.x; i < HIDDEN; i += NUM_THREADS) {
    float acc = 0.0f;
#pragma unroll
    for (int h = 0; h < HC_MULT; ++h) {
      float r = static_cast<float>(residual[h * HIDDEN + i]);
      acc += s_pre_mix[h] * r;
    }
    layer_input[i] = bf16(acc);
  }
}

} // namespace kernel
