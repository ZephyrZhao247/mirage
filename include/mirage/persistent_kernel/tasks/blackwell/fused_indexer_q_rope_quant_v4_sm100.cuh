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

// V4-Flash fused_indexer_q_rope_quant (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/fused_indexer_q_rope_quant.md
//
// FP8 (Class-B sibling under `use_fp4_cache == False`) Q-side indexer
// kernel. Performs three fused operations per (token, head):
//   1) GPT-J interleaved RoPE on the trailing rope half of `q`.
//   2) per-(token,head) UE8M0-discrete scalar `q_scale` derived from
//      max-abs across (nope, r_even, r_odd).
//   3) divide-and-cast quantization to fp8 e4m3 + folded weight write
//      ``weights_out = weights_in * q_scale * softmax_scale * head_scale``.
//
// COUPLING NOTE: this Q-side variant must be paired with the FP8 K-side
// variant (`fused_kv_compress_norm_rope_insert_indexer_attn`) under the
// `use_fp4_cache=False` flag. The downstream `fp8_fp4_mqa_logits` /
// `fp8_fp4_paged_mqa_logits` consumers select FP8 vs MXFP4 via the
// `(q_values, q_scale)` tuple: `q_scale=None` selects FP8. The
// per-(token,head) `q_scale` is folded into `weights_out` rather than
// being emitted as a separate tensor.
//
// Naive design (correctness only, no perf):
//   * One CTA per (token, head). grid = (num_tokens * n_heads, 1, 1).
//   * NUM_THREADS = 256 (Blackwell default WORKER_NUM_THREADS).
//   * Plain CUDA loop over the 128-element head; thread 0 owns the
//     scalar reductions (amax + q_scale + folded weight). No TMA / no
//     UMMA / no warp-specialization.
//
// Tensor layouts (per-(token,head) row, the catalog preoffsets):
//   q_in:        bf16  [HEAD_DIM=128]                — pre-quant Q row
//   cos_sin:     fp32  [2 * HALF_ROT_DIM = 64]      — per-position cos/sin
//                                                     (caller pre-gathers
//                                                     using `positions[t]`)
//   weights_in:  bf16  [1]                          — raw weight scalar
//   q_out:       fp8   [HEAD_DIM=128]               — quantized Q row
//   weights_out: fp32  [1]                          — folded weight scalar
//
// HEAD_DIM (128) and HALF_ROT_DIM (32) are template params; the kernel
// recovers NOPE_DIM = HEAD_DIM - 2*HALF_ROT_DIM at compile time.

namespace kernel {

namespace fused_indexer_q_rope_quant_v4_detail {

// CUDA-correct min/max for int (CUDA's host headers don't expose
// `std::min`/`max` on device).
template <typename T>
__device__ __forceinline__ T dev_max(T a, T b) {
  return a > b ? a : b;
}

} // namespace fused_indexer_q_rope_quant_v4_detail

template <int HEAD_DIM, int HALF_ROT_DIM, int NUM_THREADS = 256>
__device__ __forceinline__ void
fused_indexer_q_rope_quant_v4_sm100_impl(
    void const *q_in_ptr,        // bf16 [HEAD_DIM]
    void const *cos_sin_ptr,     // fp32 [2 * HALF_ROT_DIM]
    void const *weights_in_ptr,  // bf16 [1]
    void *q_out_ptr,             // fp8  [HEAD_DIM]
    void *weights_out_ptr,       // fp32 [1]
    float softmax_scale,
    float head_scale) {
  static_assert(HEAD_DIM > 0, "HEAD_DIM must be positive");
  static_assert(HALF_ROT_DIM > 0, "HALF_ROT_DIM must be positive");
  static_assert(2 * HALF_ROT_DIM <= HEAD_DIM,
                "2*HALF_ROT_DIM must not exceed HEAD_DIM");
  static_assert(NUM_THREADS > 0, "NUM_THREADS must be positive");

  using bf16 = type::bfloat16_t;
  using fp8 = __nv_fp8_e4m3;

  constexpr int ROT_DIM = 2 * HALF_ROT_DIM;
  constexpr int NOPE_DIM = HEAD_DIM - ROT_DIM;

  bf16 const *__restrict__ q_in = static_cast<bf16 const *>(q_in_ptr);
  float const *__restrict__ cos_sin =
      static_cast<float const *>(cos_sin_ptr);
  bf16 const *__restrict__ w_in =
      static_cast<bf16 const *>(weights_in_ptr);
  fp8 *__restrict__ q_out = static_cast<fp8 *>(q_out_ptr);
  float *__restrict__ w_out = static_cast<float *>(weights_out_ptr);

  float const *__restrict__ cos = cos_sin;
  float const *__restrict__ sin = cos_sin + HALF_ROT_DIM;

  // ------------------------------------------------------------------
  // Staging: load + transform per-element values into shared mem.
  // We need fp32 amax + the transformed values; do the rope rotation
  // first so the amax pass sees the post-rope values.
  // ------------------------------------------------------------------
  extern __shared__ char smem_raw[];
  float *q_buf = reinterpret_cast<float *>(smem_raw);    // [HEAD_DIM]
  float *reduce_buf = q_buf + HEAD_DIM;                  // [NUM_WARPS]

  // NoPE half: passthrough fp32 cast.
  for (int i = threadIdx.x; i < NOPE_DIM; i += NUM_THREADS) {
    q_buf[i] = static_cast<float>(q_in[i]);
  }
  // RoPE half: GPT-J interleaved rotation on pairs.
  // For each pair p in [0, HALF_ROT_DIM):
  //   x_even = q_in[NOPE_DIM + 2p], x_odd = q_in[NOPE_DIM + 2p + 1]
  //   r_even = x_even * cos[p] - x_odd * sin[p]
  //   r_odd  = x_odd  * cos[p] + x_even * sin[p]
  // bf16-roundtrip on (r_even, r_odd) for parity with the reference.
  for (int p = threadIdx.x; p < HALF_ROT_DIM; p += NUM_THREADS) {
    float x_even = static_cast<float>(q_in[NOPE_DIM + 2 * p]);
    float x_odd = static_cast<float>(q_in[NOPE_DIM + 2 * p + 1]);
    float c = cos[p];
    float s = sin[p];
    float r_even = x_even * c - x_odd * s;
    float r_odd = x_odd * c + x_even * s;
    // bf16 roundtrip.
    r_even = static_cast<float>(static_cast<bf16>(r_even));
    r_odd = static_cast<float>(static_cast<bf16>(r_odd));
    q_buf[NOPE_DIM + 2 * p] = r_even;
    q_buf[NOPE_DIM + 2 * p + 1] = r_odd;
  }
  __syncthreads();

  // ------------------------------------------------------------------
  // Compute per-thread amax over q_buf, then block-reduce.
  // ------------------------------------------------------------------
  float local_amax = 0.0f;
  for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
    float v = q_buf[i];
    v = v < 0.0f ? -v : v;
    if (v > local_amax) {
      local_amax = v;
    }
  }

  // warp reduce.
#pragma unroll
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    float other = shfl_xor_sync(local_amax, offset);
    if (other > local_amax) {
      local_amax = other;
    }
  }
  int lane = threadIdx.x % NUM_THREADS_PER_WARP;
  int warp = threadIdx.x / NUM_THREADS_PER_WARP;
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
  if (lane == 0) {
    reduce_buf[warp] = local_amax;
  }
  __syncthreads();
  float amax = 0.0f;
  if (threadIdx.x < NUM_WARPS) {
    amax = reduce_buf[threadIdx.x];
  }
  if (warp == 0) {
#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
      float other = shfl_xor_sync(amax, offset);
      if (other > amax) {
        amax = other;
      }
    }
    if (lane == 0) {
      reduce_buf[0] = amax;
    }
  }
  __syncthreads();
  amax = reduce_buf[0];

  // Derive UE8M0-discrete q_scale = 2^ceil(log2(max(amax, 1e-4) / 448)).
  float amax_clamped = amax < 1e-4f ? 1e-4f : amax;
  float log2_ratio = ceilf(log2f(amax_clamped / 448.0f));
  float q_scale = exp2f(log2_ratio);
  float inv_scale = 1.0f / q_scale;

  // ------------------------------------------------------------------
  // Quantize and store fp8.
  // ------------------------------------------------------------------
  for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
    float v = q_buf[i] * inv_scale;
    q_out[i] = static_cast<fp8>(v);
  }

  // ------------------------------------------------------------------
  // Fold weight: weights_out = weights_in * q_scale * softmax_scale *
  // head_scale.
  // ------------------------------------------------------------------
  if (threadIdx.x == 0) {
    float w = static_cast<float>(w_in[0]);
    w_out[0] = w * q_scale * softmax_scale * head_scale;
  }
}

} // namespace kernel
