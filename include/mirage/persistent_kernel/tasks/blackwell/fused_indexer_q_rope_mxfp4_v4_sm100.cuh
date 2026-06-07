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

// V4-Flash fused_indexer_q_rope_mxfp4 (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/fused_indexer_q_rope_mxfp4.md
//
// MXFP4 (Class-B sibling under `use_fp4_cache == True`) Q-side indexer
// kernel. Performs three fused operations per (token, head):
//   1) GPT-J interleaved RoPE on the trailing rope half of `q` (same
//      mathematics as the FP8 sibling).
//   2) per-32-element-block UE8M0 scales derived from amax(block) / 6.0.
//   3) E2M1x2 byte-packed MXFP4 quantization (2 nibbles/byte) + a folded
//      weight write ``weights_out = weights_in * softmax_scale * head_scale``
//      (note: NO q_scale fold — q_scale is per-block here and stays
//      alongside the values).
//
// COUPLING NOTE: this Q-side variant must be paired with the MXFP4
// K-side variant (`fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn`)
// under the `use_fp4_cache=True` flag. The downstream MQA-logits
// consumers select FP4 dispatch via `q_scale != None` (int32 block
// scales).
//
// Naive design (correctness only, no perf):
//   * One CTA per (token, head). grid = (num_tokens * n_heads, 1, 1).
//   * NUM_THREADS = 256 (Blackwell default).
//   * Plain CUDA loops; per-block (32-elem) amax + scale computed by
//     thread 0; nibble packing done thread-strided. No TMA / UMMA /
//     warp-spec.
//
// Tensor layouts (per-(token, head) row, the catalog preoffsets):
//   q_in:        bf16   [HEAD_DIM = 128]
//   cos_sin:     fp32   [2 * HALF_ROT_DIM = 64]
//   weights_in:  bf16   [1]
//   q_packed:    uint8  [HEAD_DIM / 2 = 64]               (2 E2M1 nibbles/byte)
//   q_scale:     uint8  [HEAD_DIM / MXFP4_BLOCK = 4]      (1 UE8M0 byte / 32-elem block)
//   weights_out: fp32   [1]
//
// HEAD_DIM (128), HALF_ROT_DIM (32), MXFP4_BLOCK (32) are template
// params; NOPE_DIM = HEAD_DIM - 2*HALF_ROT_DIM is derived.

namespace kernel {

namespace fused_indexer_q_rope_mxfp4_v4_detail {

// MXFP4 (E2M1) representable magnitudes:
//   |x| ∈ {0, 0.5, 1, 1.5, 2, 3, 4, 6}
// (sign bit handled separately).
__device__ __forceinline__ uint8_t fp32_to_e2m1(float v) {
  // Quantize a single fp32 value to a 4-bit E2M1 code (sign(1) +
  // exp(2) + mantissa(1)).
  uint8_t sign = v < 0.0f ? 0x8 : 0x0;
  float a = v < 0.0f ? -v : v;
  // round-to-nearest-even via the magnitude table.
  // Boundary points sit at midpoints between adjacent magnitudes.
  uint8_t mag;
  if (a < 0.25f) {
    mag = 0; // 0.0
  } else if (a < 0.75f) {
    mag = 1; // 0.5
  } else if (a < 1.25f) {
    mag = 2; // 1.0
  } else if (a < 1.75f) {
    mag = 3; // 1.5
  } else if (a < 2.5f) {
    mag = 4; // 2.0
  } else if (a < 3.5f) {
    mag = 5; // 3.0
  } else if (a < 5.0f) {
    mag = 6; // 4.0
  } else {
    mag = 7; // 6.0 (saturating)
  }
  return static_cast<uint8_t>(sign | mag);
}

} // namespace fused_indexer_q_rope_mxfp4_v4_detail

template <int HEAD_DIM,
          int HALF_ROT_DIM,
          int MXFP4_BLOCK,
          int NUM_THREADS = 256>
__device__ __forceinline__ void
fused_indexer_q_rope_mxfp4_v4_sm100_impl(
    void const *q_in_ptr,       // bf16 [HEAD_DIM]
    void const *cos_sin_ptr,    // fp32 [2 * HALF_ROT_DIM]
    void const *weights_in_ptr, // bf16 [1]
    void *q_packed_ptr,         // uint8 [HEAD_DIM / 2]
    void *q_scale_ptr,          // uint8 [HEAD_DIM / MXFP4_BLOCK]
    void *weights_out_ptr,      // fp32 [1]
    float softmax_scale,
    float head_scale) {
  static_assert(HEAD_DIM > 0 && (HEAD_DIM % 2) == 0,
                "HEAD_DIM must be positive and even");
  static_assert(HALF_ROT_DIM > 0, "HALF_ROT_DIM must be positive");
  static_assert(2 * HALF_ROT_DIM <= HEAD_DIM,
                "2*HALF_ROT_DIM must not exceed HEAD_DIM");
  static_assert(MXFP4_BLOCK > 0 && (MXFP4_BLOCK % 2) == 0,
                "MXFP4_BLOCK must be positive and even");
  static_assert((HEAD_DIM % MXFP4_BLOCK) == 0,
                "HEAD_DIM must be divisible by MXFP4_BLOCK");
  static_assert(NUM_THREADS > 0, "NUM_THREADS must be positive");

  using bf16 = type::bfloat16_t;

  constexpr int ROT_DIM = 2 * HALF_ROT_DIM;
  constexpr int NOPE_DIM = HEAD_DIM - ROT_DIM;
  constexpr int NUM_BLOCKS = HEAD_DIM / MXFP4_BLOCK;
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;

  bf16 const *__restrict__ q_in = static_cast<bf16 const *>(q_in_ptr);
  float const *__restrict__ cos_sin =
      static_cast<float const *>(cos_sin_ptr);
  bf16 const *__restrict__ w_in =
      static_cast<bf16 const *>(weights_in_ptr);
  uint8_t *__restrict__ q_packed = static_cast<uint8_t *>(q_packed_ptr);
  uint8_t *__restrict__ q_scale_bytes =
      static_cast<uint8_t *>(q_scale_ptr);
  float *__restrict__ w_out = static_cast<float *>(weights_out_ptr);

  float const *__restrict__ cos = cos_sin;
  float const *__restrict__ sin = cos_sin + HALF_ROT_DIM;

  extern __shared__ char smem_raw[];
  float *q_buf = reinterpret_cast<float *>(smem_raw);    // [HEAD_DIM]
  float *block_amax = q_buf + HEAD_DIM;                  // [NUM_BLOCKS]

  // ----- Stage NoPE half (passthrough cast) -----
  for (int i = threadIdx.x; i < NOPE_DIM; i += NUM_THREADS) {
    q_buf[i] = static_cast<float>(q_in[i]);
  }
  // ----- Stage RoPE half (GPT-J interleaved + bf16 roundtrip) -----
  for (int p = threadIdx.x; p < HALF_ROT_DIM; p += NUM_THREADS) {
    float x_even = static_cast<float>(q_in[NOPE_DIM + 2 * p]);
    float x_odd = static_cast<float>(q_in[NOPE_DIM + 2 * p + 1]);
    float c = cos[p];
    float s = sin[p];
    float r_even = x_even * c - x_odd * s;
    float r_odd = x_odd * c + x_even * s;
    r_even = static_cast<float>(static_cast<bf16>(r_even));
    r_odd = static_cast<float>(static_cast<bf16>(r_odd));
    q_buf[NOPE_DIM + 2 * p] = r_even;
    q_buf[NOPE_DIM + 2 * p + 1] = r_odd;
  }
  __syncthreads();

  // ----- Per-block amax (single thread per block; NUM_BLOCKS=4 is tiny) -----
  if (threadIdx.x < NUM_BLOCKS) {
    int b = threadIdx.x;
    float amax = 0.0f;
    int start = b * MXFP4_BLOCK;
#pragma unroll
    for (int i = 0; i < MXFP4_BLOCK; ++i) {
      float v = q_buf[start + i];
      v = v < 0.0f ? -v : v;
      if (v > amax) {
        amax = v;
      }
    }
    // MXFP4 subnormal floor (6.0 * 2^-126).
    float lo_floor = 6.0f * 1.1754943508222875e-38f; // 2^-126
    if (amax < lo_floor) {
      amax = lo_floor;
    }
    float log2_ratio = ceilf(log2f(amax / 6.0f));
    // Clamp into UE8M0 range [-127, 127].
    if (log2_ratio < -127.0f) {
      log2_ratio = -127.0f;
    }
    if (log2_ratio > 127.0f) {
      log2_ratio = 127.0f;
    }
    float scale = exp2f(log2_ratio);
    block_amax[b] = 1.0f / scale; // store inv_scale for the quant pass
    q_scale_bytes[b] =
        static_cast<uint8_t>(static_cast<int>(log2_ratio + 127.0f));
  }
  __syncthreads();

  // ----- Pack pairs of nibbles into bytes -----
  // q_packed has HEAD_DIM/2 bytes; byte index p packs
  //   low  nibble = q_buf[2p]   * inv_scale_of_block[p / (MXFP4_BLOCK/2)]
  //   high nibble = q_buf[2p+1] * inv_scale_of_block[p / (MXFP4_BLOCK/2)]
  constexpr int NUM_PAIR_BYTES = HEAD_DIM / 2;
  constexpr int PAIRS_PER_BLOCK = MXFP4_BLOCK / 2;
  for (int p = threadIdx.x; p < NUM_PAIR_BYTES; p += NUM_THREADS) {
    int b = p / PAIRS_PER_BLOCK;
    float inv_s = block_amax[b];
    float lo = q_buf[2 * p] * inv_s;
    float hi = q_buf[2 * p + 1] * inv_s;
    uint8_t lo_n = fused_indexer_q_rope_mxfp4_v4_detail::fp32_to_e2m1(lo);
    uint8_t hi_n = fused_indexer_q_rope_mxfp4_v4_detail::fp32_to_e2m1(hi);
    q_packed[p] = static_cast<uint8_t>((hi_n << 4) | (lo_n & 0xF));
  }

  // ----- Folded weight (NO q_scale; per-block scales stay alongside) -----
  if (threadIdx.x == 0) {
    float w = static_cast<float>(w_in[0]);
    w_out[0] = w * softmax_scale * head_scale;
  }
  (void)NUM_WARPS; // unused; reserved for future cross-warp reductions
}

} // namespace kernel
