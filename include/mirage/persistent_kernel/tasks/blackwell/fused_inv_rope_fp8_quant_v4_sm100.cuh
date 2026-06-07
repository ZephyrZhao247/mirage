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

#include <cstdint>

// V4-Flash fused_inv_rope_fp8_quant (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/fused_inv_rope_fp8_quant.md
//
// Per (token, head): inverse GPT-J RoPE on last ROPE_DIM lanes, then
// UE8M0 block-FP8 quant in QUANT_GROUP-sized chunks. Single-warp CTA
// (NUM_THREADS=32) matches Triton kernel's num_warps=1.
//
// On sm_100a o_scale is INT32 UE8M0-packed: 4 consecutive UE8M0
// exponent bytes per int32. Downstream deepseek_v4_fp8_einsum reads
// this dtype.

namespace kernel {

namespace fused_inv_rope_fp8_quant_v4_detail {

__device__ __forceinline__ uint8_t fp32_to_fp8_e4m3_satfinite(float x) {
#if defined(__CUDA_ARCH__)
  unsigned short out;
  asm("{cvt.rn.satfinite.e4m3x2.f32 %0, %1, %1;}\n"
      : "=h"(out)
      : "f"(x));
  return static_cast<uint8_t>(out & 0xFFu);
#else
  float v = fminf(fmaxf(x, -448.0f), 448.0f);
  return static_cast<uint8_t>(static_cast<int>(v) & 0xFF);
#endif
}

} // namespace fused_inv_rope_fp8_quant_v4_detail

template <int HEAD_DIM,
          int ROPE_DIM,
          int QUANT_GROUP,
          int N_GROUPS,
          int HEADS_PER_GROUP,
          int NUM_THREADS = 32>
__device__ __forceinline__ void fused_inv_rope_fp8_quant_v4_sm100_impl(
    void const *o_ptr,
    void const *positions_ptr,
    void const *cos_sin_cache_ptr,
    void *o_fp8_ptr,
    void *o_scale_ptr,
    int head_id) {
  static_assert(HEAD_DIM % QUANT_GROUP == 0,
                "HEAD_DIM must be a multiple of QUANT_GROUP");
  constexpr int CHUNKS_PER_HEAD = HEAD_DIM / QUANT_GROUP;
  constexpr int BYTES_PER_GROUP = CHUNKS_PER_HEAD * HEADS_PER_GROUP;
  constexpr int SCALE_INNER = (BYTES_PER_GROUP + 3) / 4;
  constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;

  using bf16 = type::bfloat16_t;

  bf16 const *o_h = static_cast<bf16 const *>(o_ptr) + head_id * HEAD_DIM;
  int64_t pos = *static_cast<int64_t const *>(positions_ptr);
  float const *cos_sin = static_cast<float const *>(cos_sin_cache_ptr);
  uint8_t *o_fp8_token = static_cast<uint8_t *>(o_fp8_ptr);
  int32_t *o_scale_token = static_cast<int32_t *>(o_scale_ptr);

  int g = head_id / HEADS_PER_GROUP;
  int h_in_g = head_id % HEADS_PER_GROUP;

  uint8_t *o_fp8_dst =
      o_fp8_token + (g * HEADS_PER_GROUP + h_in_g) * HEAD_DIM;
  int32_t *o_scale_group = o_scale_token + g * SCALE_INNER;

  float const *cos_table = cos_sin + pos * ROPE_DIM;
  float const *sin_table = cos_table + ROPE_DIM / 2;

  __shared__ float s_x[HEAD_DIM];
#pragma unroll 1
  for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
    s_x[i] = static_cast<float>(o_h[i]);
  }
  __syncthreads();

  // Inverse GPT-J RoPE on trailing ROPE_DIM lanes.
  // even_orig =  even'*c + odd'*s
  // odd_orig  = -even'*s + odd'*c
#pragma unroll 1
  for (int r = threadIdx.x; r < ROPE_DIM; r += NUM_THREADS) {
    int pair = r >> 1;
    bool is_even = ((r & 1) == 0);
    int partner = r ^ 1;
    float c = cos_table[pair];
    float s = sin_table[pair];
    float v = s_x[NOPE_DIM + r];
    float vp = s_x[NOPE_DIM + partner];
    float out;
    if (is_even) {
      out = v * c + vp * s;
    } else {
      out = vp * (-s) + v * c;
    }
    s_x[NOPE_DIM + r] = out;
  }
  __syncthreads();

  __shared__ float s_scale_inv[CHUNKS_PER_HEAD];
  __shared__ int s_exponent[CHUNKS_PER_HEAD];

  for (int qb = 0; qb < CHUNKS_PER_HEAD; ++qb) {
    int base = qb * QUANT_GROUP;
    float local = 0.0f;
#pragma unroll 1
    for (int j = threadIdx.x; j < QUANT_GROUP; j += NUM_THREADS) {
      local = fmaxf(local, fabsf(s_x[base + j]));
    }
#pragma unroll
    for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
      local = fmaxf(local, shfl_xor_sync(local, offset));
    }
    if (threadIdx.x == 0) {
      float absmax = fmaxf(local, 1.0e-10f);
      float ratio = absmax * (1.0f / 448.0f);
      int e = static_cast<int>(ceilf(log2f(ratio)));
      s_scale_inv[qb] = exp2f(static_cast<float>(-e));
      s_exponent[qb] = e;
    }
  }
  __syncthreads();

#pragma unroll 1
  for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
    int qb = i / QUANT_GROUP;
    float v = s_x[i] * s_scale_inv[qb];
    v = fminf(fmaxf(v, -448.0f), 448.0f);
    o_fp8_dst[i] = fused_inv_rope_fp8_quant_v4_detail::
        fp32_to_fp8_e4m3_satfinite(v);
  }

  if (threadIdx.x < CHUNKS_PER_HEAD) {
    int qb = threadIdx.x;
    int e_biased = s_exponent[qb] + 127;
    if (e_biased < 0) e_biased = 0;
    if (e_biased > 255) e_biased = 255;
    int byte_idx = h_in_g * CHUNKS_PER_HEAD + qb;
    int word_idx = byte_idx / 4;
    int byte_off = byte_idx % 4;
    unsigned int shifted =
        (static_cast<unsigned int>(e_biased) & 0xFFu) << (byte_off * 8);
    atomicOr(reinterpret_cast<unsigned int *>(o_scale_group + word_idx),
             shifted);
  }
  __syncthreads();
}

} // namespace kernel
