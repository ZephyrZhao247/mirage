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

// V4-Flash FlashMLA sparse decode (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/flash_mla_with_kvcache.md
//
// Naive port: per (token, head), walk SWA indices, gather + dequant
// from paged uint8 cache (FP8 NoPE + UE8M0 + bf16 RoPE), do QK +
// online softmax + PV.

namespace kernel {

namespace flash_mla_with_kvcache_v4_detail {

__device__ __forceinline__ float fp8_e4m3_to_fp32(uint8_t fp8) {
#if defined(__CUDA_ARCH__)
  float out;
  unsigned short packed = static_cast<unsigned short>(fp8);
  asm("{cvt.rn.f32.e4m3 %0, %1;}\n" : "=f"(out) : "h"(packed));
  return out;
#else
  return static_cast<float>(fp8);
#endif
}

__device__ __forceinline__ float ue8m0_byte_to_scale(uint8_t b) {
  int e = static_cast<int>(b) - 127;
  return exp2f(static_cast<float>(e));
}

template <int HEAD_DIM, int HEAD_V, int ROPE_DIM, int QUANT_BLOCK,
          int NUM_THREADS>
__device__ __forceinline__ void
gather_one_k_row(uint8_t const *k_cache,
                 int block_stride_bytes,
                 int cache_block_size,
                 int slot_id,
                 float *s_k) {
  constexpr int NOPE_DIM = HEAD_V;
  constexpr int NUM_QUANT_BLOCKS = NOPE_DIM / QUANT_BLOCK;
  constexpr int FP8_NOPE_BYTES = NOPE_DIM;
  constexpr int BF16_ROPE_BYTES = ROPE_DIM * 2;
  constexpr int TOKEN_BLOCK_BYTES = FP8_NOPE_BYTES + BF16_ROPE_BYTES;
  constexpr int SCALE_BYTES_PER_TOKEN = NUM_QUANT_BLOCKS;

  int block_idx = slot_id / cache_block_size;
  int pos_in_block = slot_id % cache_block_size;
  uint8_t const *block_base =
      k_cache + static_cast<int64_t>(block_idx) *
                    static_cast<int64_t>(block_stride_bytes);
  uint8_t const *token_fp8_ptr =
      block_base + pos_in_block * TOKEN_BLOCK_BYTES;
  uint8_t const *token_rope_ptr = token_fp8_ptr + FP8_NOPE_BYTES;
  uint8_t const *token_scale_ptr =
      block_base + cache_block_size * TOKEN_BLOCK_BYTES +
      pos_in_block * SCALE_BYTES_PER_TOKEN;

  __shared__ float s_scales[NUM_QUANT_BLOCKS];
  if (threadIdx.x < NUM_QUANT_BLOCKS) {
    s_scales[threadIdx.x] =
        ue8m0_byte_to_scale(token_scale_ptr[threadIdx.x]);
  }
  __syncthreads();

#pragma unroll 1
  for (int i = threadIdx.x; i < NOPE_DIM; i += NUM_THREADS) {
    uint8_t fp8 = token_fp8_ptr[i];
    float v = fp8_e4m3_to_fp32(fp8);
    int qb = i / QUANT_BLOCK;
    s_k[i] = v * s_scales[qb];
  }
#pragma unroll 1
  for (int r = threadIdx.x; r < ROPE_DIM; r += NUM_THREADS) {
    type::bfloat16_t v =
        reinterpret_cast<type::bfloat16_t const *>(token_rope_ptr)[r];
    s_k[NOPE_DIM + r] = static_cast<float>(v);
  }
  __syncthreads();
}

} // namespace flash_mla_with_kvcache_v4_detail

template <int HEAD_DIM,
          int HEAD_V,
          int ROPE_DIM,
          int QUANT_BLOCK,
          int NUM_HEADS_Q,
          int MAX_TOPK,
          int NUM_THREADS = 256>
__device__ __forceinline__ void flash_mla_with_kvcache_v4_sm100_impl(
    void const *q_ptr,
    void const *k_cache_ptr,
    void const *indices_ptr,
    void const *topk_length_ptr,
    void const *attn_sink_ptr,
    void *out_ptr,
    int block_stride_bytes,
    int cache_block_size,
    int head_id,
    float softmax_scale) {
  static_assert(HEAD_DIM > HEAD_V, "HEAD_DIM must include rope tail");
  static_assert(HEAD_DIM - HEAD_V == ROPE_DIM, "HEAD_DIM = HEAD_V + ROPE_DIM");

  using bf16 = type::bfloat16_t;

  bf16 const *q_all = static_cast<bf16 const *>(q_ptr);
  uint8_t const *k_cache = static_cast<uint8_t const *>(k_cache_ptr);
  int32_t const *indices = static_cast<int32_t const *>(indices_ptr);
  int32_t const *topk_len = static_cast<int32_t const *>(topk_length_ptr);
  float const *attn_sink = static_cast<float const *>(attn_sink_ptr);
  bf16 *out_all = static_cast<bf16 *>(out_ptr);

  bf16 const *q_h = q_all + head_id * HEAD_DIM;
  bf16 *out_h = out_all + head_id * HEAD_V;

  extern __shared__ char smem_raw[];
  float *s_q = reinterpret_cast<float *>(smem_raw);
  float *s_k = s_q + HEAD_DIM;
  float *s_acc = s_k + HEAD_DIM;

#pragma unroll 1
  for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
    s_q[i] = static_cast<float>(q_h[i]);
  }
#pragma unroll 1
  for (int i = threadIdx.x; i < HEAD_V; i += NUM_THREADS) {
    s_acc[i] = 0.0f;
  }
  __syncthreads();

  int L = topk_len[0];
  if (L <= 0) {
#pragma unroll 1
    for (int i = threadIdx.x; i < HEAD_V; i += NUM_THREADS) {
      out_h[i] = bf16(0.0f);
    }
    return;
  }

  __shared__ float s_m;
  __shared__ float s_l;
  if (threadIdx.x == 0) {
    s_m = -INFINITY;
    s_l = 0.0f;
  }
  __syncthreads();

  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
  __shared__ float s_red[NUM_WARPS];

  for (int k = 0; k < L; ++k) {
    int slot = indices[k];
    if (slot < 0) continue;

    flash_mla_with_kvcache_v4_detail::
        gather_one_k_row<HEAD_DIM, HEAD_V, ROPE_DIM, QUANT_BLOCK,
                         NUM_THREADS>(
            k_cache, block_stride_bytes, cache_block_size, slot, s_k);

    float partial = 0.0f;
#pragma unroll 1
    for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
      partial += s_q[i] * s_k[i];
    }
#pragma unroll
    for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
      partial += shfl_xor_sync(partial, offset);
    }
    int lane = threadIdx.x % NUM_THREADS_PER_WARP;
    int warp = threadIdx.x / NUM_THREADS_PER_WARP;
    if (lane == 0) s_red[warp] = partial;
    __syncthreads();
    if (warp == 0) {
      float v = (threadIdx.x < NUM_WARPS) ? s_red[threadIdx.x] : 0.0f;
#pragma unroll
      for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
        v += shfl_xor_sync(v, offset);
      }
      if (lane == 0) s_red[0] = v;
    }
    __syncthreads();
    float qk = s_red[0] * softmax_scale;

    float old_m = s_m;
    float new_m = fmaxf(old_m, qk);
    float alpha = (old_m == -INFINITY) ? 0.0f : expf(old_m - new_m);
    float p = expf(qk - new_m);
    if (threadIdx.x == 0) {
      s_m = new_m;
      s_l = s_l * alpha + p;
    }
    __syncthreads();

#pragma unroll 1
    for (int i = threadIdx.x; i < HEAD_V; i += NUM_THREADS) {
      s_acc[i] = s_acc[i] * alpha + p * s_k[i];
    }
    __syncthreads();
  }

  float l = s_l;
  float m = s_m;
  float sink_factor = 1.0f;
  if (attn_sink != nullptr && l > 0.0f) {
    float sink = attn_sink[head_id];
    float exp_sink_minus_m = expf(sink - m);
    sink_factor = l / (l + exp_sink_minus_m);
  }
  float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
#pragma unroll 1
  for (int i = threadIdx.x; i < HEAD_V; i += NUM_THREADS) {
    out_h[i] = bf16(s_acc[i] * inv_l * sink_factor);
  }
  __syncthreads();
}

} // namespace kernel
