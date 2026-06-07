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
#include <cuda_fp8.h>

// V4-Flash quantize_and_insert_k_kernel (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/quantize_and_insert_k_kernel.md
//
// UE8M0 FP8 quantization of compressed K + insert into paged cache.
//   - 7 real quant blocks (each 64 fp32 lanes -> 64 fp8 bytes + 1 UE8M0
//     exponent byte) covering NoPE lanes [0, 448).
//   - 1 padding scale byte at index 7 (always zero) so per-token scale
//     region is 8 bytes (aligned).
//   - RoPE bf16 [448, 512) written verbatim as 128 bytes.
//
// Naive design: one CTA per input token; NUM_THREADS = 256. Each token
// processes 7 quant blocks sequentially. Skips slot_mapping == -1.

namespace kernel {

namespace quantize_and_insert_k_v4_detail {

__device__ __forceinline__ unsigned char ue8m0_byte_from_exponent(int exponent) {
  int enc = exponent + 127;
  if (enc < 0) enc = 0;
  if (enc > 255) enc = 255;
  return static_cast<unsigned char>(enc);
}

} // namespace quantize_and_insert_k_v4_detail

template <int CACHE_BLOCK_SIZE = 64,
          int FP8_DIM = 448,
          int BF16_DIM = 64,
          int QUANT_BLOCK = 64,
          int NUM_THREADS = 256>
__device__ __forceinline__ void quantize_and_insert_k_v4_sm100_impl(
    void const *k_ptr,            // bf16 [512] -- this token's row
    void const *slot_mapping_ptr, // int64 [1] -- this token's slot id
    void *k_cache_ptr,            // uint8 [num_blocks, block_stride] raw bytes
    int block_stride              // bytes per paged block
) {
  static_assert(CACHE_BLOCK_SIZE == 64, "SWA cache locked to block_size=64");
  static_assert(FP8_DIM == 448, "NoPE 448 lanes locked");
  static_assert(BF16_DIM == 64, "RoPE 64 lanes locked");
  static_assert(QUANT_BLOCK == 64, "UE8M0 quant block size = 64");
  static_assert(NUM_THREADS > 0 && NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a positive warp multiple");

  constexpr int N_QUANT_BLOCKS_REAL = FP8_DIM / QUANT_BLOCK; // 7
  constexpr int SCALE_DIM = 8;                                // 7 real + 1 pad
  constexpr int TOKEN_DATA_SIZE = FP8_DIM + 2 * BF16_DIM;     // 576
  constexpr float FP8_MAX = 448.0f;
  constexpr float AMAX_FLOOR = 1e-4f;

  using bf16 = type::bfloat16_t;

  bf16 const *__restrict__ k_in = static_cast<bf16 const *>(k_ptr);
  long long const *__restrict__ slot_in =
      static_cast<long long const *>(slot_mapping_ptr);
  unsigned char *__restrict__ k_cache_bytes =
      static_cast<unsigned char *>(k_cache_ptr);

  long long slot = slot_in[0];
  if (slot < 0) {
    return;
  }

  long long block_idx = slot / CACHE_BLOCK_SIZE;
  int pos_in_block = static_cast<int>(slot % CACHE_BLOCK_SIZE);
  long long base = block_idx * static_cast<long long>(block_stride);

  unsigned char *token_data_ptr =
      k_cache_bytes + base + static_cast<long long>(pos_in_block) *
                                  static_cast<long long>(TOKEN_DATA_SIZE);
  unsigned char *token_scale_ptr =
      k_cache_bytes + base +
      static_cast<long long>(CACHE_BLOCK_SIZE) *
          static_cast<long long>(TOKEN_DATA_SIZE) +
      static_cast<long long>(pos_in_block) *
          static_cast<long long>(SCALE_DIM);

  // Two-warp amax reduction (QUANT_BLOCK = 64 = 2 warps' worth of lanes).
  __shared__ float s_warp_amax[2];
  __shared__ float s_scale;

#pragma unroll 1
  for (int qb = 0; qb < N_QUANT_BLOCKS_REAL; ++qb) {
    int q_off = qb * QUANT_BLOCK;

    float my_v = 0.0f;
    if (threadIdx.x < QUANT_BLOCK) {
      my_v = static_cast<float>(k_in[q_off + threadIdx.x]);
    }
    float amax = (threadIdx.x < QUANT_BLOCK) ? fabsf(my_v) : 0.0f;
    // Warp reduce.
#pragma unroll
    for (int off = NUM_THREADS_PER_WARP / 2; off > 0; off /= 2) {
      amax = fmaxf(amax, shfl_xor_sync(amax, off));
    }
    int lane = threadIdx.x % NUM_THREADS_PER_WARP;
    int warp = threadIdx.x / NUM_THREADS_PER_WARP;
    if (lane == 0 && warp < 2) {
      s_warp_amax[warp] = amax;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      float global_amax = fmaxf(s_warp_amax[0], s_warp_amax[1]);
      if (global_amax < AMAX_FLOOR) {
        global_amax = AMAX_FLOOR;
      }
      // UE8M0: scale = 2^ceil(log2(amax / FP8_MAX)). Extract exponent
      // via IEEE 754 bits so an exact power-of-two doesn't round up
      // under fast-math.
      float raw = global_amax / FP8_MAX;
      raw = fmaxf(raw, 1e-30f);
      uint32_t bits = __float_as_uint(raw);
      int exp_unbiased = static_cast<int>((bits >> 23) & 0xFF) - 127;
      uint32_t mantissa = bits & 0x7FFFFF;
      int exponent = (mantissa == 0) ? exp_unbiased : exp_unbiased + 1;
      s_scale = exp2f(static_cast<float>(exponent));
      token_scale_ptr[qb] =
          quantize_and_insert_k_v4_detail::ue8m0_byte_from_exponent(exponent);
    }
    __syncthreads();

    float scale = s_scale;
    if (threadIdx.x < QUANT_BLOCK) {
      float q = my_v / scale;
      if (q > FP8_MAX) q = FP8_MAX;
      if (q < -FP8_MAX) q = -FP8_MAX;
      __nv_fp8_e4m3 packed = __nv_fp8_e4m3(q);
      unsigned char byte = *reinterpret_cast<unsigned char *>(&packed);
      token_data_ptr[q_off + threadIdx.x] = byte;
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    token_scale_ptr[N_QUANT_BLOCKS_REAL] = 0;
  }

  // Pass-through bf16 RoPE (64 lanes -> 128 bytes).
  unsigned char *rope_dst = token_data_ptr + FP8_DIM;
  if (threadIdx.x < BF16_DIM) {
    bf16 v = k_in[FP8_DIM + threadIdx.x];
    unsigned short bits = *reinterpret_cast<unsigned short *>(&v);
    rope_dst[2 * threadIdx.x + 0] = static_cast<unsigned char>(bits & 0xFF);
    rope_dst[2 * threadIdx.x + 1] =
        static_cast<unsigned char>((bits >> 8) & 0xFF);
  }
  __syncthreads();
}

} // namespace kernel
