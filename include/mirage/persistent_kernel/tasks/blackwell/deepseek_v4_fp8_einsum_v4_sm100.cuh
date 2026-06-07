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

// V4-Flash deepseek_v4_fp8_einsum (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/deepseek_v4_fp8_einsum.md
//
// Einsum equation: "bhr,hdr->bhd"
//   - b = num_tokens (T)
//   - h = n_local_groups
//   - r = heads_per_group * head_dim    (contracted)
//   - d = o_lora_rank                   (output last dim)
//
// I/O contract (matches wave-5A `fused_inv_rope_fp8_quant` producer on sm_100a):
//   - o_fp8     : float8_e4m3fn [T, n_groups, d_in=r]    (caller-visible, post-transpose)
//   - o_scale   : int32         [T, n_groups, scale_inner]
//                 (UE8M0-packed: 4 exponent bytes per int32; scale_inner =
//                  ceil(num_blocks_per_head*heads_per_group / 4))
//   - wo_a_fp8  : float8_e4m3fn [n_groups, d_out, d_in]
//   - wo_a_scale: int32         [n_groups, d_out, scale_inner_w]
//                 (UE8M0-packed, same encoding as o_scale; one packed
//                  uint32 covers 4 of the 128-wide r-blocks)
//   - out       : bf16          [T, n_groups, d_out]
//
// Math: out[t, h, dout] = sum_{r_blk} sum_{r in r_blk}
//                           ( o_fp8[h, t, r] * o_scale[h, t, r_blk] )
//                         * ( wo_a_fp8[h, dout, r] * wo_a_scale[h, dout, r_blk] )
// where scale[..., r_blk] = exp2(ue8m0_byte - 127).
//
// Naive design (correctness only):
//   - One CTA per (token, group). grid = (T, n_groups, 1).
//   - For each (t, h) the CTA loops over d_out outputs sequentially;
//     for each output index dout, threads stripe across d_in (r) and
//     accumulate fp32 dequantized products with block-scale.
//   - block + warp reduction; thread 0 writes bf16 out.
//
// NOTE: this kernel uses the on-device `o_fp8`/`o_scale` layout
// `[n_groups, T, ...]` produced by `fused_inv_rope_fp8_quant`
// *before* the caller's `.transpose(0, 1)` view.

namespace kernel {

namespace deepseek_v4_fp8_einsum_v4_detail {

__device__ __forceinline__ float ue8m0_decode_byte(unsigned char b) {
  // scale = 2^(b - 127). exp2f handles negative integer powers cleanly.
  return exp2f(static_cast<float>(static_cast<int>(b) - 127));
}

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

} // namespace deepseek_v4_fp8_einsum_v4_detail

template <int D_IN, int D_OUT, int QUANT_BLOCK = 128, int NUM_THREADS = 256>
__device__ __forceinline__ void deepseek_v4_fp8_einsum_v4_sm100_impl(
    void const *o_fp8_ptr,        // fp8 [d_in]            (this (h, t) row)
    void const *o_scale_ptr,      // int32 [scale_inner]   (this (h, t) row, UE8M0-packed)
    void const *wo_a_fp8_ptr,     // fp8 [d_out, d_in]     (this group's weight slab)
    void const *wo_a_scale_ptr,   // int32 [d_out, scale_inner] (this group's weight scales)
    void *out_ptr                 // bf16 [d_out]          (this (t, h) output row)
) {
  static_assert(D_IN > 0 && D_OUT > 0, "D_IN and D_OUT must be positive");
  static_assert(D_IN % QUANT_BLOCK == 0, "D_IN must be divisible by QUANT_BLOCK");
  static_assert(NUM_THREADS > 0 && NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a positive warp multiple");

  constexpr int N_R_BLOCKS = D_IN / QUANT_BLOCK;
  constexpr int SCALES_PER_INT32 = 4;
  constexpr int SCALE_INNER = (N_R_BLOCKS + SCALES_PER_INT32 - 1) /
                              SCALES_PER_INT32;

  using bf16 = type::bfloat16_t;

  unsigned char const *__restrict__ o_fp8_bytes =
      static_cast<unsigned char const *>(o_fp8_ptr);
  uint32_t const *__restrict__ o_scale_packed =
      static_cast<uint32_t const *>(o_scale_ptr);
  unsigned char const *__restrict__ w_fp8_bytes =
      static_cast<unsigned char const *>(wo_a_fp8_ptr);
  uint32_t const *__restrict__ w_scale_packed =
      static_cast<uint32_t const *>(wo_a_scale_ptr);
  bf16 *__restrict__ out_row = static_cast<bf16 *>(out_ptr);

  __shared__ float reduce_smem[NUM_THREADS / NUM_THREADS_PER_WARP];

  // Decode all activation scales for this (h, t) into shared once per CTA.
  __shared__ float o_scale_decoded[N_R_BLOCKS];
  if (threadIdx.x < N_R_BLOCKS) {
    int packed_idx = threadIdx.x / SCALES_PER_INT32;
    int byte_idx = threadIdx.x % SCALES_PER_INT32;
    uint32_t packed = o_scale_packed[packed_idx];
    unsigned char b =
        static_cast<unsigned char>((packed >> (byte_idx * 8)) & 0xFF);
    o_scale_decoded[threadIdx.x] =
        deepseek_v4_fp8_einsum_v4_detail::ue8m0_decode_byte(b);
  }
  __syncthreads();

#pragma unroll 1
  for (int dout = 0; dout < D_OUT; ++dout) {
    float acc = 0.0f;

    // Iterate over r-blocks; within each block stripe across threads.
#pragma unroll 1
    for (int rb = 0; rb < N_R_BLOCKS; ++rb) {
      // Decode the matching weight scale byte for this (dout, rb).
      int packed_idx = rb / SCALES_PER_INT32;
      int byte_idx = rb % SCALES_PER_INT32;
      uint32_t packed = w_scale_packed[
          static_cast<long long>(dout) * static_cast<long long>(SCALE_INNER) +
          packed_idx];
      unsigned char wb =
          static_cast<unsigned char>((packed >> (byte_idx * 8)) & 0xFF);
      float w_scale =
          deepseek_v4_fp8_einsum_v4_detail::ue8m0_decode_byte(wb);

      float a_scale = o_scale_decoded[rb];
      float block_scale = a_scale * w_scale;

      int r_base = rb * QUANT_BLOCK;
      float local = 0.0f;
      for (int r = threadIdx.x; r < QUANT_BLOCK; r += NUM_THREADS) {
        // a_fp8: [h, t, r] -- but this CTA already received the per-(h,t) row,
        // so o_fp8_bytes indexes by r_base + r in [0, D_IN).
        unsigned char a_byte = o_fp8_bytes[r_base + r];
        __nv_fp8_e4m3 a_packed;
        *reinterpret_cast<unsigned char *>(&a_packed) = a_byte;
        float a = static_cast<float>(a_packed);

        // w_fp8: [h, d_out, d_in] -- this group's slab; row dout, col (r_base+r).
        unsigned char w_byte = w_fp8_bytes[
            static_cast<long long>(dout) * static_cast<long long>(D_IN) +
            static_cast<long long>(r_base + r)];
        __nv_fp8_e4m3 w_packed;
        *reinterpret_cast<unsigned char *>(&w_packed) = w_byte;
        float w = static_cast<float>(w_packed);

        local += a * w;
      }
      acc += local * block_scale;
    }

    // Cross-thread reduce.
    float final_val =
        deepseek_v4_fp8_einsum_v4_detail::warp_block_reduce_sum<NUM_THREADS>(
            acc, reduce_smem);

    if (threadIdx.x == 0) {
      out_row[dout] = static_cast<bf16>(final_val);
    }
    __syncthreads();
  }
}

} // namespace kernel
