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

// V4-Flash fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert (NAIVE
// Blackwell SM100 impl). THE BIG ONE: 5-op monolithic kernel.
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.md
//
// Per-token:
//   1. RMSNorm (no learnable weight) per Q head.
//   2. Forward GPT-J RoPE on Q[..., 448:512].
//   3. Forward GPT-J RoPE on KV[..., 448:512].
//   4. UE8M0 FP8 block-quant on KV[..., 0:448] in 7 blocks of 64 each.
//   5. Scatter the quantized KV (FP8 + UE8M0 scales + bf16 RoPE)
//      into the paged uint8 cache at slot_mapping[t].
// Pad heads in [num_heads_q, q_head_padded) are zero-filled.
//
// Inputs (TBGraph order):
//   input_ptrs[0] = q_in           bf16  [NUM_HEADS_Q, HEAD_DIM=512]
//   input_ptrs[1] = kv_in          bf16  [HEAD_DIM=512]
//   input_ptrs[2] = slot_mapping   int64 (scalar after partition)
//   input_ptrs[3] = positions      int64 (scalar after partition)
//   input_ptrs[4] = cos_sin_cache  fp32  [max_pos, ROPE_DIM=64]
//
// Outputs:
//   output_ptrs[0] = q_out         bf16  [Q_HEAD_PADDED, HEAD_DIM=512]
//   output_ptrs[1] = k_cache       uint8 [num_blocks, block_stride_bytes]
//
// Cache layout per token slot (slot_id = slot_mapping[t], when >= 0):
//   block_idx     = slot_id / cache_block_size
//   pos_in_block  = slot_id % cache_block_size
//   block_base    = k_cache + block_idx * block_stride_bytes
//   block_base + pos_in_block * 576                  : 448 B fp8 nope
//   block_base + pos_in_block * 576 + 448            : 128 B bf16 rope
//   block_base + cache_block_size * 576 + pos_in_block * 8 : 7 B UE8M0 + 1 pad

namespace kernel {

namespace fused_dsv4_qnorm_rope_kv_insert_v4_detail {

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
  if (lane == 0) smem[warp] = val;
  __syncthreads();
  float out = (threadIdx.x < NUM_WARPS) ? smem[threadIdx.x] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
      out += shfl_xor_sync(out, offset);
    }
    if (lane == 0) smem[0] = out;
  }
  __syncthreads();
  return smem[0];
}

template <int NUM_THREADS>
__device__ __forceinline__ float warp_block_reduce_max(float val,
                                                       float *smem) {
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
#pragma unroll
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    val = fmaxf(val, shfl_xor_sync(val, offset));
  }
  int lane = threadIdx.x % NUM_THREADS_PER_WARP;
  int warp = threadIdx.x / NUM_THREADS_PER_WARP;
  if (lane == 0) smem[warp] = val;
  __syncthreads();
  float out = (threadIdx.x < NUM_WARPS) ? smem[threadIdx.x] : -INFINITY;
  if (warp == 0) {
#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
      out = fmaxf(out, shfl_xor_sync(out, offset));
    }
    if (lane == 0) smem[0] = out;
  }
  __syncthreads();
  return smem[0];
}

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

} // namespace fused_dsv4_qnorm_rope_kv_insert_v4_detail

template <int HEAD_DIM,
          int ROPE_DIM,
          int QUANT_BLOCK,
          int NUM_HEADS_Q,
          int Q_HEAD_PADDED,
          int NUM_THREADS = 256>
__device__ __forceinline__ void fused_dsv4_qnorm_rope_kv_insert_v4_sm100_impl(
    void const *q_in_ptr,
    void const *kv_in_ptr,
    void const *slot_mapping_ptr,
    void const *positions_ptr,
    void const *cos_sin_cache_ptr,
    void *q_out_ptr,
    void *k_cache_ptr,
    int block_stride_bytes,
    int cache_block_size,
    float eps) {
  static_assert(HEAD_DIM == 512, "V4-Flash HEAD_DIM is 512");
  static_assert(ROPE_DIM == 64, "V4-Flash ROPE_DIM is 64");
  static_assert(QUANT_BLOCK == 64, "V4-Flash quant block is 64");
  constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
  constexpr int NUM_QUANT_BLOCKS = NOPE_DIM / QUANT_BLOCK;
  constexpr int SCALE_BYTES_PER_TOKEN = NUM_QUANT_BLOCKS + 1;

  using bf16 = type::bfloat16_t;

  extern __shared__ char smem_raw[];
  float *reduce_smem = reinterpret_cast<float *>(smem_raw);

  bf16 const *q_in = static_cast<bf16 const *>(q_in_ptr);
  bf16 const *kv_in = static_cast<bf16 const *>(kv_in_ptr);
  int64_t pos = *static_cast<int64_t const *>(positions_ptr);
  int64_t slot_id = *static_cast<int64_t const *>(slot_mapping_ptr);
  float const *cos_sin = static_cast<float const *>(cos_sin_cache_ptr);
  bf16 *q_out = static_cast<bf16 *>(q_out_ptr);
  uint8_t *k_cache = static_cast<uint8_t *>(k_cache_ptr);

  float const *cos_table = cos_sin + pos * ROPE_DIM;
  float const *sin_table = cos_table + ROPE_DIM / 2;

  // ---- Stage 1: live-Q heads (RMSNorm no-weight + RoPE forward) ----
  for (int h = 0; h < NUM_HEADS_Q; ++h) {
    bf16 const *q_h_in = q_in + h * HEAD_DIM;
    bf16 *q_h_out = q_out + h * HEAD_DIM;

    float partial = 0.0f;
#pragma unroll 1
    for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
      float v = static_cast<float>(q_h_in[i]);
      partial += v * v;
    }
    float sumsq = fused_dsv4_qnorm_rope_kv_insert_v4_detail::
        warp_block_reduce_sum<NUM_THREADS>(partial, reduce_smem);
    float inv_rms = rsqrtf(sumsq / static_cast<float>(HEAD_DIM) + eps);

#pragma unroll 1
    for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
      float v = static_cast<float>(q_h_in[i]) * inv_rms;
      if (i >= NOPE_DIM) {
        int r = i - NOPE_DIM;
        int pair = r >> 1;
        bool is_even = ((r & 1) == 0);
        int partner_lane = NOPE_DIM + (r ^ 1);
        float partner =
            static_cast<float>(q_h_in[partner_lane]) * inv_rms;
        float c = cos_table[pair];
        float s = sin_table[pair];
        float out;
        if (is_even) {
          out = v * c - partner * s;
        } else {
          out = v * c + partner * s;
        }
        q_h_out[i] = bf16(out);
      } else {
        q_h_out[i] = bf16(v);
      }
    }
    __syncthreads();
  }

  // ---- Stage 2: pad-Q zero fill ----
  for (int h = NUM_HEADS_Q; h < Q_HEAD_PADDED; ++h) {
    bf16 *q_h_out = q_out + h * HEAD_DIM;
#pragma unroll 1
    for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
      q_h_out[i] = bf16(0.0f);
    }
  }
  __syncthreads();

  // ---- Stage 3: KV slot (RoPE, FP8 quant, scatter) ----
  bool has_slot = (slot_id >= 0);

  __shared__ float s_kv[HEAD_DIM];
#pragma unroll 1
  for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
    float v = static_cast<float>(kv_in[i]);
    if (i >= NOPE_DIM) {
      int r = i - NOPE_DIM;
      int pair = r >> 1;
      bool is_even = ((r & 1) == 0);
      int partner_lane = NOPE_DIM + (r ^ 1);
      float partner = static_cast<float>(kv_in[partner_lane]);
      float c = cos_table[pair];
      float s = sin_table[pair];
      float out;
      if (is_even) {
        out = v * c - partner * s;
      } else {
        out = v * c + partner * s;
      }
      s_kv[i] = out;
    } else {
      s_kv[i] = v;
    }
  }
  __syncthreads();

  if (!has_slot) return;

  int64_t block_idx = slot_id / cache_block_size;
  int pos_in_block = static_cast<int>(slot_id % cache_block_size);
  uint8_t *block_base =
      k_cache + block_idx * static_cast<int64_t>(block_stride_bytes);

  __shared__ float s_scale_inv[NUM_QUANT_BLOCKS];
  __shared__ int s_exponent[NUM_QUANT_BLOCKS];

  for (int qb = 0; qb < NUM_QUANT_BLOCKS; ++qb) {
    int base = qb * QUANT_BLOCK;
    float local = 0.0f;
#pragma unroll 1
    for (int j = threadIdx.x; j < QUANT_BLOCK; j += NUM_THREADS) {
      local = fmaxf(local, fabsf(s_kv[base + j]));
    }
    float absmax = fused_dsv4_qnorm_rope_kv_insert_v4_detail::
        warp_block_reduce_max<NUM_THREADS>(local, reduce_smem);
    absmax = fmaxf(absmax, 1.0e-4f);
    float ratio = absmax * (1.0f / 448.0f);
    int e = static_cast<int>(ceilf(log2f(ratio)));
    if (threadIdx.x == 0) {
      s_scale_inv[qb] = exp2f(static_cast<float>(-e));
      s_exponent[qb] = e;
    }
  }
  __syncthreads();

  uint8_t *token_fp8_ptr = block_base + pos_in_block * 576;
#pragma unroll 1
  for (int i = threadIdx.x; i < NOPE_DIM; i += NUM_THREADS) {
    int qb = i / QUANT_BLOCK;
    float v = s_kv[i] * s_scale_inv[qb];
    v = fminf(fmaxf(v, -448.0f), 448.0f);
    token_fp8_ptr[i] = fused_dsv4_qnorm_rope_kv_insert_v4_detail::
        fp32_to_fp8_e4m3_satfinite(v);
  }

  uint8_t *token_rope_ptr = token_fp8_ptr + NOPE_DIM;
#pragma unroll 1
  for (int r = threadIdx.x; r < ROPE_DIM; r += NUM_THREADS) {
    bf16 v = bf16(s_kv[NOPE_DIM + r]);
    reinterpret_cast<bf16 *>(token_rope_ptr)[r] = v;
  }

  uint8_t *token_scale_ptr =
      block_base + cache_block_size * 576 +
      pos_in_block * SCALE_BYTES_PER_TOKEN;
  if (threadIdx.x < NUM_QUANT_BLOCKS) {
    int e = s_exponent[threadIdx.x];
    int biased = e + 127;
    if (biased < 0) biased = 0;
    if (biased > 255) biased = 255;
    token_scale_ptr[threadIdx.x] = static_cast<uint8_t>(biased);
  } else if (threadIdx.x == NUM_QUANT_BLOCKS) {
    token_scale_ptr[NUM_QUANT_BLOCKS] = 0u;
  }
  __syncthreads();
}

} // namespace kernel
