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

// V4-Flash fused_kv_compress_norm_rope_insert_indexer_attn (FP8 K-side).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/
//       fused_kv_compress_norm_rope_insert_indexer_attn.md
// Triton ref:
//   vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:302-474
//
// Per-token kernel for the indexer's compressor (head_dim=128). Same
// gather-window + softmax + weighted-sum + RMSNorm + RoPE pipeline as
// the sparse-attn (head_dim=512) sibling, but with:
//   * head_dim=128, ROPE_HEAD_DIM=64, NOPE_HEAD_DIM=64;
//   * single quant block (QUANT_BLOCK == HEAD_SIZE == 128) -> ONE
//     fp32 scale per token (not a UE8M0 byte);
//   * the entire post-norm vector (nope + rope-rotated tail) is FP8
//     quantized as a single block.
//
// Class B coupling: this kernel is the K-side under use_fp4_cache=False.
// The Q-side must also be FP8 (fused_indexer_q_rope_quant); when
// use_fp4_cache=True both Q and K use the MXFP4 sibling pair.
//
// Cache layout (per-token, paged):
//   bytes [0, 128)                          = FP8 values    (128 e4m3)
//   per-block scale region:
//     bytes [block_size*128 + slot_in_block*4, +4) = ONE fp32 scale
//   TOKEN_STRIDE = 128, SCALE_DIM = 4.
//
// COMPRESS_RATIO is always 4 for the indexer (OVERLAP=1, W=8).

namespace kernel {

namespace fused_kv_compress_norm_rope_insert_indexer_attn_v4_detail {

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

template <int NUM_THREADS>
__device__ __forceinline__ float warp_block_reduce_max(float val,
                                                       float *smem) {
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
#pragma unroll
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    float other = shfl_xor_sync(val, offset);
    val = fmaxf(val, other);
  }
  int lane = threadIdx.x % NUM_THREADS_PER_WARP;
  int warp = threadIdx.x / NUM_THREADS_PER_WARP;
  if (lane == 0) {
    smem[warp] = val;
  }
  __syncthreads();
  float out =
      (threadIdx.x < NUM_WARPS) ? smem[threadIdx.x] : -INFINITY;
  if (warp == 0) {
#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
      float other = shfl_xor_sync(out, offset);
      out = fmaxf(out, other);
    }
    if (lane == 0) {
      smem[0] = out;
    }
  }
  __syncthreads();
  return smem[0];
}

} // namespace fused_kv_compress_norm_rope_insert_indexer_attn_v4_detail

template <int HEAD_SIZE, int COMPRESS_RATIO, int BLOCK_SIZE, int MAX_BLOCKS,
          int KV_BLOCK_SIZE, int NUM_THREADS = 256>
__device__ __forceinline__ void
    fused_kv_compress_norm_rope_insert_indexer_attn_v4_sm100_impl(
        void const *state_cache_ptr,
        void const *token_to_req_indices_ptr,
        void const *positions_ptr,
        void const *slot_mapping_ptr,
        void const *block_table_ptr,
        void const *rms_norm_weight_ptr,
        void const *cos_sin_cache_ptr,
        void const *kv_slot_mapping_ptr,
        void *k_cache_ptr,
        float rms_norm_eps) {
  static_assert(HEAD_SIZE == 128,
                "indexer compressor uses head_dim=128 only");
  static_assert(COMPRESS_RATIO == 4,
                "indexer compressor always uses COMPRESS_RATIO=4");

  constexpr int ROPE_HEAD_DIM = 64;
  constexpr int NOPE_HEAD_DIM = HEAD_SIZE - ROPE_HEAD_DIM; // 64
  constexpr int STATE_WIDTH = 2 * HEAD_SIZE;               // coff=2 -> 256
  constexpr int OVERLAP = 1;                               // ratio=4
  constexpr int W = (1 + OVERLAP) * COMPRESS_RATIO;        // 8
  constexpr int TOKEN_STRIDE = HEAD_SIZE;                  // 128
  constexpr int SCALE_DIM = 4;                             // 1 fp32 scale
  constexpr int KV_BLOCK_STRIDE =
      KV_BLOCK_SIZE * TOKEN_STRIDE + KV_BLOCK_SIZE * SCALE_DIM;
  constexpr float FP8_MAX = 448.0f;

  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a warp multiple");

  using bf16 = type::bfloat16_t;

  int64_t slot = *static_cast<int64_t const *>(slot_mapping_ptr);
  int64_t position = *static_cast<int64_t const *>(positions_ptr);
  if (slot < 0) return;
  if (((position + 1) % static_cast<int64_t>(COMPRESS_RATIO)) != 0) return;
  int64_t kv_slot = *static_cast<int64_t const *>(kv_slot_mapping_ptr);
  if (kv_slot < 0) return;
  int req_idx = *static_cast<int32_t const *>(token_to_req_indices_ptr);

  float const *__restrict__ state_cache =
      static_cast<float const *>(state_cache_ptr);
  int32_t const *__restrict__ block_table =
      static_cast<int32_t const *>(block_table_ptr);
  bf16 const *__restrict__ rms_w =
      static_cast<bf16 const *>(rms_norm_weight_ptr);
  float const *__restrict__ cos_sin_cache =
      static_cast<float const *>(cos_sin_cache_ptr);
  uint8_t *__restrict__ k_cache = static_cast<uint8_t *>(k_cache_ptr);

  // Smem: ~5 KiB worst-case for HEAD_SIZE=128, W=8.
  //   softmax_max   : float[HEAD_SIZE]
  //   softmax_sum   : float[HEAD_SIZE]
  //   accum         : float[HEAD_SIZE]
  //   norm_buf      : float[HEAD_SIZE]
  //   score_window  : float[W * HEAD_SIZE]   (= 8*128 = 1024 floats)
  //   reduce        : float[64]
  // = (4 * 128 + 8*128 + 64) * 4 = (512 + 1024 + 64) * 4 = 6400 bytes.
  extern __shared__ char smem_raw[];
  float *softmax_max  = reinterpret_cast<float *>(smem_raw);
  float *softmax_sum  = softmax_max + HEAD_SIZE;
  float *accum        = softmax_sum + HEAD_SIZE;
  float *norm_buf     = accum + HEAD_SIZE;
  float *reduce_smem  = norm_buf + HEAD_SIZE;
  float *score_window = reduce_smem + 64;

  // ------------------------------------------------------------------
  // Pass 1: gather score-half rows + compute per-element max.
  // ------------------------------------------------------------------
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    softmax_max[i] = -INFINITY;
    softmax_sum[i] = 0.0f;
  }
  __syncthreads();

  int64_t window_start = position - (W - 1);

  for (int w = 0; w < W; ++w) {
    int64_t pos_w = window_start + w;
    bool mask = (pos_w >= 0);
    int head_off = (w >= COMPRESS_RATIO) ? HEAD_SIZE : 0;

    int32_t blk_no = 0;
    int blk_off = 0;
    if (mask) {
      int64_t blk_idx = pos_w / static_cast<int64_t>(BLOCK_SIZE);
      blk_off = static_cast<int>(pos_w %
                                 static_cast<int64_t>(BLOCK_SIZE));
      blk_no = block_table[req_idx * MAX_BLOCKS + blk_idx];
    }
    int64_t row_base =
        (static_cast<int64_t>(blk_no) * BLOCK_SIZE + blk_off) *
        static_cast<int64_t>(2 * STATE_WIDTH);
    float const *__restrict__ score_row =
        state_cache + row_base + head_off + STATE_WIDTH;
    float *row_dst = score_window + w * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
      float v = mask ? score_row[i] : -INFINITY;
      row_dst[i] = v;
      softmax_max[i] = fmaxf(softmax_max[i], v);
    }
    __syncthreads();
  }

  // Pass 2: per-element sum(exp(s - max)).
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float m = softmax_max[i];
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < W; ++w) {
      s += __expf(score_window[w * HEAD_SIZE + i] - m);
    }
    softmax_sum[i] = s;
  }
  __syncthreads();

  // Pass 3: weighted-sum kv-half.
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    accum[i] = 0.0f;
  }
  __syncthreads();

  for (int w = 0; w < W; ++w) {
    int64_t pos_w = window_start + w;
    bool mask = (pos_w >= 0);
    int head_off = (w >= COMPRESS_RATIO) ? HEAD_SIZE : 0;
    int32_t blk_no = 0;
    int blk_off = 0;
    if (mask) {
      int64_t blk_idx = pos_w / static_cast<int64_t>(BLOCK_SIZE);
      blk_off = static_cast<int>(pos_w %
                                 static_cast<int64_t>(BLOCK_SIZE));
      blk_no = block_table[req_idx * MAX_BLOCKS + blk_idx];
    }
    int64_t row_base =
        (static_cast<int64_t>(blk_no) * BLOCK_SIZE + blk_off) *
        static_cast<int64_t>(2 * STATE_WIDTH);
    float const *__restrict__ kv_row = state_cache + row_base + head_off;
    float const *__restrict__ score_row =
        state_cache + row_base + head_off + STATE_WIDTH;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
      float kv_v = mask ? kv_row[i] : 0.0f;
      float s_v  = mask ? score_row[i] : -INFINITY;
      float wgt = __expf(s_v - softmax_max[i]) / softmax_sum[i];
      accum[i] += kv_v * wgt;
    }
    __syncthreads();
  }

  // ------------------------------------------------------------------
  // RMSNorm in fp32.
  // ------------------------------------------------------------------
  float partial = 0.0f;
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float v = accum[i];
    partial += v * v;
  }
  float sumsq = fused_kv_compress_norm_rope_insert_indexer_attn_v4_detail::
      warp_block_reduce_sum<NUM_THREADS>(partial, reduce_smem);
  float inv_rms =
      rsqrtf(sumsq / static_cast<float>(HEAD_SIZE) + rms_norm_eps);

  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float w = static_cast<float>(rms_w[i]);
    norm_buf[i] = accum[i] * inv_rms * w;
  }
  __syncthreads();

  // ------------------------------------------------------------------
  // Forward GPT-J RoPE on rope tail [NOPE..HEAD).
  // ------------------------------------------------------------------
  int64_t compressed_pos =
      (position / static_cast<int64_t>(COMPRESS_RATIO)) *
      static_cast<int64_t>(COMPRESS_RATIO);
  float const *__restrict__ cs_row =
      cos_sin_cache + compressed_pos * ROPE_HEAD_DIM;
  constexpr int ROPE_PAIRS = ROPE_HEAD_DIM / 2; // 32

  // Apply RoPE in-place in norm_buf for the rope tail.
  // First read all even/odd then write back (avoid stomp).
  for (int p = threadIdx.x; p < ROPE_PAIRS; p += NUM_THREADS) {
    float even = norm_buf[NOPE_HEAD_DIM + 2 * p + 0];
    float odd  = norm_buf[NOPE_HEAD_DIM + 2 * p + 1];
    float cos = cs_row[p];
    float sin = cs_row[ROPE_PAIRS + p];
    float new_even = even * cos - odd * sin;
    float new_odd  = odd  * cos + even * sin;
    norm_buf[NOPE_HEAD_DIM + 2 * p + 0] = new_even;
    norm_buf[NOPE_HEAD_DIM + 2 * p + 1] = new_odd;
  }
  __syncthreads();

  // ------------------------------------------------------------------
  // Single-block FP8 quant of the full HEAD_SIZE vector.
  // bf16 roundtrip to match reference numerics.
  // ------------------------------------------------------------------
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    bf16 v_bf16 = bf16(norm_buf[i]);
    norm_buf[i] = static_cast<float>(v_bf16);
  }
  __syncthreads();

  // Per-token absmax across HEAD_SIZE.
  float partial_max = 0.0f;
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    partial_max = fmaxf(partial_max, fabsf(norm_buf[i]));
  }
  float amax = fused_kv_compress_norm_rope_insert_indexer_attn_v4_detail::
      warp_block_reduce_max<NUM_THREADS>(partial_max, reduce_smem);
  amax = fmaxf(amax, 1e-4f);
  float exponent = ceilf(__log2f(amax / FP8_MAX));
  float inv_scale = exp2f(-exponent);
  float scale_fp32 = exp2f(exponent); // stored per-token

  // ------------------------------------------------------------------
  // Cache write: FP8 values + fp32 scale.
  // ------------------------------------------------------------------
  int64_t kv_blk = kv_slot / static_cast<int64_t>(KV_BLOCK_SIZE);
  int kv_off = static_cast<int>(kv_slot %
                                static_cast<int64_t>(KV_BLOCK_SIZE));
  uint8_t *blk_base =
      k_cache + kv_blk * static_cast<int64_t>(KV_BLOCK_STRIDE);
  uint8_t *token_data =
      blk_base + static_cast<int64_t>(kv_off) * TOKEN_STRIDE;
  float *token_scale_fp32 = reinterpret_cast<float *>(
      blk_base + KV_BLOCK_SIZE * TOKEN_STRIDE + kv_off * SCALE_DIM);

  __nv_fp8_e4m3 *fp8_out = reinterpret_cast<__nv_fp8_e4m3 *>(token_data);
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float v = norm_buf[i] * inv_scale;
    v = fminf(fmaxf(v, -FP8_MAX), FP8_MAX);
    fp8_out[i] = __nv_fp8_e4m3(v);
  }
  if (threadIdx.x == 0) {
    *token_scale_fp32 = scale_fp32;
  }
}

} // namespace kernel
