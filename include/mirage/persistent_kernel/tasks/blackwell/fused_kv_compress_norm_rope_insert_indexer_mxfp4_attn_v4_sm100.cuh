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

// V4-Flash fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn (MXFP4
// K-side; NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/
//       fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn.md
// Triton ref:
//   vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:479-667
//
// Per-token kernel for the indexer compressor (head_dim=128) when the
// model is built with use_fp4_cache=True. Same gather-window + softmax +
// weighted-sum + RMSNorm + RoPE pipeline as the FP8 K-side sibling, but
// the quant tail is MXFP4:
//   * 4 quant blocks of 32 elements (HEAD_SIZE / QUANT_BLOCK = 128/32);
//   * each block's scale = ceil(log2(amax / 6.0)) -> UE8M0 byte
//     (E2M1 max magnitude = 6.0);
//   * per-block scale uses both (new_even, new_odd) halves' absmax
//     (the rotated rope pairs interleave the two halves);
//   * values packed two E2M1 nibbles per byte (low = even-position,
//     high = odd-position).
//
// Class-B coupling: this K-side is selected by use_fp4_cache=True; the
// paired Q-side is fused_indexer_q_rope_mxfp4 (NOT fused_indexer_q_rope_quant).
//
// Cache layout (per-token, paged):
//   bytes [0, TOKEN_STRIDE=64) = MXFP4 packed values (128 E2M1 nibbles).
//   per-block scale region:
//     bytes [block_size*64 + slot_in_block*4, +4) = 4 UE8M0 bytes
//       (one per 32-element quant block).
//   The same physical cache slot size (132 bytes) is allocated as the
//   FP8 sibling, but only 68 of those bytes are used (the rest is
//   padding).

namespace kernel {

namespace fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn_v4_detail {

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

// Quantize a single fp32 value to a 4-bit E2M1 code (sign(1) + exp(2)
// + mantissa(1)). Representable magnitudes: {0,0.5,1,1.5,2,3,4,6}.
// Round-to-nearest-even via midpoint table; saturating to 6.0.
__device__ __forceinline__ uint8_t fp32_to_e2m1(float v) {
  uint8_t sign = v < 0.0f ? 0x8 : 0x0;
  float a = v < 0.0f ? -v : v;
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
    mag = 7; // 6.0
  }
  return static_cast<uint8_t>(sign | mag);
}

} // namespace fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn_v4_detail

template <int HEAD_SIZE, int COMPRESS_RATIO, int BLOCK_SIZE, int MAX_BLOCKS,
          int KV_BLOCK_SIZE, int NUM_THREADS = 256>
__device__ __forceinline__ void
    fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn_v4_sm100_impl(
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
  constexpr int STATE_WIDTH = 2 * HEAD_SIZE;               // 256
  constexpr int OVERLAP = 1;
  constexpr int W = (1 + OVERLAP) * COMPRESS_RATIO; // 8
  constexpr int QUANT_BLOCK = 32;                   // MXFP4 block size
  constexpr int N_BLOCKS = HEAD_SIZE / QUANT_BLOCK; // 4
  constexpr int TOKEN_STRIDE = HEAD_SIZE / 2;       // 64 packed bytes
  constexpr int SCALE_DIM = N_BLOCKS;               // 4 UE8M0 bytes
  // The cache slot is allocated with the FP8 size (132 bytes / token);
  // the MXFP4 path uses 64 (data) + 4 (scale) = 68 bytes and ignores
  // the rest. We mirror the FP8 sibling's layout: data region of size
  // KV_BLOCK_SIZE * 128 bytes followed by a scale region of size
  // KV_BLOCK_SIZE * 4 bytes. The MXFP4 quant data lives in the first
  // 64 bytes of each 128-byte data slot.
  constexpr int FP8_TOKEN_STRIDE = 128; // physical data stride in bytes
  constexpr int FP8_SCALE_DIM = 4;      // physical scale stride in bytes
  constexpr int KV_BLOCK_STRIDE =
      KV_BLOCK_SIZE * FP8_TOKEN_STRIDE + KV_BLOCK_SIZE * FP8_SCALE_DIM;
  constexpr float E2M1_MAX = 6.0f;

  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a warp multiple");

  using bf16 = type::bfloat16_t;
  namespace detail =
      fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn_v4_detail;

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

  // Same smem layout as the FP8 indexer sibling.
  extern __shared__ char smem_raw[];
  float *softmax_max  = reinterpret_cast<float *>(smem_raw);
  float *softmax_sum  = softmax_max + HEAD_SIZE;
  float *accum        = softmax_sum + HEAD_SIZE;
  float *norm_buf     = accum + HEAD_SIZE;
  float *reduce_smem  = norm_buf + HEAD_SIZE;
  float *score_window = reduce_smem + 64;

  // ------------------------------------------------------------------
  // Pass 1: gather score rows + per-element max.
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
  float sumsq = detail::warp_block_reduce_sum<NUM_THREADS>(partial, reduce_smem);
  float inv_rms =
      rsqrtf(sumsq / static_cast<float>(HEAD_SIZE) + rms_norm_eps);

  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float w = static_cast<float>(rms_w[i]);
    norm_buf[i] = accum[i] * inv_rms * w;
  }
  __syncthreads();

  // ------------------------------------------------------------------
  // Forward GPT-J RoPE on rope tail. MXFP4 packs (new_even, new_odd)
  // pairs (low nibble / high nibble), so we leave them in pair-split
  // form in norm_buf:
  //   norm_buf[NOPE + 2p + 0] = new_even (post-rotation)
  //   norm_buf[NOPE + 2p + 1] = new_odd  (post-rotation)
  // This matches the FP8 indexer sibling's layout, so the per-block
  // amax for blocks straddling the nope/rope boundary is well-defined.
  // ------------------------------------------------------------------
  int64_t compressed_pos =
      (position / static_cast<int64_t>(COMPRESS_RATIO)) *
      static_cast<int64_t>(COMPRESS_RATIO);
  float const *__restrict__ cs_row =
      cos_sin_cache + compressed_pos * ROPE_HEAD_DIM;
  constexpr int ROPE_PAIRS = ROPE_HEAD_DIM / 2; // 32

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

  // bf16 roundtrip for the entire HEAD_SIZE vector (matches the
  // reference numerics; the Triton version casts (new_even, new_odd)
  // halves bf16->fp32 before MXFP4 quant).
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    bf16 v_bf16 = bf16(norm_buf[i]);
    norm_buf[i] = static_cast<float>(v_bf16);
  }
  __syncthreads();

  // ------------------------------------------------------------------
  // MXFP4 quant: per-block (32-elem) amax + UE8M0 scale + E2M1 pack.
  // ------------------------------------------------------------------
  int64_t kv_blk = kv_slot / static_cast<int64_t>(KV_BLOCK_SIZE);
  int kv_off = static_cast<int>(kv_slot %
                                static_cast<int64_t>(KV_BLOCK_SIZE));
  uint8_t *blk_base =
      k_cache + kv_blk * static_cast<int64_t>(KV_BLOCK_STRIDE);
  // MXFP4 values live in the first 64 bytes of each 128-byte data slot.
  uint8_t *token_data =
      blk_base + static_cast<int64_t>(kv_off) * FP8_TOKEN_STRIDE;
  uint8_t *token_scale =
      blk_base + KV_BLOCK_SIZE * FP8_TOKEN_STRIDE + kv_off * FP8_SCALE_DIM;

  for (int b = 0; b < N_BLOCKS; ++b) {
    int base = b * QUANT_BLOCK;
    // Per-block absmax (over QUANT_BLOCK=32 elems).
    float partial_max = 0.0f;
    for (int i = threadIdx.x; i < QUANT_BLOCK; i += NUM_THREADS) {
      partial_max = fmaxf(partial_max, fabsf(norm_buf[base + i]));
    }
    float amax = detail::warp_block_reduce_max<NUM_THREADS>(partial_max,
                                                            reduce_smem);
    // MXFP4 subnormal floor: 6.0 * 2^-126.
    amax = fmaxf(amax, 6.0f * exp2f(-126.0f));
    float log2_ratio = ceilf(__log2f(amax / E2M1_MAX));
    // Clamp into byte range; encoded as byte = log2_ratio + 127.
    if (log2_ratio < -127.0f) log2_ratio = -127.0f;
    if (log2_ratio > 127.0f)  log2_ratio = 127.0f;
    float inv_scale = exp2f(-log2_ratio);
    if (threadIdx.x == 0) {
      int byte = static_cast<int>(log2_ratio) + 127;
      if (byte < 0) byte = 0;
      if (byte > 254) byte = 254;
      token_scale[b] = static_cast<uint8_t>(byte);
    }
    // Pack 32 nibbles into 16 bytes. byte i in [0, 16):
    //   low nibble = E2M1(norm_buf[base + 2i + 0] * inv_scale)
    //   high nibble = E2M1(norm_buf[base + 2i + 1] * inv_scale)
    int pack_base = base / 2; // 0, 16, 32, 48
    for (int i = threadIdx.x; i < QUANT_BLOCK / 2; i += NUM_THREADS) {
      float lo_f = norm_buf[base + 2 * i + 0] * inv_scale;
      float hi_f = norm_buf[base + 2 * i + 1] * inv_scale;
      uint8_t lo = detail::fp32_to_e2m1(lo_f);
      uint8_t hi = detail::fp32_to_e2m1(hi_f);
      token_data[pack_base + i] =
          static_cast<uint8_t>((hi << 4) | (lo & 0x0F));
    }
    __syncthreads();
  }
}

} // namespace kernel
