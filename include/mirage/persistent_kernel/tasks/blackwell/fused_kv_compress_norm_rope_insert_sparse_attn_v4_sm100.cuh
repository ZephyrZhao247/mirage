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

// V4-Flash fused_kv_compress_norm_rope_insert_sparse_attn (NAIVE Blackwell).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/
//       fused_kv_compress_norm_rope_insert_sparse_attn.md
// Triton ref:
//   vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:112-297
//
// Per-token compressor kernel for the attention compressor (head_dim=512):
//   1. Early-exit on slot < 0 or (position + 1) % COMPRESS_RATIO != 0.
//   2. Gather a window of (1 + OVERLAP)*COMPRESS_RATIO rows from the
//      compressor state cache (the two halves "overlap" / "current"
//      selected by head_offset toggle inside the window).
//   3. Softmax across the window on the score-half; weighted sum on the
//      kv-half -> compressed_kv[HEAD_SIZE] fp32.
//   4. RMSNorm (fp32) with rms_norm_weight.
//   5. UE8M0 block-FP8 quantize of the NoPE region (7 blocks * 64 elems
//      for HEAD_SIZE=512, NOPE=448) -> 448 fp8e4m3 bytes + 7 ue8m0 scale
//      bytes + 1 pad byte (total 8 scale bytes per token).
//   6. Forward GPT-J RoPE on the rope-tail (64 elements) using the
//      *Compressor's* cos_sin_cache (compress_rope_theta=160000),
//      position WINDOW-aligned: (pos // COMPRESS_RATIO) * COMPRESS_RATIO.
//      Store as bf16 immediately after the FP8 nope region.
//
// Cache layout (per-token in the paged K cache):
//   bytes [0,        NOPE_HEAD_DIM)        = FP8 nope    (HEAD_SIZE-64 = 448)
//   bytes [NOPE_HEAD_DIM, NOPE+2*ROPE)     = bf16 rope   (64 * 2 = 128)
//   per-block scale region:
//     bytes [block_size*TOKEN_STRIDE + slot_in_block*SCALE_DIM,
//            ... + SCALE_DIM) = SCALE_DIM ue8m0 bytes
//
//   TOKEN_STRIDE = NOPE_HEAD_DIM + 2*ROPE_HEAD_DIM = 448 + 128 = 576
//   SCALE_DIM    = N_NOPE_BLOCKS + 1 pad = 7 + 1 = 8
//
// Naive design (correctness only):
//   * One CTA per token. grid=(num_tokens,1,1). block=(256,1,1).
//   * fp32 throughout; per-element thread-strided loops.
//   * Smem layout (worst-case): one float[HEAD_SIZE] for compressed_kv,
//     plus one float[NUM_WARPS] for cross-warp reductions, plus
//     one float[W * HEAD_SIZE] gather buffer for the window-rows (used
//     only for the score-half softmax; the kv-half is gathered once more
//     during the weighted sum).
//   * Window size W = (1 + OVERLAP) * COMPRESS_RATIO  (W=8 for ratio=4,
//     W=128 for ratio=128 -- expect ratio=4 in practice).
//   * RoPE uses cos_sin_cache built with compress_rope_theta=160000;
//     positions are window-aligned.
//
// Inputs (TBGraph operator order):
//   [0] state_cache           fp32   [num_blocks, block_size, 2*STATE_WIDTH]
//                                    BROADCAST -- indexed via block_table.
//   [1] token_to_req_indices  int32  [num_tokens]        PARTITION dim 0.
//   [2] positions             int64  [num_tokens]        PARTITION dim 0.
//   [3] slot_mapping          int64  [num_tokens]        PARTITION dim 0.
//   [4] block_table           int32  [num_reqs, max_blocks]  BROADCAST.
//   [5] rms_norm_weight       bf16   [HEAD_SIZE]         BROADCAST.
//   [6] cos_sin_cache         fp32   [max_pos, ROPE_HEAD_DIM]  BROADCAST.
//   [7] kv_slot_mapping       int64  [num_tokens]        PARTITION dim 0.
//
// Outputs (in-place writes; the paged K cache pointer is passed as an
// output so the runtime sets up the TensorDesc, but we treat it as a
// full-buffer destination -- BROADCAST, no per-CTA pre-offset):
//   [0] k_cache               uint8 [num_kv_blocks, kv_block_size, 1,
//                                    TOKEN_STRIDE + SCALE_DIM] -- packed
//                                    FP8 + bf16 RoPE + UE8M0 scales.

namespace kernel {

namespace fused_kv_compress_norm_rope_insert_sparse_attn_v4_detail {

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

} // namespace fused_kv_compress_norm_rope_insert_sparse_attn_v4_detail

// HEAD_SIZE = attention compressor head_dim (512).
// COMPRESS_RATIO = 4 (OVERLAP=true, W=8) or 128 (OVERLAP=false, W=128).
// BLOCK_SIZE = state cache tokens per block.
// MAX_BLOCKS = block_table.shape[1].
// KV_BLOCK_SIZE = k_cache tokens per block.
// ROPE_HEAD_DIM = 64. NOPE_HEAD_DIM = HEAD_SIZE - ROPE_HEAD_DIM = 448.
// QUANT_BLOCK = 64. N_NOPE_BLOCKS = NOPE_HEAD_DIM / QUANT_BLOCK = 7.
// TOKEN_STRIDE = NOPE_HEAD_DIM + 2*ROPE_HEAD_DIM = 576 (bytes/token in
//                k_cache data region).
// SCALE_DIM    = N_NOPE_BLOCKS + 1 pad = 8 (bytes/token in scale region).
template <int HEAD_SIZE, int COMPRESS_RATIO, int BLOCK_SIZE, int MAX_BLOCKS,
          int KV_BLOCK_SIZE, int NUM_THREADS = 256>
__device__ __forceinline__ void
    fused_kv_compress_norm_rope_insert_sparse_attn_v4_sm100_impl(
        void const *state_cache_ptr,         // fp32 [nb, bs, 2*HEAD_SIZE]
        void const *token_to_req_indices_ptr, // int32 [1] this-token req
        void const *positions_ptr,            // int64 [1] this-token pos
        void const *slot_mapping_ptr,         // int64 [1] this-token slot
        void const *block_table_ptr,          // int32 [num_reqs, MAX_BLOCKS]
        void const *rms_norm_weight_ptr,      // bf16 [HEAD_SIZE]
        void const *cos_sin_cache_ptr,        // fp32 [max_pos, ROPE_HEAD_DIM]
        void const *kv_slot_mapping_ptr,      // int64 [1] this-token kv slot
        void *k_cache_ptr,                    // uint8 paged cache (full base)
        float rms_norm_eps) {
  static_assert(HEAD_SIZE == 512,
                "sparse_attn compressor uses head_dim=512 only");
  static_assert(COMPRESS_RATIO == 4 || COMPRESS_RATIO == 128,
                "COMPRESS_RATIO must be 4 (overlap) or 128 (non-overlap)");

  constexpr int ROPE_HEAD_DIM = 64;
  constexpr int NOPE_HEAD_DIM = HEAD_SIZE - ROPE_HEAD_DIM; // 448
  constexpr int QUANT_BLOCK = 64;
  constexpr int N_NOPE_BLOCKS = NOPE_HEAD_DIM / QUANT_BLOCK; // 7
  constexpr int SCALE_DIM = N_NOPE_BLOCKS + 1;               // 8 (with pad)
  constexpr int TOKEN_STRIDE = NOPE_HEAD_DIM + 2 * ROPE_HEAD_DIM; // 576
  constexpr int STATE_WIDTH = 2 * HEAD_SIZE; // coff*head_dim, coff=2 here
  constexpr int OVERLAP = (COMPRESS_RATIO == 4) ? 1 : 0;
  constexpr int W = (1 + OVERLAP) * COMPRESS_RATIO; // 8 or 128
  constexpr int KV_BLOCK_STRIDE =
      KV_BLOCK_SIZE * TOKEN_STRIDE + KV_BLOCK_SIZE * SCALE_DIM;
  constexpr float FP8_MAX = 448.0f;

  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a warp multiple");
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;

  using bf16 = type::bfloat16_t;

  // Early-exit on padded or non-boundary tokens.
  int64_t slot = *static_cast<int64_t const *>(slot_mapping_ptr);
  int64_t position = *static_cast<int64_t const *>(positions_ptr);
  if (slot < 0) {
    return;
  }
  if (((position + 1) % static_cast<int64_t>(COMPRESS_RATIO)) != 0) {
    return;
  }
  int64_t kv_slot = *static_cast<int64_t const *>(kv_slot_mapping_ptr);
  if (kv_slot < 0) {
    return;
  }
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

  // Shared memory layout (extern dynamic smem; MAX_DYNAMIC_SHARED_MEMORY_SIZE
  // on Blackwell is ~220 KiB, more than enough for the layout below):
  //   compressed_kv  : float[HEAD_SIZE]            (HEAD_SIZE = 512 floats)
  //   norm_buf       : float[HEAD_SIZE]            (post-RMSNorm)
  //   reduce         : float[64] scratch           (warp combine + extras)
  //   score_window   : float[W * HEAD_SIZE]        (gathered score rows)
  // For W=8 HEAD=512: total ~20 KiB. We reuse these slabs as scratch
  // during the multi-pass softmax + weighted-sum (see in-line comments).
  extern __shared__ char smem_raw[];
  float *compressed_kv = reinterpret_cast<float *>(smem_raw);
  float *norm_buf = compressed_kv + HEAD_SIZE;
  float *reduce_smem = norm_buf + HEAD_SIZE;
  // score window buffer follows reduce_smem (only used when OVERLAP==1).
  float *score_window = reduce_smem + 64;

  // -----------------------------------------------------------------
  // Step 1+2: gather window, softmax, weighted-sum compress
  // -----------------------------------------------------------------
  // We do this in two passes:
  //   pass A: load the score-half of each window row into score_window
  //           [W, HEAD_SIZE], computing per-element max along W via
  //           on-the-fly reduction is awkward, so we do it as:
  //             - first scan all W rows to find per-element max
  //             - then compute per-element sum(exp(s - max))
  //             - then for each j compute exp(s_w - max) / sum
  //   For naive correctness we go simpler: just load all W rows into
  //   smem (only feasible when W*HEAD_SIZE fits, i.e. OVERLAP=1),
  //   then do per-element softmax in registers.
  //
  // Compute the per-window-row global token index and the
  // (block_no, block_off, head_off) gather indices.
  int64_t window_start = position - (W - 1);

  // First clear compressed_kv.
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    compressed_kv[i] = 0.0f;
  }
  __syncthreads();

  // Stage A: for OVERLAP==1, materialize the entire [W, HEAD_SIZE] score
  // tile in smem (16 KiB). For OVERLAP==0 (ratio=128), the tile is too
  // large; for the naive port we still gate behind the smem cap and
  // assume the caller only hits ratio=4. (Caller already only runs the
  // sparse-attn kernel with whatever ratio the config gates; ratio=128
  // takes the CuteDSL path on NVIDIA -- this naive Triton-equivalent
  // path is exercised primarily for ratio=4 in tests.)
  static_assert(W * HEAD_SIZE * 4 <= 200 * 1024,
                "Naive sparse-attn impl supports COMPRESS_RATIO=4 only "
                "(ratio=128 score_window smem would exceed Blackwell "
                "limits; the ratio=128 path uses the locked CuteDSL "
                "kernel on NVIDIA and is not exercised here).");

  // Compute per-row mask + (block_no, block_off, head_off) and load the
  // score row + kv row.
  // We iterate W rows in a serial outer loop; each row is processed
  // cooperatively by all threads.
  // For the softmax-max + softmax-sum we keep per-thread (max, sum) in
  // smem indexed by HEAD_SIZE dim.
  float *softmax_max = norm_buf; // reuse norm_buf as a scratch tile for max
  float *softmax_sum = compressed_kv; // reuse compressed_kv as sum tile
  // Initialize max=-inf, sum=0 per HEAD_SIZE element.
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    softmax_max[i] = -INFINITY;
    softmax_sum[i] = 0.0f;
  }
  __syncthreads();

  // Pass 1: gather all W rows' score halves into score_window AND compute
  // per-element max across the W rows. Also remember mask for later.
  // We pack (mask, blk_no, blk_off, head_off) lookups inline; for naive
  // simplicity we re-compute them on Pass 2.
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

    // score row base (in state_cache, fp32):
    //   state_cache[blk_no, blk_off,
    //               head_off + STATE_WIDTH : head_off + STATE_WIDTH + HEAD_SIZE]
    // = state_cache + blk_no*BLOCK_SIZE*(2*STATE_WIDTH)
    //              + blk_off*(2*STATE_WIDTH)
    //              + head_off + STATE_WIDTH
    // (last dim has size 2*STATE_WIDTH because score lives in the upper
    //  half of the kv|score concatenation).
    int64_t row_base =
        (static_cast<int64_t>(blk_no) * BLOCK_SIZE + blk_off) *
        static_cast<int64_t>(2 * STATE_WIDTH);
    float const *__restrict__ score_row =
        state_cache + row_base + head_off + STATE_WIDTH;

    float *row_dst = score_window + w * HEAD_SIZE;
    for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
      float v = mask ? score_row[i] : -INFINITY;
      row_dst[i] = v;
      // running per-element max
      float cur = softmax_max[i];
      softmax_max[i] = fmaxf(cur, v);
    }
    __syncthreads();
  }

  // Pass 2: compute per-element sum(exp(s - max)).
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float m = softmax_max[i];
    float s = 0.0f;
#pragma unroll
    for (int w = 0; w < W; ++w) {
      float v = score_window[w * HEAD_SIZE + i];
      s += __expf(v - m);
    }
    softmax_sum[i] = s;
  }
  __syncthreads();

  // Pass 3: weighted sum on kv-half + softmax-normalize.
  // We accumulate compressed_kv[j] = sum_w (kv_w[j] * exp(score_w[j] - m[j]) /
  // sum[j]). We can't reuse compressed_kv to hold sum *and* the output, so
  // commit the sum to a local tile in registers per index.
  // For naive simplicity: re-walk W rows, compute weighted kv accumulator
  // into a fresh smem buffer. Use score_window memory as the kv weighted
  // accumulator base (overwriting it; we no longer need the raw scores).
  float *accum = score_window; // [HEAD_SIZE] reused
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
    float const *__restrict__ kv_row =
        state_cache + row_base + head_off;
    float const *__restrict__ score_row =
        state_cache + row_base + head_off + STATE_WIDTH;

    for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
      float kv_v = mask ? kv_row[i] : 0.0f;
      float s_v = mask ? score_row[i] : -INFINITY;
      float wgt = __expf(s_v - softmax_max[i]) / softmax_sum[i];
      accum[i] += kv_v * wgt;
    }
    __syncthreads();
  }

  // -----------------------------------------------------------------
  // Step 3: RMSNorm in fp32.
  // -----------------------------------------------------------------
  // compressed_kv & softmax_sum tile have served their purpose. Reuse
  // them for the norm pipeline.
  // 1. Compute sum of squares of accum -> variance.
  float partial = 0.0f;
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float v = accum[i];
    partial += v * v;
  }
  float sumsq = fused_kv_compress_norm_rope_insert_sparse_attn_v4_detail::
      warp_block_reduce_sum<NUM_THREADS>(partial, reduce_smem);
  float inv_rms = rsqrtf(sumsq / static_cast<float>(HEAD_SIZE) + rms_norm_eps);

  // Store normed * weight into norm_buf (overwrites scratch tile).
  for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float w = static_cast<float>(rms_w[i]);
    norm_buf[i] = accum[i] * inv_rms * w;
  }
  __syncthreads();

  // -----------------------------------------------------------------
  // Cache pointer setup.
  // -----------------------------------------------------------------
  // Token's data slot in the paged K cache. Within a block, the per-
  // token data region is contiguous: blk*KV_BLOCK_STRIDE + slot_in_blk
  // * TOKEN_STRIDE. The scale region for the SAME block follows after
  // all token data: blk*KV_BLOCK_STRIDE + KV_BLOCK_SIZE*TOKEN_STRIDE +
  // slot_in_blk*SCALE_DIM.
  int64_t kv_blk = kv_slot / static_cast<int64_t>(KV_BLOCK_SIZE);
  int kv_off =
      static_cast<int>(kv_slot % static_cast<int64_t>(KV_BLOCK_SIZE));
  uint8_t *blk_base = k_cache + kv_blk * static_cast<int64_t>(KV_BLOCK_STRIDE);
  uint8_t *token_data = blk_base + static_cast<int64_t>(kv_off) * TOKEN_STRIDE;
  uint8_t *token_scale = blk_base + KV_BLOCK_SIZE * TOKEN_STRIDE +
                         kv_off * SCALE_DIM;

  // -----------------------------------------------------------------
  // Step 4: UE8M0 block-FP8 quant of NoPE [0, NOPE_HEAD_DIM).
  // -----------------------------------------------------------------
  // bf16 roundtrip on the nope half to match the reference.
  // Per-block (size 64) absmax via two-stage block reduce.
  __nv_fp8_e4m3 *fp8_out = reinterpret_cast<__nv_fp8_e4m3 *>(token_data);

  // Cast nope -> bf16 -> fp32 IN-PLACE in norm_buf for the nope region.
  for (int i = threadIdx.x; i < NOPE_HEAD_DIM; i += NUM_THREADS) {
    bf16 v_bf16 = bf16(norm_buf[i]);
    norm_buf[i] = static_cast<float>(v_bf16);
  }
  __syncthreads();

  // 7 blocks. Process each block sequentially (naive).
  // Each block: absmax across QUANT_BLOCK=64 elements.
  for (int b = 0; b < N_NOPE_BLOCKS; ++b) {
    float partial_max = 0.0f;
    int base = b * QUANT_BLOCK;
    for (int i = threadIdx.x; i < QUANT_BLOCK; i += NUM_THREADS) {
      partial_max = fmaxf(partial_max, fabsf(norm_buf[base + i]));
    }
    float amax =
        fused_kv_compress_norm_rope_insert_sparse_attn_v4_detail::
            warp_block_reduce_max<NUM_THREADS>(partial_max, reduce_smem);
    amax = fmaxf(amax, 1e-4f);
    // exponent = ceil(log2(amax / FP8_MAX))
    float exponent = ceilf(__log2f(amax / FP8_MAX));
    // clamp into byte range; byte = clamp(exponent + 127, 0, 255).
    int byte = static_cast<int>(exponent) + 127;
    if (byte < 0) byte = 0;
    if (byte > 255) byte = 255;
    float inv_scale = exp2f(-exponent);

    if (threadIdx.x == 0) {
      token_scale[b] = static_cast<uint8_t>(byte);
    }

    for (int i = threadIdx.x; i < QUANT_BLOCK; i += NUM_THREADS) {
      float v = norm_buf[base + i] * inv_scale;
      v = fminf(fmaxf(v, -FP8_MAX), FP8_MAX);
      fp8_out[base + i] = __nv_fp8_e4m3(v);
    }
    __syncthreads();
  }

  // Pad byte 7 (per spec: zero pad).
  if (threadIdx.x == 0) {
    token_scale[N_NOPE_BLOCKS] = 0;
  }

  // -----------------------------------------------------------------
  // Step 5: Forward GPT-J RoPE on rope tail [NOPE_HEAD_DIM, HEAD_SIZE).
  // -----------------------------------------------------------------
  // Window-aligned position for the cos/sin lookup.
  // pairs view: rope = norm_buf[NOPE..HEAD_SIZE], reshape [ROPE_HEAD_DIM/2, 2]
  // i.e. (even, odd) pairs. Apply rotation:
  //   new_even = even*cos - odd*sin
  //   new_odd  = odd *cos + even*sin
  // cos and sin come from cos_sin_cache[compressed_pos, :ROPE/2]
  // and cos_sin_cache[compressed_pos, ROPE/2:].
  int64_t compressed_pos =
      (position / static_cast<int64_t>(COMPRESS_RATIO)) *
      static_cast<int64_t>(COMPRESS_RATIO);
  float const *__restrict__ cs_row =
      cos_sin_cache + compressed_pos * ROPE_HEAD_DIM;
  constexpr int ROPE_PAIRS = ROPE_HEAD_DIM / 2; // 32

  // bf16 rope output target: bytes [NOPE_HEAD_DIM, NOPE+128) of token_data.
  bf16 *rope_out = reinterpret_cast<bf16 *>(token_data + NOPE_HEAD_DIM);

  for (int p = threadIdx.x; p < ROPE_PAIRS; p += NUM_THREADS) {
    float even = norm_buf[NOPE_HEAD_DIM + 2 * p + 0];
    float odd  = norm_buf[NOPE_HEAD_DIM + 2 * p + 1];
    float cos = cs_row[p];
    float sin = cs_row[ROPE_PAIRS + p];
    float new_even = even * cos - odd * sin;
    float new_odd  = odd  * cos + even * sin;
    rope_out[2 * p + 0] = bf16(new_even);
    rope_out[2 * p + 1] = bf16(new_odd);
  }
}

} // namespace kernel
