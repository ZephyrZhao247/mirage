/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// =============================================================================
// inv_rope_fp8_quant_o_sm100.cuh — DeepSeek V4-Flash MLA post-attention fused
// inverse-RoPE + per-block FP8e4m3fn quantization on the attention output.
//
// For each token t and each head h:
//   1. Inverse RoPE (GPT-J interleaved) on the last `rope_dim` of
//      o[t, h, :head_dim]. Pairs are (o[2k], o[2k+1]).
//
//        new o[2k]   = o[2k]   * cos[k] + o[2k+1] * sin[k]
//        new o[2k+1] = o[2k+1] * cos[k] - o[2k]   * sin[k]
//
//      i.e. conjugate-of-forward (forward RoPE flips the sign of sin in
//      both formulas; inverse uses cos for the same-channel coeff and
//      cross-couples with +sin / -sin respectively). See
//      `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:232-244`
//      `apply_rotary_emb(..., inverse=True)`.
//
//   2. Per-block FP8 quantization on the full head_dim row, grouped into
//      `head_dim / block_size` blocks of `block_size` (=128) elements:
//
//        absmax = max(|o[block]|)
//        scale  = max(absmax, eps) / fp8_max   (plain fp32 scale, v1)
//        o_fp8  = clamp(o / scale, [-fp8_max, fp8_max]) cast to fp8_e4m3
//
//   Output layout:
//     o_fp8  [T, H, head_dim]              fp8_e4m3fn (stored as uint8)
//     o_scale[T, H, head_dim / block_size] fp32
//
// Layout notes:
//   - The `cos_sin_cache` stores the two halves concatenated along the
//     last dim, matching the vLLM Triton convention: the first
//     `rope_dim / 2` entries are cos, the next `rope_dim / 2` are sin.
//     `cs_idx = rope_local >> 1` indexes the half-table for the pair k.
//
// One CTA per token: every CTA covers all H heads of one token. The CTA
// derives its `t` from `task_desc->task_metadata.token_offset`
// (blockIdx-agnostic, per MPK convention).
// =============================================================================

#pragma once
#include "tasks/common/common_header.cuh"
#include <cstdint>
#include <cuda_fp8.h>

namespace kernel {

template <int NUM_HEADS,
          int HEAD_DIM,
          int ROPE_DIM,
          int BLOCK_SIZE,
          int NUM_THREADS,
          typename IN_T,
          typename CACHE_T>
__device__ __forceinline__ void inv_rope_fp8_quant_o_task_impl(
    void const *__restrict__ o_ptr,           // bf16 [T, H, HEAD_DIM]
    void const *__restrict__ cos_sin_cache_ptr, // bf16 [max_pos, ROPE_DIM]
    void const *__restrict__ positions_ptr,   // int32 [T]
    void *__restrict__ o_fp8_ptr,             // fp8 (uint8) [T, H, HEAD_DIM]
    void *__restrict__ o_scale_ptr,           // fp32 [T, H, HEAD_DIM / BLOCK_SIZE]
    int token_offset,
    int num_tokens_per_task,
    int num_tokens_total) {
  // Gate inactive threads. The MPK worker launches WORKER_NUM_THREADS
  // threads regardless of this kernel's needs; we operate with only the
  // first NUM_THREADS lanes. The runtime issues __syncthreads() around
  // _execute_task(), so the returning lanes do not need to participate in
  // intra-task barriers (none are used here).
  if (threadIdx.x >= NUM_THREADS) {
    return;
  }

  static_assert(HEAD_DIM % BLOCK_SIZE == 0,
                "HEAD_DIM must be a multiple of BLOCK_SIZE for per-block quant");
  static_assert(ROPE_DIM % 2 == 0,
                "ROPE_DIM must be even (GPT-J interleaved pairs)");
  static_assert(ROPE_DIM <= HEAD_DIM,
                "ROPE_DIM cannot exceed HEAD_DIM");
  static_assert((HEAD_DIM - ROPE_DIM) % 2 == 0,
                "NOPE_DIM = HEAD_DIM - ROPE_DIM must be even so that "
                "partner = off ^ 1 keeps rope-tail elements within the "
                "rope tail (GPT-J pair invariant)");

  constexpr int NUM_BLOCKS_PER_ROW = HEAD_DIM / BLOCK_SIZE;
  constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
  constexpr int HALF_ROPE = ROPE_DIM / 2;
  constexpr float kEps = 1e-12f;
  constexpr float kFp8Max = 448.0f;  // FP8 E4M3 max representable

  // Typed pointers.
  IN_T const *__restrict__ o_in = static_cast<IN_T const *>(o_ptr);
  CACHE_T const *__restrict__ cs_cache =
      static_cast<CACHE_T const *>(cos_sin_cache_ptr);
  int const *__restrict__ positions = static_cast<int const *>(positions_ptr);
  __nv_fp8_e4m3 *__restrict__ o_fp8 =
      static_cast<__nv_fp8_e4m3 *>(o_fp8_ptr);
  float *__restrict__ o_scale = static_cast<float *>(o_scale_ptr);

  // Iterate over the contiguous slice of tokens owned by this CTA. In v1
  // num_tokens_per_task == 1 (one CTA per token); the loop is written
  // generically so larger slices work unchanged.
  for (int local = 0; local < num_tokens_per_task; ++local) {
    int const t = token_offset + local;
    if (t >= num_tokens_total) {
      break;
    }

    int const pos = positions[t];
    // Per-token cos / sin base pointers. Layout: the first HALF_ROPE
    // entries of cs_cache[pos, :] are cos, the next HALF_ROPE are sin.
    CACHE_T const *__restrict__ cos_base = cs_cache + pos * ROPE_DIM;
    CACHE_T const *__restrict__ sin_base = cos_base + HALF_ROPE;

    // ----------- Process each head independently. -----------
    for (int h = 0; h < NUM_HEADS; ++h) {
      int const row_base = (t * NUM_HEADS + h) * HEAD_DIM;

      // Step 1: Inverse-RoPE the rope tail in-place into a thread-local
      // register-resident copy (we do not write back to global `o_in`).
      // We then re-load the nope chunk during quantization.
      //
      // Strategy: thread-local rotation of each pair, where each thread
      // owns a contiguous stride of pairs in the rope tail. Then per-
      // block (BLOCK_SIZE) quantize the entire HEAD_DIM, where threads
      // collaborate via warp shuffles per block.
      //
      // Per block of BLOCK_SIZE elements, each thread handles
      // BLOCK_SIZE / NUM_THREADS elements (or NUM_THREADS / BLOCK_SIZE
      // threads share one element — we assume NUM_THREADS divides
      // BLOCK_SIZE for v1).

      // We process one quant block at a time so we don't need shmem.
      for (int blk = 0; blk < NUM_BLOCKS_PER_ROW; ++blk) {
        int const block_base = row_base + blk * BLOCK_SIZE;
        int const block_offset_in_row = blk * BLOCK_SIZE;

        // Phase A: load (and possibly rotate) block elements into per-thread
        // fp32 registers.
        constexpr int ELEMS_PER_THREAD =
            (BLOCK_SIZE + NUM_THREADS - 1) / NUM_THREADS;
        float regs[ELEMS_PER_THREAD];
        float local_max = kEps;

#pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
          int const lane_offset = i * NUM_THREADS + (int)threadIdx.x;
          if (lane_offset >= BLOCK_SIZE) {
            regs[i] = 0.0f;
            continue;
          }
          int const off_in_row = block_offset_in_row + lane_offset;
          float v = static_cast<float>(o_in[block_base + lane_offset]);

          // Check whether this element is in the rope tail. The rope
          // tail occupies the last ROPE_DIM elements of the row, i.e.
          // off_in_row >= NOPE_DIM. If so, rotate using the partner.
          if (off_in_row >= NOPE_DIM) {
            int const rope_local = off_in_row - NOPE_DIM;  // [0, ROPE_DIM)
            int const cs_idx = rope_local >> 1;             // pair index
            // Partner offset within the row (toggle bit 0 of rope_local
            // → toggle bit 0 of the in-row offset since NOPE_DIM is even).
            int const partner_off = off_in_row ^ 1;
            float partner =
                static_cast<float>(o_in[row_base + partner_off]);
            float cos_v = static_cast<float>(cos_base[cs_idx]);
            float sin_v = static_cast<float>(sin_base[cs_idx]);
            // Inverse RoPE (conjugate of forward GPT-J):
            //   even rope_local (i.e. the "x" of the pair):
            //     new = x * cos + partner * sin
            //   odd rope_local (i.e. the "y" of the pair):
            //     new = y * cos - partner * sin
            bool is_even = ((rope_local & 1) == 0);
            float rotated = is_even ? (v * cos_v + partner * sin_v)
                                    : (v * cos_v - partner * sin_v);
            v = rotated;
          }
          regs[i] = v;
          float absv = fabsf(v);
          if (absv > local_max) {
            local_max = absv;
          }
        }

        // Phase B: cross-thread max-reduce over the active lanes of this
        // block. With NUM_THREADS lanes participating, a single warp
        // shuffle pass over the full warp width works as long as
        // NUM_THREADS <= 32; otherwise fall back to shmem reduction.
        float block_max = local_max;
#pragma unroll
        for (int offset = (NUM_THREADS >> 1); offset > 0; offset >>= 1) {
          float other = __shfl_xor_sync(0xffffffff, block_max, offset,
                                        NUM_THREADS);
          block_max = fmaxf(block_max, other);
        }

        // Phase C: derive plain-fp32 scale.
        float scale = fmaxf(block_max, kEps) / kFp8Max;
        float inv_scale = 1.0f / scale;

        // Lane 0 writes the scale for this block.
        if (threadIdx.x == 0) {
          int const scale_idx =
              (t * NUM_HEADS + h) * NUM_BLOCKS_PER_ROW + blk;
          o_scale[scale_idx] = scale;
        }

        // Phase D: quantize and store.
#pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
          int const lane_offset = i * NUM_THREADS + (int)threadIdx.x;
          if (lane_offset >= BLOCK_SIZE) {
            continue;
          }
          float q = regs[i] * inv_scale;
          q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
          o_fp8[block_base + lane_offset] = __nv_fp8_e4m3(q);
        }
      }
    }
  }
}

} // namespace kernel
