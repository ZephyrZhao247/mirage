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
// compressor_compress_sm100.cuh — DeepSeek V4-Flash Compressor *compress* step
// (Wave-2 sub-batch C2). Sibling of ``compressor_state_update_sm100``.
//
// Math for a single boundary token (position p, batch b, R = COMPRESS_RATIO,
// W = (1 + OVERLAP) * R the gather window width — v1 = COFF * R):
//
//   state = state_cache[b, :W, :]      shape (W, 2 * HEAD_DIM) bf16
//   kv_part    = state[:, :HEAD_DIM]                        (fp32 promoted)
//   score_part = state[:, HEAD_DIM:2*HEAD_DIM] + ape        (fp32; ape bf16)
//   w     = softmax(score_part, dim=0)                      # over the window
//   pooled[d]   = sum_w(kv_part[w, d] * w[w, d])            # gated pool
//   rrms        = rsqrt(mean(pooled^2) + eps)
//   normed[d]   = pooled[d] * rrms * norm_weight[d]
//   # GPT-J interleaved RoPE on the last ROPE_DIM dims, at the boundary-
//   # aligned position pos_compress = positions[token] / COMPRESS_RATIO:
//   pair (x, y) -> (x * cos - y * sin, y * cos + x * sin)
//   # Per-block FP8e4m3 quant on the first NOPE_DIM = HEAD_DIM - ROPE_DIM dims:
//   scale[blk] = max(absmax_block, eps) / 448.0     # plain fp32 scale (v1)
//   out_fp8[i] = clamp(normed_nope[i] / scale[blk(i)], +/- 448).to_fp8
//   # Cache slot byte layout, per compressed token (slot stride =
//   # NOPE_DIM + 2 * ROPE_DIM + 4 * NUM_SCALE_BLOCKS):
//   [0,                                NOPE_DIM):                 fp8 nope
//   [NOPE_DIM,                NOPE_DIM + 2*ROPE_DIM):              bf16 rope
//   [NOPE_DIM + 2*ROPE_DIM,
//    NOPE_DIM + 2*ROPE_DIM + 4*NUM_SCALE_BLOCKS):                  fp32 scales
//
// The kernel itself does NOT gate on `(p+1) % R == 0`; the dispatching
// Python layer is responsible for issuing this task only on compression
// boundaries (the cap on grid_dim already reflects that). On entry the
// CTA owns one boundary token; ``task_metadata.token_offset`` indexes the
// boundary list contiguously (batch == token_offset in v1's single-batch
// test path).
//
// blockIdx-agnostic per MPK convention; all routing comes from
// ``task_desc->task_metadata.token_offset``.
// =============================================================================

#pragma once
#include "tasks/common/common_header.cuh"
#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace kernel {

namespace compressor_compress_detail {

// Block reduction primitive: reduce ``local`` across the first NUM_THREADS
// lanes via warp shuffles + a small smem cross-warp combine. Returns the
// reduced value broadcast to every participating thread (via smem slot 0).
template <int NUM_THREADS, typename Combine>
__device__ __forceinline__ float
block_reduce(float local, float *reduce_smem, Combine combine) {
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a multiple of warp size");
#pragma unroll
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    float other = __shfl_xor_sync(0xffffffff, local, offset);
    local = combine(local, other);
  }
  int lane = threadIdx.x % NUM_THREADS_PER_WARP;
  int warp = threadIdx.x / NUM_THREADS_PER_WARP;
  if (lane == 0) {
    reduce_smem[warp] = local;
  }
  __syncthreads();
  if (warp == 0) {
    float v = (threadIdx.x < NUM_WARPS) ? reduce_smem[threadIdx.x]
                                        : reduce_smem[0];
#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
      float other = __shfl_xor_sync(0xffffffff, v, offset);
      v = combine(v, other);
    }
    if (threadIdx.x == 0) {
      reduce_smem[0] = v;
    }
  }
  __syncthreads();
  return reduce_smem[0];
}

struct AddOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return a + b;
  }
};
struct MaxOp {
  __device__ __forceinline__ float operator()(float a, float b) const {
    return fmaxf(a, b);
  }
};

} // namespace compressor_compress_detail

// Template parameters:
//   HEAD_DIM        : full per-head width (kv part width; e.g. 512 in V4).
//   ROPE_DIM        : trailing rope width (e.g. 64 in V4). Must be even.
//   COMPRESS_RATIO  : R — softmax window stride (e.g. 4 or 128).
//   OVERLAP         : 0 or 1. v1 currently uses the full COFF * R window
//                      width directly (the Python side sizes state_cache's
//                      ``window_size`` dim accordingly).
//   BLOCK_SIZE      : FP8 quant block width over the NOPE region. Must
//                      divide NOPE_DIM = HEAD_DIM - ROPE_DIM.
//   NUM_THREADS     : threads per CTA used by this task. Lanes
//                      ``threadIdx.x >= NUM_THREADS`` early-out so the
//                      runtime can host the task on a wider worker block.
template <int HEAD_DIM,
          int ROPE_DIM,
          int COMPRESS_RATIO,
          bool OVERLAP,
          int BLOCK_SIZE,
          int NUM_THREADS>
__device__ __forceinline__ void compressor_compress_task_impl(
    void const *__restrict__ state_cache_ptr,    // bf16
    void const *__restrict__ ape_ptr,            // bf16 [R, HEAD_DIM]
    void const *__restrict__ cos_sin_cache_ptr,  // bf16 [max_pos, ROPE_DIM]
    void const *__restrict__ norm_weight_ptr,    // bf16 [HEAD_DIM]
    void const *__restrict__ positions_ptr,      // int32 [T]
    void *__restrict__ kv_cache_ptr,             // uint8 paged, BYTES_PER_SLOT/slot
    int token_offset,
    int num_tokens_per_task,
    int num_tokens_total,
    int batch_offset,        // ``batch`` index for state_cache addressing
    int kv_cache_token_stride_bytes,
    float eps) {
  // Gate inactive lanes so the worker block's WORKER_NUM_THREADS hosts the
  // task without wasted work. The runtime issues __syncthreads() around
  // _execute_task(); intra-task barriers use the active lanes only.
  if (threadIdx.x >= NUM_THREADS) {
    return;
  }

  static_assert(ROPE_DIM % 2 == 0, "ROPE_DIM must be even (GPT-J pairs)");
  static_assert(ROPE_DIM <= HEAD_DIM, "ROPE_DIM cannot exceed HEAD_DIM");
  static_assert((HEAD_DIM - ROPE_DIM) % BLOCK_SIZE == 0,
                "BLOCK_SIZE must divide NOPE_DIM = HEAD_DIM - ROPE_DIM");
  static_assert(COMPRESS_RATIO > 0, "COMPRESS_RATIO must be positive");
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a multiple of warp size");

  constexpr int NOPE_DIM = HEAD_DIM - ROPE_DIM;
  constexpr int HALF_ROPE = ROPE_DIM / 2;
  constexpr int NUM_SCALE_BLOCKS = NOPE_DIM / BLOCK_SIZE;
  // Gather window. The v1 path treats overlap as a layout-only flag — the
  // window width is COFF * R, where COFF = (OVERLAP ? 2 : 1). The Python
  // side sizes ``state_cache.dim(1)`` to COFF * R so a single softmax over
  // the full leading window axis is the correct semantics.
  constexpr int WINDOW = (OVERLAP ? 2 : 1) * COMPRESS_RATIO;
  // Per-slot byte stride. Used to compute the destination pointer for the
  // current compressed-token's slot when ``kv_cache_token_stride_bytes`` is
  // zero (i.e. the caller hands us a plain ``[T_compressed, BYTES]`` slab
  // rather than a paged layout).
  constexpr int DEFAULT_SLOT_BYTES =
      NOPE_DIM + 2 * ROPE_DIM + 4 * NUM_SCALE_BLOCKS;

  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
  // Shared workspace:
  //   * ``pooled[HEAD_DIM]``  fp32  - gated-softmax sum across the window.
  //   * ``smax[WINDOW]``      fp32  - per-window-row softmax max (one
  //                                    independent softmax per (window, d)?
  //                                    No — see math: softmax is along the
  //                                    *window* axis with one score per
  //                                    (window, dim) entry; the partition
  //                                    function is per-dim. We use a
  //                                    dim-major two-pass softmax.
  //   * ``reduce_smem[NUM_WARPS]``  fp32  - cross-warp reductions.
  //
  // We size the workspace as the union of the three needs. For the
  // softmax pass we keep the per-(window, dim) score in fp32 registers
  // distributed across threads (each thread owns a contiguous stride of
  // dims); the per-dim max + sum reductions are cross-window inside one
  // thread, so no shared memory is required for them.
  __shared__ float pooled_smem[HEAD_DIM];
  __shared__ float reduce_smem[NUM_WARPS > 1 ? NUM_WARPS : 1];

  // Typed pointers.
  __nv_bfloat16 const *__restrict__ state_cache =
      static_cast<__nv_bfloat16 const *>(state_cache_ptr);
  __nv_bfloat16 const *__restrict__ ape =
      static_cast<__nv_bfloat16 const *>(ape_ptr);
  __nv_bfloat16 const *__restrict__ cos_sin =
      static_cast<__nv_bfloat16 const *>(cos_sin_cache_ptr);
  __nv_bfloat16 const *__restrict__ norm_w =
      static_cast<__nv_bfloat16 const *>(norm_weight_ptr);
  int const *__restrict__ positions =
      static_cast<int const *>(positions_ptr);
  uint8_t *__restrict__ kv_cache_bytes =
      static_cast<uint8_t *>(kv_cache_ptr);

  int const slot_bytes = (kv_cache_token_stride_bytes > 0)
                             ? kv_cache_token_stride_bytes
                             : DEFAULT_SLOT_BYTES;

  for (int local = 0; local < num_tokens_per_task; ++local) {
    int const t = token_offset + local;
    if (t >= num_tokens_total) {
      break;
    }
    int const b = batch_offset + t;
    int const pos = positions[t];
    int const pos_compress = pos / COMPRESS_RATIO;

    // ------------------------------------------------------------------
    // Phase 1: gated-softmax pool.
    //   pooled[d] = sum_w(kv[w, d] * softmax_w(score[w, d] + ape[w, d]))
    // Implemented as a two-pass per-(window, d) softmax over the window
    // axis. The softmax partition is independent per d (no cross-d
    // coupling). We tile threads over d (column-major); each thread owns
    // a stride of dims and runs its own per-d max / sum sequentially over
    // the WINDOW rows.
    // ------------------------------------------------------------------
    size_t const state_row_stride = static_cast<size_t>(2) * HEAD_DIM;
    size_t const state_base_off =
        static_cast<size_t>(b) * WINDOW * state_row_stride;

    for (int d = threadIdx.x; d < HEAD_DIM; d += NUM_THREADS) {
      // Pass A: per-d max.
      // Use -FLT_MAX as the softmax-max initial sentinel; we do not need
      // -inf here because the per-d max is computed over a small finite
      // window of fp32 scores.
      float dmax = -3.4028235e38f;
#pragma unroll 1
      for (int w = 0; w < WINDOW; ++w) {
        size_t const row_off = state_base_off + (size_t)w * state_row_stride;
        float s = __bfloat162float(state_cache[row_off + HEAD_DIM + d]);
        // ape is broadcast over the window axis modulo COMPRESS_RATIO; the
        // window may be COFF*R wide so we wrap with %.
        float a = __bfloat162float(ape[(w % COMPRESS_RATIO) * HEAD_DIM + d]);
        float sa = s + a;
        if (sa > dmax) {
          dmax = sa;
        }
      }
      // Pass B: per-d sum-of-exp.
      float dsum = 0.f;
#pragma unroll 1
      for (int w = 0; w < WINDOW; ++w) {
        size_t const row_off = state_base_off + (size_t)w * state_row_stride;
        float s = __bfloat162float(state_cache[row_off + HEAD_DIM + d]);
        float a = __bfloat162float(ape[(w % COMPRESS_RATIO) * HEAD_DIM + d]);
        dsum += expf((s + a) - dmax);
      }
      float inv_dsum = 1.f / dsum;
      // Pass C: weighted sum into pooled[d].
      float pooled = 0.f;
#pragma unroll 1
      for (int w = 0; w < WINDOW; ++w) {
        size_t const row_off = state_base_off + (size_t)w * state_row_stride;
        float kv = __bfloat162float(state_cache[row_off + d]);
        float s = __bfloat162float(state_cache[row_off + HEAD_DIM + d]);
        float a = __bfloat162float(ape[(w % COMPRESS_RATIO) * HEAD_DIM + d]);
        float w_val = expf((s + a) - dmax) * inv_dsum;
        pooled += kv * w_val;
      }
      pooled_smem[d] = pooled;
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 2: RMSNorm in fp32 over HEAD_DIM.
    //   var = mean(pooled^2)
    //   normed[d] = pooled[d] * rsqrt(var + eps) * norm_weight[d]
    // ------------------------------------------------------------------
    float local_sqsum = 0.f;
    for (int d = threadIdx.x; d < HEAD_DIM; d += NUM_THREADS) {
      float v = pooled_smem[d];
      local_sqsum += v * v;
    }
    float ssum = compressor_compress_detail::block_reduce<NUM_THREADS>(
        local_sqsum, reduce_smem, compressor_compress_detail::AddOp{});
    float rrms = rsqrtf(ssum / static_cast<float>(HEAD_DIM) + eps);
    for (int d = threadIdx.x; d < HEAD_DIM; d += NUM_THREADS) {
      float v = pooled_smem[d];
      float w = __bfloat162float(norm_w[d]);
      pooled_smem[d] = v * rrms * w;
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 3: GPT-J interleaved RoPE on the trailing ROPE_DIM dims.
    //   pair (x, y) at (2k, 2k+1) ->
    //       new_x = x * cos[k] - y * sin[k]
    //       new_y = y * cos[k] + x * sin[k]
    // Reads cos/sin from cos_sin_cache at row ``pos_compress`` with cos
    // in [:HALF_ROPE) and sin in [HALF_ROPE, ROPE_DIM).
    // ------------------------------------------------------------------
    {
      __nv_bfloat16 const *__restrict__ cos_base =
          cos_sin + (size_t)pos_compress * ROPE_DIM;
      __nv_bfloat16 const *__restrict__ sin_base = cos_base + HALF_ROPE;
      for (int k = threadIdx.x; k < HALF_ROPE; k += NUM_THREADS) {
        int const x_off = NOPE_DIM + 2 * k;
        int const y_off = x_off + 1;
        float x = pooled_smem[x_off];
        float y = pooled_smem[y_off];
        float c = __bfloat162float(cos_base[k]);
        float s = __bfloat162float(sin_base[k]);
        pooled_smem[x_off] = x * c - y * s;
        pooled_smem[y_off] = y * c + x * s;
      }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Phase 4: per-block FP8 quant over NOPE_DIM + bf16 RoPE store + fp32
    // scale store. Per-block absmax via __shfl_xor_sync over the lanes
    // owning that block; here threads stride over (block, in-block-elem)
    // jointly, so for each block we re-scan the in-block lanes and pick
    // the max, then write fp8 bytes. With NUM_THREADS divisible by
    // BLOCK_SIZE * NUM_SCALE_BLOCKS this is contiguous and simple. The
    // generic path below uses a per-block loop where each lane owns
    // ELEMS_PER_THREAD elements of the block and we reduce the max via
    // warp shuffle inside the (assumed) single-warp block; if BLOCK_SIZE
    // > 32 we fall back to a smem reduction across the active block-lanes.
    // ------------------------------------------------------------------
    constexpr float kFp8Max = 448.0f;
    constexpr float kEps = 1e-12f;
    constexpr int ELEMS_PER_THREAD_BLOCK =
        (BLOCK_SIZE + NUM_THREADS - 1) / NUM_THREADS;
    constexpr int ACTIVE_LANES_PER_BLOCK =
        BLOCK_SIZE < NUM_THREADS ? BLOCK_SIZE : NUM_THREADS;

    // Process one quant block at a time. For each block we re-load its
    // BLOCK_SIZE elements from pooled_smem into per-thread fp32 regs,
    // compute absmax via a block-reduce over NUM_THREADS lanes (lanes
    // outside the block contribute 0), derive a plain-fp32 scale, then
    // quantize and write.
    uint8_t *__restrict__ slot_base =
        kv_cache_bytes + (size_t)pos_compress * slot_bytes;
    // Note: in v1 the slot index in the output paged cache is the
    // ``pos_compress`` value itself (single batch, contiguous slab). A
    // future revision can plumb a slot_mapping array through the same
    // pattern as ``inv_rope_fp8_quant_o`` for true paged writes.
    __nv_fp8_e4m3 *__restrict__ fp8_out =
        reinterpret_cast<__nv_fp8_e4m3 *>(slot_base);
    __nv_bfloat16 *__restrict__ rope_out =
        reinterpret_cast<__nv_bfloat16 *>(slot_base + NOPE_DIM);
    float *__restrict__ scale_out = reinterpret_cast<float *>(
        slot_base + NOPE_DIM + 2 * ROPE_DIM);

    for (int blk = 0; blk < NUM_SCALE_BLOCKS; ++blk) {
      int const block_base = blk * BLOCK_SIZE;
      float regs[ELEMS_PER_THREAD_BLOCK];
      float local_max = kEps;
#pragma unroll
      for (int i = 0; i < ELEMS_PER_THREAD_BLOCK; ++i) {
        int const lane_off = i * NUM_THREADS + (int)threadIdx.x;
        if (lane_off < BLOCK_SIZE) {
          float v = pooled_smem[block_base + lane_off];
          regs[i] = v;
          float av = fabsf(v);
          if (av > local_max) {
            local_max = av;
          }
        } else {
          regs[i] = 0.f;
        }
      }
      // Block reduce max across NUM_THREADS lanes. Out-of-block lanes
      // contributed kEps so they do not perturb the result.
      float block_max =
          compressor_compress_detail::block_reduce<NUM_THREADS>(
              local_max, reduce_smem,
              compressor_compress_detail::MaxOp{});
      float scale = fmaxf(block_max, kEps) / kFp8Max;
      float inv_scale = 1.f / scale;
      if (threadIdx.x == 0) {
        scale_out[blk] = scale;
      }
#pragma unroll
      for (int i = 0; i < ELEMS_PER_THREAD_BLOCK; ++i) {
        int const lane_off = i * NUM_THREADS + (int)threadIdx.x;
        if (lane_off < BLOCK_SIZE) {
          float q = regs[i] * inv_scale;
          q = fminf(fmaxf(q, -kFp8Max), kFp8Max);
          fp8_out[block_base + lane_off] = __nv_fp8_e4m3(q);
        }
      }
      __syncthreads();
      (void)ACTIVE_LANES_PER_BLOCK;
    }

    // RoPE bf16 store: pack rotated tail back as bf16.
    for (int d = threadIdx.x; d < ROPE_DIM; d += NUM_THREADS) {
      rope_out[d] = __float2bfloat16(pooled_smem[NOPE_DIM + d]);
    }
    __syncthreads();
  }
}

} // namespace kernel
