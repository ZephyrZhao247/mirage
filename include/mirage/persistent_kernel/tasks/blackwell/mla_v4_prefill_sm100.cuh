/* Copyright 2025 CMU
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
// mla_v4_prefill_sm100.cuh — DeepSeek V4-Flash MLA prefill over a gathered
// KV workspace (single-row K==V, MLA-style).
//
// Math (per Q tile, FlashAttn online softmax, FP32 accumulators):
//
//   q             : [T_q, NUM_HEADS, HEAD_DIM]   bf16, projected+RoPE'd+norm'd
//   gathered_kv   : [T_kv,         HEAD_DIM]     bf16, MLA single-row K==V
//   attn_sink     : [NUM_HEADS]                  fp32, additive logit sink slot
//
//   logits[t, h, k] = (q[t, h] . gathered_kv[k]) * softmax_scale
//   weights[t, h, :] = softmax( logits[t, h, :] + causal_mask + sink_slot )
//   o[t, h]          = weights[t, h, :] @ gathered_kv
//
// For ``compress_ratio == 0`` layers the gathered_kv is just the SWA cache
// contents up to the current sequence length (it was staged by the sibling
// ``mla_v4_prefill_gather`` task). Causal mask: ``q_pos >= kv_pos`` where
// the Q row's absolute position is taken to be ``q_tile_offset + q_local``.
// In the v1 SWA-only path the caller passes ``T_q == T_kv`` so the absolute
// positions line up; future revisions (ratio=4/128) will pass an explicit
// per-token KV-length list -- v1 only sees a single ``T_kv``.
//
// MPK conventions:
//   * ``__device__ __forceinline__`` task impl.
//   * blockIdx-agnostic; per-task slice is selected via
//     ``task_metadata.token_offset`` (here interpreted as the *Q-tile* offset
//     in rows, i.e. the first Q row processed by this CTA is
//     ``q_tile_offset = token_offset``). When ``Q_TILE > 1`` one CTA covers
//     ``Q_TILE`` consecutive Q rows.
//   * Threads with ``threadIdx.x >= NUM_THREADS`` early-out so a smaller-CTA
//     task can ride a 256-thread worker block.
//   * No TMA / warp specialization in v1; plain shared-memory tiles with
//     FP32 logit/output accumulators.
// =============================================================================

#pragma once
#include "tasks/common/common_header.cuh"
#include <cfloat>
#include <cuda_bf16.h>
#include <float.h>

namespace kernel {

// Local sentinel for "no logit observed yet" used by the FlashAttn online
// softmax state. We use a finite sentinel (-1e30f) instead of -FLT_MAX so
// arithmetic on the sentinel (e.g. ``exp(m_state - m_new)``) stays well-
// defined when m_state has not yet been updated. The branch
// ``m_state == kNegInf`` then selects ``scale_old = 0`` (the
// first-tile initialization), matching the standard FlashAttention init.
namespace mla_v4_prefill_detail {

constexpr float kNegInf = -1e30f;


// Block-reduce ``local_max`` across the first NUM_THREADS lanes via warp
// shuffles + smem cross-warp broadcast. ``reduce_smem`` must have at least
// ``NUM_WARPS`` floats; the result is broadcast to every lane through slot 0.
template <int NUM_THREADS>
__device__ __forceinline__ float
block_reduce_max(float local_max, float *reduce_smem) {
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a multiple of warp size");
#pragma unroll
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    local_max = fmaxf(local_max, shfl_xor_sync(local_max, offset));
  }
  int lane = threadIdx.x % NUM_THREADS_PER_WARP;
  int warp = threadIdx.x / NUM_THREADS_PER_WARP;
  if (lane == 0) {
    reduce_smem[warp] = local_max;
  }
  __syncthreads();
  float v = (threadIdx.x < NUM_WARPS) ? reduce_smem[threadIdx.x] : kNegInf;
  if (warp == 0) {
#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
      v = fmaxf(v, shfl_xor_sync(v, offset));
    }
    if (threadIdx.x == 0) {
      reduce_smem[0] = v;
    }
  }
  __syncthreads();
  return reduce_smem[0];
}

template <int NUM_THREADS>
__device__ __forceinline__ float
block_reduce_sum(float local_sum, float *reduce_smem) {
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a multiple of warp size");
#pragma unroll
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    local_sum += shfl_xor_sync(local_sum, offset);
  }
  int lane = threadIdx.x % NUM_THREADS_PER_WARP;
  int warp = threadIdx.x / NUM_THREADS_PER_WARP;
  if (lane == 0) {
    reduce_smem[warp] = local_sum;
  }
  __syncthreads();
  float v = (threadIdx.x < NUM_WARPS) ? reduce_smem[threadIdx.x] : 0.f;
  if (warp == 0) {
#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
      v += shfl_xor_sync(v, offset);
    }
    if (threadIdx.x == 0) {
      reduce_smem[0] = v;
    }
  }
  __syncthreads();
  return reduce_smem[0];
}

} // namespace mla_v4_prefill_detail

// Template parameters:
//   NUM_HEADS  : number of attention heads (per-token Q rows we loop over).
//   HEAD_DIM   : per-head dim (= MLA latent width, e.g. 512 in V4-Flash).
//   Q_TILE     : Q-rows per CTA. v1: 1 per CTA (one Q-row per task).
//   KV_TILE    : KV-rows processed per inner step. v1: 32 (any power-of-2
//                that fits in shared memory).
//   NUM_THREADS: active threads per CTA (must divide HEAD_DIM and KV_TILE).
template <int NUM_HEADS, int HEAD_DIM, int Q_TILE, int KV_TILE,
          int NUM_THREADS = 128>
__device__ __forceinline__ void mla_v4_prefill_task_impl(
    void const *q_ptr,              // bf16 [T_q, NUM_HEADS, HEAD_DIM]
    void const *gathered_kv_ptr,    // bf16 [T_kv, HEAD_DIM]
    void const *attn_sink_ptr,      // fp32 [NUM_HEADS]
    void *o_ptr,                    // bf16 [T_q, NUM_HEADS, HEAD_DIM]
    int q_tile_offset,              // first Q-row processed by this CTA
    int T_q,
    int T_kv,
    float softmax_scale) {
  static_assert(Q_TILE >= 1, "Q_TILE must be >= 1");
  static_assert(KV_TILE >= 1, "KV_TILE must be >= 1");
  static_assert(HEAD_DIM % NUM_THREADS == 0 || NUM_THREADS % HEAD_DIM == 0,
                "HEAD_DIM and NUM_THREADS must be compatible for the parallel "
                "loops below");
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a multiple of warp size");

  // Worker thread blocks have WORKER_NUM_THREADS lanes; only the first
  // NUM_THREADS participate. The rest early-exit so the kernel composes
  // with any larger worker block.
  if (threadIdx.x >= NUM_THREADS) {
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
  __shared__ float reduce_smem[NUM_WARPS > 1 ? NUM_WARPS : 1];

  __nv_bfloat16 const *__restrict__ q =
      reinterpret_cast<__nv_bfloat16 const *>(q_ptr);
  __nv_bfloat16 const *__restrict__ kv =
      reinterpret_cast<__nv_bfloat16 const *>(gathered_kv_ptr);
  float const *__restrict__ sink =
      reinterpret_cast<float const *>(attn_sink_ptr);
  __nv_bfloat16 *__restrict__ o = reinterpret_cast<__nv_bfloat16 *>(o_ptr);

  // Each CTA processes ``Q_TILE`` consecutive Q rows starting at
  // ``q_tile_offset``. The kernel loops over them in serial (rows are
  // independent under MLA prefill). For v1 we expect ``Q_TILE == 1``; the
  // loop keeps the kernel correct for larger tiles too.
#pragma unroll 1
  for (int qi = 0; qi < Q_TILE; ++qi) {
    int q_row = q_tile_offset + qi;
    if (q_row >= T_q) {
      break;
    }
    // Absolute position of this Q row inside the full sequence. In the v1
    // SWA-only (compress_ratio == 0) path, ``T_q == T_kv`` so q_pos lines
    // up with kv_pos directly. The causal mask threshold is therefore
    // ``q_pos == q_row`` (the gather staged exactly the rows the kernel
    // is allowed to attend to).
    int const q_pos = q_row;

#pragma unroll 1
    for (int h = 0; h < NUM_HEADS; ++h) {
      // Per-head online softmax state.
      float m_state = mla_v4_prefill_detail::kNegInf;   // running max
      float d_state = 0.f;        // running normalizer (sum of exp)
      float o_acc[HEAD_DIM];      // running output accumulator (fp32)
#pragma unroll
      for (int d = 0; d < HEAD_DIM; ++d) {
        o_acc[d] = 0.f;
      }

      __nv_bfloat16 const *q_row_ptr =
          q + (static_cast<size_t>(q_row) * NUM_HEADS + h) * HEAD_DIM;
      // The attn_sink contributes a single extra logit slot per head; it
      // participates in the softmax denominator but its value-contribution
      // is zero (MLA's sink is a no-token slot). We fold it into the
      // FlashAttn online softmax by treating it as a single "virtual" KV
      // row with logit ``sink[h]`` and value vector 0.
      float sink_h = sink[h];

      // Loop over KV tiles. The causal mask trims the loop bound to
      // ``min(T_kv, q_pos + 1)``; rows past that contribute -inf.
      int const kv_end = min(T_kv, q_pos + 1);

#pragma unroll 1
      for (int kv_base = 0; kv_base < kv_end; kv_base += KV_TILE) {
        // Per-thread logit for one KV row inside the tile.
        float logit_local[KV_TILE];
#pragma unroll
        for (int k = 0; k < KV_TILE; ++k) {
          logit_local[k] = mla_v4_prefill_detail::kNegInf;
        }

        // Compute logits for this tile. All NUM_THREADS lanes cooperate on
        // each (q_row, k) dot product; ``HEAD_DIM`` must divide NUM_THREADS
        // (e.g. HEAD_DIM=64, NUM_THREADS=128 -> the lower 64 lanes load,
        // others get 0 from the strided loop and the reduce handles it).
#pragma unroll 1
        for (int k = 0; k < KV_TILE; ++k) {
          int kv_row = kv_base + k;
          if (kv_row >= kv_end) {
            continue;
          }
          // Partial dot product across HEAD_DIM, strided by NUM_THREADS.
          float partial = 0.f;
          __nv_bfloat16 const *kv_row_ptr =
              kv + static_cast<size_t>(kv_row) * HEAD_DIM;
          for (int d = threadIdx.x; d < HEAD_DIM; d += NUM_THREADS) {
            float qv = __bfloat162float(q_row_ptr[d]);
            float kvv = __bfloat162float(kv_row_ptr[d]);
            partial += qv * kvv;
          }
          // Warp + cross-warp reduction.
          float dot =
              mla_v4_prefill_detail::block_reduce_sum<NUM_THREADS>(
                  partial, reduce_smem);
          logit_local[k] = dot * softmax_scale;
        }

        // Stage 1: tile max.
        float tile_max = sink_h; // include sink slot in the max
#pragma unroll
        for (int k = 0; k < KV_TILE; ++k) {
          int kv_row = kv_base + k;
          if (kv_row < kv_end) {
            tile_max = fmaxf(tile_max, logit_local[k]);
          }
        }
        // (All lanes share the same logit_local because block_reduce_sum
        // broadcasts.) tile_max is therefore identical across lanes.

        float m_new = fmaxf(m_state, tile_max);
        float scale_old =
            (m_state == mla_v4_prefill_detail::kNegInf)
                ? 0.f
                : __expf(m_state - m_new);
        float scale_sink = __expf(sink_h - m_new);

        // Rescale running outputs / normalizer; on the first tile this is
        // the identity (scale_old == 0 picks up sink + p contributions).
        float d_new = d_state * scale_old + scale_sink;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
          o_acc[d] *= scale_old;
        }

        // Stage 2: accumulate p_k * V_k = p_k * gathered_kv[kv_row] (MLA).
#pragma unroll 1
        for (int k = 0; k < KV_TILE; ++k) {
          int kv_row = kv_base + k;
          if (kv_row >= kv_end) {
            continue;
          }
          float p = __expf(logit_local[k] - m_new);
          d_new += p;
          __nv_bfloat16 const *kv_row_ptr =
              kv + static_cast<size_t>(kv_row) * HEAD_DIM;
          for (int d = threadIdx.x; d < HEAD_DIM; d += NUM_THREADS) {
            float vd = __bfloat162float(kv_row_ptr[d]);
            o_acc[d] += p * vd;
          }
        }

        m_state = m_new;
        d_state = d_new;
      }

      // Epilogue: normalize and write back.
      float inv_d = (d_state > 0.f) ? (1.f / d_state) : 0.f;
      __nv_bfloat16 *o_row_ptr =
          o + (static_cast<size_t>(q_row) * NUM_HEADS + h) * HEAD_DIM;
      for (int d = threadIdx.x; d < HEAD_DIM; d += NUM_THREADS) {
        float v = o_acc[d] * inv_d;
        o_row_ptr[d] = __float2bfloat16(v);
      }
      // block_reduce_* end with a __syncthreads(), so reduce_smem can be
      // safely reused on the next head's first reduce call. The output
      // writes above target disjoint global memory, so no extra barrier is
      // needed between heads.
    }
  }
}

} // namespace kernel
