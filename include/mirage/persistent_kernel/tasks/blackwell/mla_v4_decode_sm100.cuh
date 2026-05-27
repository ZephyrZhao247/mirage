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
// mla_v4_decode_sm100.cuh — DeepSeek V4-Flash MLA decode (v1: SWA-only).
//
// V4-Flash MLA decode reads two caches (SWA + optional compressed) and
// applies an optional sparse top-K indexing scheme. For v1 we only
// support ``compress_ratio == 0`` (the simplest case):
//   * SWA cache only.
//   * No compressed cache.
//   * No sparse top-K indices.
// The dual-cache / sparse paths are reserved for v2; this kernel exposes
// signature slots for them (currently unused) so the v2 port can extend
// in place. See docs/mpk/deepseek_v4/attention.md § mla_v4_decode_layer.
//
// Math (single-token decode at position ``pos``):
//
//   * Q is per-head (NUM_HEADS heads, HEAD_DIM cols, last ROPE_DIM are
//     RoPE'd — already applied upstream).
//   * KV is the **same row** for K and V (MLA latent), HEAD_DIM cols.
//   * Attend over positions 0..pos-1 in the SWA cache, capped at
//     ``sliding_window`` (= cache stride). For v1 we assume the test
//     constructs the cache so that the linear position index into the
//     ``[num_pages, page_size, HEAD_DIM]`` cache equals the absolute
//     position; multi-page paging will be added with the v2 port that
//     plumbs the paged-KV indptr buffer.
//
//     logits[h, p] = (q[h] @ kv[p].T) * softmax_scale
//     full[h, .]   = concat(logits[h, :pos], attn_sink[h])        // sink lane
//     w[h, .]      = softmax(full[h, .])                          // FP32
//     o[h, d]      = sum_{p < pos}(w[h, p] * kv[p, d])            // K == V
//
// MPK convention:
//   * ``__device__ __forceinline__`` task impl.
//   * blockIdx-agnostic: derives ``token_offset`` from
//     ``task_desc->task_metadata``.
//   * v1: one CTA per token (head loop is internal). Templated on
//     ``NUM_HEADS, HEAD_DIM, ROPE_DIM, NUM_THREADS``.
//   * FP32 accumulators throughout; online-max softmax (Flash-attention
//     trick) so the per-token loop only walks the cache once.
//   * No TMA, no warp specialization, no tcgen05 — naive global loads
//     into FP32 registers.
//
// Optional inputs (v2 placeholders; unused in v1):
//   * extra_k_cache_ptr     — compressed (ratio>0) cache.
//   * topk_indices_ptr      — sparse top-K positions (per request).
//
// =============================================================================

#pragma once
#include "tasks/common/common_header.cuh"
#include <cuda_bf16.h>

namespace kernel {

namespace mla_v4_decode_detail {

// Block-wide max reduction across the first NUM_THREADS lanes via warp
// shuffles + a small smem cross-warp reduction. Returns the same value
// to every thread (broadcast via smem slot 0). Caller is responsible
// for the surrounding __syncthreads().
template <int NUM_THREADS>
__device__ __forceinline__ float
block_reduce_max(float local_val, float *reduce_smem) {
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a multiple of warp size");
  // Warp-local max.
#pragma unroll
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    float other = shfl_xor_sync(local_val, offset);
    local_val = fmaxf(local_val, other);
  }
  int lane = threadIdx.x % NUM_THREADS_PER_WARP;
  int warp = threadIdx.x / NUM_THREADS_PER_WARP;
  if (lane == 0) {
    reduce_smem[warp] = local_val;
  }
  __syncthreads();
  float v =
      (threadIdx.x < NUM_WARPS) ? reduce_smem[threadIdx.x] : -CUDART_INF_F;
  if (warp == 0) {
#pragma unroll
    for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
      float other = shfl_xor_sync(v, offset);
      v = fmaxf(v, other);
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
block_reduce_sum(float local_val, float *reduce_smem) {
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a multiple of warp size");
#pragma unroll
  for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
    local_val += shfl_xor_sync(local_val, offset);
  }
  int lane = threadIdx.x % NUM_THREADS_PER_WARP;
  int warp = threadIdx.x / NUM_THREADS_PER_WARP;
  if (lane == 0) {
    reduce_smem[warp] = local_val;
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

} // namespace mla_v4_decode_detail

// =====================================================================
// mla_v4_decode_task_impl: v1 single-token MLA decode (SWA-only).
//
// Inputs (pointers):
//   q_ptr         bf16 [num_tokens_total, NUM_HEADS, HEAD_DIM]
//   swa_cache_ptr bf16 [SWA_TOTAL, HEAD_DIM]   (logically flattened from
//                                              [num_pages, page_size, HEAD_DIM];
//                                              v1 treats it as contiguous)
//   positions_ptr int32 [num_tokens_total]
//   attn_sink_ptr fp32  [NUM_HEADS]
//   o_ptr         bf16 [num_tokens_total, NUM_HEADS, HEAD_DIM]
//
// Runtime params:
//   token_offset       — which token this CTA handles.
//   num_tokens_total   — bound for the q/o leading axis.
//   sliding_window     — cap on the attend length (positions
//                        [pos - sliding_window, pos) are valid; v1
//                        clamps to the contiguous prefix [0, pos)).
//   softmax_scale      — scalar logit scale.
//
// Layout assumptions (v1):
//   * For each token at position ``pos``, attend over the rows
//     ``swa_cache[0..pos-1]`` of the linearised cache (the test harness
//     stages a contiguous cache). v2 will plumb the paged-KV indptr /
//     last-page-len buffers from the MPK runtime here.
//   * One CTA per token; all NUM_THREADS lanes cooperate on every head's
//     dot products.
// =====================================================================
template <int NUM_HEADS, int HEAD_DIM, int ROPE_DIM, int NUM_THREADS = 256>
__device__ __forceinline__ void
mla_v4_decode_task_impl(void const *q_ptr,
                        void const *swa_cache_ptr,
                        void const *positions_ptr,
                        void const *attn_sink_ptr,
                        void *o_ptr,
                        int token_offset,
                        int num_tokens_total,
                        int sliding_window,
                        float softmax_scale) {
  static_assert(HEAD_DIM > 0, "HEAD_DIM must be positive");
  static_assert(ROPE_DIM >= 0 && ROPE_DIM <= HEAD_DIM,
                "0 <= ROPE_DIM <= HEAD_DIM required");
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a multiple of warp size");

  // Worker thread blocks may have up to WORKER_NUM_THREADS threads. Only
  // the first NUM_THREADS lanes participate.
  if (threadIdx.x >= NUM_THREADS) {
    return;
  }
  if (token_offset >= num_tokens_total) {
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;

  __nv_bfloat16 const *__restrict__ q_base =
      reinterpret_cast<__nv_bfloat16 const *>(q_ptr);
  __nv_bfloat16 const *__restrict__ kv_base =
      reinterpret_cast<__nv_bfloat16 const *>(swa_cache_ptr);
  int const *__restrict__ positions =
      reinterpret_cast<int const *>(positions_ptr);
  float const *__restrict__ attn_sink =
      reinterpret_cast<float const *>(attn_sink_ptr);
  __nv_bfloat16 *__restrict__ o_base = reinterpret_cast<__nv_bfloat16 *>(o_ptr);

  int const t = token_offset;
  int const pos = positions[t];
  // V1: attend over [0, pos), capped at sliding_window. (Future v2: the
  // SWA cache uses a ring buffer indexed by ``pos % sliding_window``.)
  int const kv_len = (pos < sliding_window) ? pos : sliding_window;

  // Pointers for this token row.
  __nv_bfloat16 const *__restrict__ q_t =
      q_base + static_cast<size_t>(t) * NUM_HEADS * HEAD_DIM;
  __nv_bfloat16 *__restrict__ o_t =
      o_base + static_cast<size_t>(t) * NUM_HEADS * HEAD_DIM;

  // Shared workspace: per-warp reduction buffer + per-head running stats
  // (m = running max, l = running normaliser). Logit values are large so
  // we keep stats in FP32. We avoid storing all logits; we use the
  // online-max trick: walk p once, maintaining m_h, l_h, and the
  // accumulated FP32 output o[h, :] in registers.
  //
  // For each head h we accumulate ``acc[h, d]`` (FP32) over all p; with
  // HEAD_DIM cols per head and NUM_THREADS lanes the per-thread slice is
  // ``(NUM_HEADS * HEAD_DIM) / NUM_THREADS`` elements. We use a simple
  // mapping: thread ``tid`` owns the column indices ``d`` such that
  // ``tid == (h * HEAD_DIM + d) % NUM_THREADS``. With small shapes
  // (HEAD_DIM <= 512, NUM_HEADS <= 64) NUM_THREADS=256 covers everything
  // in a handful of strided slots. To keep the kernel simple and within
  // the v1 spec, we use shared-memory accumulators and have all lanes
  // contribute via atomic-free coarse partitioning.
  //
  // Pragmatic layout: stash ``acc[NUM_HEADS, HEAD_DIM]`` in smem (fp32).
  // For NUM_HEADS=4, HEAD_DIM=64 this is 1 KB — trivial. For production
  // sizes (64, 512) it is 128 KB — too large for naive smem. The v1
  // tests use small shapes; we cap the static smem at a reasonable
  // ceiling and reject (at compile time) configurations beyond it.

  constexpr int ACC_ELEMS = NUM_HEADS * HEAD_DIM;
  static_assert(ACC_ELEMS <= 32 * 1024,
                "v1 kernel: NUM_HEADS * HEAD_DIM exceeds 32K fp32 entries");

  __shared__ float reduce_smem[NUM_WARPS > 1 ? NUM_WARPS : 1];
  __shared__ float acc_smem[ACC_ELEMS];      // fp32 accumulator
  __shared__ float m_smem[NUM_HEADS];        // running max per head
  __shared__ float l_smem[NUM_HEADS];        // running normaliser per head

  // Initialise running stats. Start max = -inf; once we fold in
  // attn_sink_h below we'll have a finite running max even for kv_len=0.
  for (int i = threadIdx.x; i < NUM_HEADS; i += NUM_THREADS) {
    m_smem[i] = -CUDART_INF_F;
    l_smem[i] = 0.f;
  }
  for (int i = threadIdx.x; i < ACC_ELEMS; i += NUM_THREADS) {
    acc_smem[i] = 0.f;
  }
  __syncthreads();

  // We walk p = 0..kv_len-1, computing logits for ALL heads at this p
  // and updating each head's running stats independently. (Per-head
  // online-softmax — heads are independent.)
  for (int p = 0; p < kv_len; ++p) {
    __nv_bfloat16 const *__restrict__ kv_p =
        kv_base + static_cast<size_t>(p) * HEAD_DIM;

    // Compute logits[h] = (q[h] @ kv_p) * softmax_scale  for each h.
    // Cooperative dot-product: distribute HEAD_DIM across NUM_THREADS.
    // (For HEAD_DIM=64 + NUM_THREADS=256, each thread handles 0 or 1
    // element; we keep it generic.)
    for (int h = 0; h < NUM_HEADS; ++h) {
      __nv_bfloat16 const *__restrict__ q_h = q_t + h * HEAD_DIM;
      float local = 0.f;
      for (int d = threadIdx.x; d < HEAD_DIM; d += NUM_THREADS) {
        float qv = __bfloat162float(q_h[d]);
        float kv = __bfloat162float(kv_p[d]);
        local += qv * kv;
      }
      float dot = mla_v4_decode_detail::block_reduce_sum<NUM_THREADS>(
          local, reduce_smem);
      float logit = dot * softmax_scale;

      // Online softmax update for head h:
      //   new_m = max(m_h, logit)
      //   alpha = exp(m_h - new_m)
      //   l_h <- alpha * l_h + exp(logit - new_m)
      //   acc[h, :] <- alpha * acc[h, :] + exp(logit - new_m) * kv_p[:]
      // Performed by thread 0 (m, l) and all threads (acc).
      float m_prev = m_smem[h];
      float m_new = fmaxf(m_prev, logit);
      // Guard against -inf - -inf when both are -inf (kv_len=0 case
      // can't reach here, but be paranoid).
      float alpha =
          (m_prev == -CUDART_INF_F) ? 0.f : __expf(m_prev - m_new);
      float w = __expf(logit - m_new);
      if (threadIdx.x == 0) {
        l_smem[h] = alpha * l_smem[h] + w;
        m_smem[h] = m_new;
      }
      // Rescale + accumulate o[h, :].
      for (int d = threadIdx.x; d < HEAD_DIM; d += NUM_THREADS) {
        float kvv = __bfloat162float(kv_p[d]);
        float prev = acc_smem[h * HEAD_DIM + d];
        acc_smem[h * HEAD_DIM + d] = alpha * prev + w * kvv;
      }
      __syncthreads();
    }
  }

  // Fold the attn_sink term into the softmax denominator (a "sink lane"
  // contributes only to l, never to acc — it has no corresponding KV
  // row).
  for (int h = 0; h < NUM_HEADS; ++h) {
    float sink = attn_sink[h];
    float m_prev = m_smem[h];
    float m_new = fmaxf(m_prev, sink);
    float alpha = (m_prev == -CUDART_INF_F) ? 0.f : __expf(m_prev - m_new);
    float w = __expf(sink - m_new);
    if (threadIdx.x == 0) {
      l_smem[h] = alpha * l_smem[h] + w;
      m_smem[h] = m_new;
    }
    for (int d = threadIdx.x; d < HEAD_DIM; d += NUM_THREADS) {
      float prev = acc_smem[h * HEAD_DIM + d];
      acc_smem[h * HEAD_DIM + d] = alpha * prev;
    }
    __syncthreads();
  }

  // Normalise and write back.
  for (int h = 0; h < NUM_HEADS; ++h) {
    float l_inv = 1.f / l_smem[h];
    for (int d = threadIdx.x; d < HEAD_DIM; d += NUM_THREADS) {
      float v = acc_smem[h * HEAD_DIM + d] * l_inv;
      o_t[h * HEAD_DIM + d] = __float2bfloat16(v);
    }
  }
}

} // namespace kernel
