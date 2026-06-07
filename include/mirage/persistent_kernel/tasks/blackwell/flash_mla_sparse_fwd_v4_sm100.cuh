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

// V4-Flash FlashMLA sparse prefill (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/flash_mla_sparse_fwd.md
//
// Per (token, head) one CTA; KV is bf16 pre-gathered upstream by
// dequantize_and_gather_k_cache. Online softmax over the index list.

namespace kernel {

template <int HEAD_DIM,
          int HEAD_V,
          int NUM_HEADS_Q,
          int NUM_THREADS = 256>
__device__ __forceinline__ void flash_mla_sparse_fwd_v4_sm100_impl(
    void const *q_ptr,
    void const *kv_ptr,
    void const *indices_ptr,
    void const *topk_length_ptr,
    void const *attn_sink_ptr,
    void *out_ptr,
    int s_kv,
    int head_id,
    float softmax_scale) {
  static_assert(HEAD_DIM > HEAD_V, "HEAD_DIM must include rope tail");

  using bf16 = type::bfloat16_t;

  bf16 const *q_all = static_cast<bf16 const *>(q_ptr);
  bf16 const *kv_all = static_cast<bf16 const *>(kv_ptr);
  int32_t const *indices = static_cast<int32_t const *>(indices_ptr);
  int32_t const *topk_len = static_cast<int32_t const *>(topk_length_ptr);
  float const *attn_sink = static_cast<float const *>(attn_sink_ptr);
  bf16 *out_all = static_cast<bf16 *>(out_ptr);

  bf16 const *q_h = q_all + head_id * HEAD_DIM;
  bf16 *out_h = out_all + head_id * HEAD_V;

  extern __shared__ char smem_raw[];
  float *s_q = reinterpret_cast<float *>(smem_raw);
  float *s_k = s_q + HEAD_DIM;
  float *s_acc = s_k + HEAD_DIM;

#pragma unroll 1
  for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
    s_q[i] = static_cast<float>(q_h[i]);
  }
#pragma unroll 1
  for (int i = threadIdx.x; i < HEAD_V; i += NUM_THREADS) {
    s_acc[i] = 0.0f;
  }
  __syncthreads();

  int L = topk_len[0];
  __shared__ float s_m;
  __shared__ float s_l;
  if (threadIdx.x == 0) {
    s_m = -INFINITY;
    s_l = 0.0f;
  }
  __syncthreads();
  if (L <= 0) {
#pragma unroll 1
    for (int i = threadIdx.x; i < HEAD_V; i += NUM_THREADS) {
      out_h[i] = bf16(0.0f);
    }
    return;
  }

  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
  __shared__ float s_red[NUM_WARPS];

  for (int k = 0; k < L; ++k) {
    int slot = indices[k];
    if (slot < 0 || slot >= s_kv) continue;

    bf16 const *k_row = kv_all + static_cast<int64_t>(slot) * HEAD_DIM;
#pragma unroll 1
    for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
      s_k[i] = static_cast<float>(k_row[i]);
    }
    __syncthreads();

    float partial = 0.0f;
#pragma unroll 1
    for (int i = threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
      partial += s_q[i] * s_k[i];
    }
#pragma unroll
    for (int offset = NUM_THREADS_PER_WARP / 2; offset > 0; offset /= 2) {
      partial += shfl_xor_sync(partial, offset);
    }
    int lane = threadIdx.x % NUM_THREADS_PER_WARP;
    int warp = threadIdx.x / NUM_THREADS_PER_WARP;
    if (lane == 0) s_red[warp] = partial;
    __syncthreads();
    if (warp == 0) {
      float v = (threadIdx.x < NUM_WARPS) ? s_red[threadIdx.x] : 0.0f;
#pragma unroll
      for (int offset = NUM_WARPS / 2; offset > 0; offset /= 2) {
        v += shfl_xor_sync(v, offset);
      }
      if (lane == 0) s_red[0] = v;
    }
    __syncthreads();
    float qk = s_red[0] * softmax_scale;

    float old_m = s_m;
    float new_m = fmaxf(old_m, qk);
    float alpha = (old_m == -INFINITY) ? 0.0f : expf(old_m - new_m);
    float p = expf(qk - new_m);
    if (threadIdx.x == 0) {
      s_m = new_m;
      s_l = s_l * alpha + p;
    }
    __syncthreads();

#pragma unroll 1
    for (int i = threadIdx.x; i < HEAD_V; i += NUM_THREADS) {
      s_acc[i] = s_acc[i] * alpha + p * s_k[i];
    }
    __syncthreads();
  }

  float l = s_l;
  float m = s_m;
  float sink_factor = 1.0f;
  if (attn_sink != nullptr && l > 0.0f) {
    float sink = attn_sink[head_id];
    float exp_sink_minus_m = expf(sink - m);
    sink_factor = l / (l + exp_sink_minus_m);
  }
  float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
#pragma unroll 1
  for (int i = threadIdx.x; i < HEAD_V; i += NUM_THREADS) {
    out_h[i] = bf16(s_acc[i] * inv_l * sink_factor);
  }
  __syncthreads();
}

} // namespace kernel
