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

// V4-Flash fp8_fp4_mqa_logits (NAIVE Blackwell SM100 impl, prefill non-paged).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/fp8_fp4_mqa_logits.md
//
// FP8 path only in the naive port. Non-paged: K is contiguous
// `[N, HEAD_DIM]` fp8 with a companion `[N]` fp32 per-token scale —
// the prefill sibling of `fp8_fp4_paged_mqa_logits`. Per-Q-row masking
// uses `[cu_seqlen_ks[q], cu_seqlen_ke[q])` ranges.
//
// Per-(q, kv) logit:
//   logits[kv] = sum_h weights[h] * (q_fp32[h] . k_fp32(kv))
//   q_fp32[h, d] = (fp8 -> fp32) q[h, d]
//   k_fp32[d]    = (fp8 -> fp32) k_packed[kv, d] * k_scales[kv]
//   weights = [N_HEADS] folded by Q-side kernel.
//
// Naive design (correctness only, no perf):
//   * One CTA per q-row. grid = (T_chunk, 1, 1). The catalog preoffsets
//     `q_ptr`, `weights_ptr`, `cu_seqlen_ks_ptr`, `cu_seqlen_ke_ptr`,
//     `logits_out_ptr` to this q-row.
//   * Each CTA loops over kv in [0, N). For each unmasked kv the
//     threads cooperatively compute the MQA reduction (sum over (h, d)
//     of N_HEADS*HEAD_DIM elements). Masked-out slots are left
//     untouched (clean_logits=False).
//   * NUM_THREADS = 256.
//   * No TMA / no UMMA / no warp-spec / no scheduler.
//
// Tensor layouts:
//   q_in        : fp8   [N_HEADS, HEAD_DIM]      (this q-row)
//   k_packed    : fp8   [N, HEAD_DIM]            (FULL K buffer)
//   k_scales    : fp32  [N]                      (FULL scale buffer)
//   weights     : fp32  [N_HEADS]                (this q-row)
//   cu_seqlen_ks: int32 [1]                      (this q-row entry)
//   cu_seqlen_ke: int32 [1]                      (this q-row entry)
//   logits_out  : fp32  [N]                      (this q-row's full logits row)
//
// Runtime param: `N` (int) — total KV positions in this chunk.

namespace kernel {

namespace fp8_fp4_mqa_logits_v4_detail {

template <int NUM_THREADS>
__device__ __forceinline__ float
warp_block_reduce_sum_mqa(float val, float *smem) {
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

} // namespace fp8_fp4_mqa_logits_v4_detail

template <int N_HEADS, int HEAD_DIM, int NUM_THREADS = 256>
__device__ __forceinline__ void fp8_fp4_mqa_logits_v4_sm100_impl(
    void const *q_in_ptr,         // fp8  [N_HEADS, HEAD_DIM]
    void const *k_packed_ptr,     // fp8  [N, HEAD_DIM]
    void const *k_scales_ptr,     // fp32 [N]
    void const *weights_ptr,      // fp32 [N_HEADS]
    void const *cu_seqlen_ks_ptr, // int32 [1]
    void const *cu_seqlen_ke_ptr, // int32 [1]
    void *logits_out_ptr,         // fp32 [N]
    int N) {
  static_assert(N_HEADS > 0, "N_HEADS must be positive");
  static_assert(HEAD_DIM > 0, "HEAD_DIM must be positive");
  static_assert(NUM_THREADS > 0, "NUM_THREADS must be positive");

  using fp8 = __nv_fp8_e4m3;

  fp8 const *__restrict__ q = static_cast<fp8 const *>(q_in_ptr);
  fp8 const *__restrict__ k_packed =
      static_cast<fp8 const *>(k_packed_ptr);
  float const *__restrict__ k_scales =
      static_cast<float const *>(k_scales_ptr);
  float const *__restrict__ w = static_cast<float const *>(weights_ptr);
  int32_t const *__restrict__ ks_arr =
      static_cast<int32_t const *>(cu_seqlen_ks_ptr);
  int32_t const *__restrict__ ke_arr =
      static_cast<int32_t const *>(cu_seqlen_ke_ptr);
  float *__restrict__ logits_out = static_cast<float *>(logits_out_ptr);

  int const ks = ks_arr[0];
  int const ke = ke_arr[0];

  extern __shared__ char smem_raw[];
  float *reduce_buf = reinterpret_cast<float *>(smem_raw);

  for (int kv = 0; kv < N; ++kv) {
    if (kv < ks || kv >= ke) {
      // clean_logits=False: untouched.
      continue;
    }
    float const k_scale = k_scales[kv];
    fp8 const *__restrict__ k_row =
        k_packed + static_cast<size_t>(kv) * HEAD_DIM;

    float local = 0.0f;
    int const total = N_HEADS * HEAD_DIM;
    for (int idx = threadIdx.x; idx < total; idx += NUM_THREADS) {
      int h = idx / HEAD_DIM;
      int d = idx - h * HEAD_DIM;
      float qv = static_cast<float>(q[h * HEAD_DIM + d]);
      float kv_v = static_cast<float>(k_row[d]) * k_scale;
      local += w[h] * qv * kv_v;
    }
    float reduced =
        fp8_fp4_mqa_logits_v4_detail::warp_block_reduce_sum_mqa<NUM_THREADS>(
            local, reduce_buf);

    if (threadIdx.x == 0) {
      logits_out[kv] = reduced;
    }
    __syncthreads();
  }
}

} // namespace kernel
