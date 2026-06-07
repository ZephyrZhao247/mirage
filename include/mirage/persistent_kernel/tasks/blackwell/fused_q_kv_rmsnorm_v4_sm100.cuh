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

// V4-Flash fused_q_kv_rmsnorm (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/fused_q_kv_rmsnorm.md
// Triton ref: vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py
//
// Joint per-token kernel running two RMSNorms back-to-back in a single
// CTA:
//   Q stream:  qr[t, :Q_SIZE]  --norm(q_weight, eps)-->  qr_out[t, :Q_SIZE]
//   KV stream: kv[t, :KV_SIZE] --norm(kv_weight, eps)--> kv_out[t, :KV_SIZE]
//
// Naive design (Blackwell, single CTA per token, no TMA/UMMA/warp-spec):
//   * One CTA per token, NUM_THREADS = 256 (Blackwell default).
//   * Sequential two-pass within the CTA: Q norm first, then KV norm.
//   * fp32 sum-of-squares + warp-shfl_xor + cross-warp combine in smem.
//   * bf16 in / bf16 out; weights are bf16 in V4-Flash.
//
// Inputs (TBGraph order):
//   input_ptrs[0] = qr           bf16  [num_tokens, Q_SIZE]
//   input_ptrs[1] = q_weight     bf16  [Q_SIZE]
//   input_ptrs[2] = kv           bf16  [num_tokens, KV_SIZE]
//   input_ptrs[3] = kv_weight    bf16  [KV_SIZE]
//
// Outputs:
//   output_ptrs[0] = qr_out      bf16  [num_tokens, Q_SIZE]
//   output_ptrs[1] = kv_out      bf16  [num_tokens, KV_SIZE]

namespace kernel {

namespace fused_q_kv_rmsnorm_v4_detail {

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

template <int SIZE, int NUM_THREADS>
__device__ __forceinline__ void
rmsnorm_row_naive(type::bfloat16_t const *__restrict__ x_in,
                  type::bfloat16_t const *__restrict__ w_in,
                  type::bfloat16_t *__restrict__ x_out,
                  float *reduce_smem,
                  float eps) {
  float partial = 0.0f;
#pragma unroll 1
  for (int i = threadIdx.x; i < SIZE; i += NUM_THREADS) {
    float v = static_cast<float>(x_in[i]);
    partial += v * v;
  }
  float sumsq =
      warp_block_reduce_sum<NUM_THREADS>(partial, reduce_smem);
  float inv_rms = rsqrtf(sumsq / static_cast<float>(SIZE) + eps);

#pragma unroll 1
  for (int i = threadIdx.x; i < SIZE; i += NUM_THREADS) {
    float v = static_cast<float>(x_in[i]);
    float w = static_cast<float>(w_in[i]);
    x_out[i] = type::bfloat16_t(v * inv_rms * w);
  }
  __syncthreads();
}

} // namespace fused_q_kv_rmsnorm_v4_detail

template <int Q_SIZE, int KV_SIZE, int NUM_THREADS = 256>
__device__ __forceinline__ void fused_q_kv_rmsnorm_v4_sm100_impl(
    void const *qr_ptr,
    void const *q_weight_ptr,
    void const *kv_ptr,
    void const *kv_weight_ptr,
    void *qr_out_ptr,
    void *kv_out_ptr,
    float eps) {
  static_assert(Q_SIZE > 0 && KV_SIZE > 0, "sizes must be positive");
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a warp multiple");

  using bf16 = type::bfloat16_t;

  extern __shared__ char smem[];
  float *reduce_smem = reinterpret_cast<float *>(smem);

  bf16 const *qr = static_cast<bf16 const *>(qr_ptr);
  bf16 const *qw = static_cast<bf16 const *>(q_weight_ptr);
  bf16 const *kv = static_cast<bf16 const *>(kv_ptr);
  bf16 const *kw = static_cast<bf16 const *>(kv_weight_ptr);
  bf16 *qr_out = static_cast<bf16 *>(qr_out_ptr);
  bf16 *kv_out = static_cast<bf16 *>(kv_out_ptr);

  fused_q_kv_rmsnorm_v4_detail::rmsnorm_row_naive<Q_SIZE, NUM_THREADS>(
      qr, qw, qr_out, reduce_smem, eps);
  fused_q_kv_rmsnorm_v4_detail::rmsnorm_row_naive<KV_SIZE, NUM_THREADS>(
      kv, kw, kv_out, reduce_smem, eps);
}

} // namespace kernel
