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

// V4-Flash dsv3_router_gemm (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/dsv3_router_gemm.md
// vLLM reference: GateLinear.forward() with 4-tier dispatch:
//   Tier 1: dsv3_router_gemm  (DSV3-specialized, H=7168, E∈{256,384}, M<=16)
//   Tier 2: fp32_router_gemm  (fp32-specialized, H=3072, E=256, M<=32)
//   Tier 3: torch.mm bf16xbf16->fp32 (cuBLASLt)         [V4-Flash plain matmul]
//   Tier 4: F.linear fallback (ReplicatedLinear.forward)[V4-Flash fallback]
//
// V4-Flash hidden_size = 4096, n_routed_experts = 256, so it lands on
// Tier 3 / Tier 4 (plain matmul). This naive port implements **just the
// Tier 3 / Tier 4 path**: a plain bf16 in / bf16 weight / fp32 out matmul.
// Tier-1 and Tier-2 are locked alternative pointers (DSV3 / Kimi K2 specific
// shape gates); see docstring in the Python catalog for those references.
//
// Computes per-token row: out[m, e] = sum_k mat_a[m, k].float()
//                                          * mat_b[e, k].float()
// One CTA per (token, expert) is too wide (256 experts * T CTAs);
// for V4-Flash we map one CTA per TOKEN, and the CTA produces all
// E outputs sequentially with a thread-strided K loop. fp32 accumulator.
//
// Naive design (correctness only, no perf):
//   * One CTA per token. grid = (num_tokens, 1, 1).
//   * NUM_THREADS = 256 (Blackwell default WORKER_NUM_THREADS).
//   * For each expert e in [0, NUM_EXPERTS), thread-strided dot-product
//     over K = hidden_size; then warp + cross-warp reduce; thread 0 writes.
//   * No TMA / no UMMA / no warp-spec / no split-K.
//
// Tensor contract:
//   * mat_a (hidden_states): bf16 [T, H]  (this CTA's view: bf16 [H])
//   * mat_b (gate weight):   bf16 [E, H]  (broadcast: full E*H weight)
//   * out   (router_logits): fp32 [T, E]  (this CTA's view: fp32 [E])

namespace kernel {

namespace dsv3_router_gemm_v4_detail {

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

} // namespace dsv3_router_gemm_v4_detail

template <int NUM_EXPERTS, int HIDDEN_SIZE, int NUM_THREADS = 256>
__device__ __forceinline__ void dsv3_router_gemm_v4_sm100_impl(
    void const *mat_a_ptr, // bf16 [HIDDEN_SIZE] (this token's row)
    void const *mat_b_ptr, // bf16 [NUM_EXPERTS, HIDDEN_SIZE] (broadcast)
    void *out_ptr          // fp32 [NUM_EXPERTS] (this token's row)
) {
  static_assert(NUM_EXPERTS > 0, "NUM_EXPERTS must be positive");
  static_assert(HIDDEN_SIZE > 0, "HIDDEN_SIZE must be positive");
  static_assert(NUM_THREADS > 0, "NUM_THREADS must be positive");
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a warp multiple");

  using bf16 = type::bfloat16_t;
  namespace dt = dsv3_router_gemm_v4_detail;

  bf16 const *__restrict__ a = static_cast<bf16 const *>(mat_a_ptr);
  bf16 const *__restrict__ b = static_cast<bf16 const *>(mat_b_ptr);
  float *__restrict__ out = static_cast<float *>(out_ptr);

  extern __shared__ char smem[];
  float *reduce_smem = reinterpret_cast<float *>(smem);

  // Walk experts sequentially; for each, do a thread-strided fp32 dot-
  // product over K, warp + cross-warp reduce, thread 0 writes.
#pragma unroll 1
  for (int e = 0; e < NUM_EXPERTS; ++e) {
    bf16 const *__restrict__ b_row = b + e * HIDDEN_SIZE;
    float partial = 0.0f;
    for (int k = threadIdx.x; k < HIDDEN_SIZE; k += NUM_THREADS) {
      float av = static_cast<float>(a[k]);
      float bv = static_cast<float>(b_row[k]);
      partial += av * bv;
    }
    float dot = dt::warp_block_reduce_sum<NUM_THREADS>(partial, reduce_smem);
    if (threadIdx.x == 0) {
      out[e] = dot;
    }
    __syncthreads();
  }
}

} // namespace kernel
