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

// V4-Flash fp8_fp4_mega_moe (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/fp8_fp4_mega_moe.md
//
// The vLLM kernel is a heavyweight persistent-grid cluster-MMA kernel
// that fuses MoE dispatch + L1 GEMM + SwiGLU + L2 GEMM + combine +
// allreduce in a single CTA grid with NVLink multicast.  Our NAIVE port
// keeps the same per-(token) output contract -- y[T, H] bf16 -- but
// uses bf16 weights and a single CTA per token: L1 matmul -> SwiGLU
// (with optional swiglu_limit clamp) -> L2 matmul -> per-topk combine.
// No FP4/FP8 packing, no multicast, no all-reduce.  Higher-precision
// (FP4 weights, FP8 acts) is a follow-up; the I/O contract is preserved
// at the bf16 level so the surrounding model code can swap in the
// fused kernel later.
//
// Naive design:
//   * One CTA per token.  Grid = (T, 1, 1).  request_id = bid.x.
//   * NUM_THREADS = 256.
//   * For each topk slot k in [0, TOP_K):
//       - expert e = topk_idx[t, k]
//       - L1 GEMM: l1_acc[2*I] = A[t, :] @ W13[e].T  in fp32
//       - SwiGLU: gate = l1_acc[:I], up = l1_acc[I:2*I]
//                 (optional clamp by ACTIVATION_CLAMP -- when > 0:
//                  gate <- min(gate, clamp); up <- clip(up, -clamp, clamp))
//                 h = silu(gate) * up   (fp32, exact silu)
//       - L2 GEMM: l2_acc[H] = h @ W2[e].T   in fp32
//       - Accumulate: y_acc[H] += topk_w[t, k] * l2_acc
//   * Store y[t, :] = bf16(y_acc).
//
// I/O contract:
//   inputs:
//     [0] A          bf16 [T, H]                 -- per-token activation
//     [1] W13        bf16 [E, 2*I, H]            -- gate+up combined
//     [2] W2         bf16 [E, H, I]              -- down projection
//     [3] topk_idx   int64 [T, TOP_K]            -- routing indices
//     [4] topk_w     fp32  [T, TOP_K]            -- router weights
//   outputs:
//     [0] y          bf16  [T, H]
//
// Template parameters:
//   T_DIM, TOP_K, HIDDEN, INTERMEDIATE, NUM_EXPERTS

namespace kernel {

namespace fp8_fp4_mega_moe_v4_detail {

__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + __expf(-x));
}

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

} // namespace fp8_fp4_mega_moe_v4_detail

template <int T_DIM,
          int TOP_K,
          int HIDDEN,
          int INTERMEDIATE,
          int NUM_EXPERTS,
          int NUM_THREADS = 256>
__device__ __forceinline__ void fp8_fp4_mega_moe_v4_sm100_impl(
    void const *a_ptr,
    void const *w13_ptr,
    void const *w2_ptr,
    void const *topk_idx_ptr,
    void const *topk_w_ptr,
    void *y_ptr,
    float activation_clamp /* <= 0 disables */) {
  using bf16 = type::bfloat16_t;

  static_assert(INTERMEDIATE > 0, "INTERMEDIATE must be positive");
  static_assert(HIDDEN > 0, "HIDDEN must be positive");
  constexpr int TWO_I = 2 * INTERMEDIATE;

  bf16 const *__restrict__ A = static_cast<bf16 const *>(a_ptr);
  bf16 const *__restrict__ W13 = static_cast<bf16 const *>(w13_ptr);
  bf16 const *__restrict__ W2 = static_cast<bf16 const *>(w2_ptr);
  int64_t const *__restrict__ topk_idx =
      static_cast<int64_t const *>(topk_idx_ptr);
  float const *__restrict__ topk_w =
      static_cast<float const *>(topk_w_ptr);
  bf16 *__restrict__ Y = static_cast<bf16 *>(y_ptr);

  // Shared-memory layout (all fp32, packed):
  //   y_acc    [HIDDEN]        -- running per-token output (lives across
  //                                topk iterations).
  //   gate_buf [INTERMEDIATE]  -- post-clamp gate row.  After SwiGLU
  //                                this slot is repurposed as h_buf
  //                                (silu(gate)*up).
  //   up_buf   [INTERMEDIATE]  -- post-clamp up row.  Only used during
  //                                the SwiGLU stage; freed thereafter
  //                                but kept allocated for simplicity.
  // Total SMEM: (HIDDEN + 2 * INTERMEDIATE) * 4 bytes.
  // For V4-Flash test shapes this is well under 96KB.
  extern __shared__ char smem_raw[];
  float *y_acc = reinterpret_cast<float *>(smem_raw);
  float *gate_buf = y_acc + HIDDEN;
  float *up_buf = gate_buf + INTERMEDIATE;
  float *h_buf = gate_buf; // reused for h after SwiGLU

  // Zero y_acc.
  for (int i = threadIdx.x; i < HIDDEN; i += NUM_THREADS) {
    y_acc[i] = 0.0f;
  }
  __syncthreads();

  // Top-k loop.
#pragma unroll 1
  for (int k = 0; k < TOP_K; ++k) {
    int64_t const e = topk_idx[k];
    float const w = topk_w[k];
    if (e < 0 || e >= NUM_EXPERTS) {
      continue;
    }

    int64_t const w13_off = e * static_cast<int64_t>(TWO_I) *
                            static_cast<int64_t>(HIDDEN);
    int64_t const w2_off = e * static_cast<int64_t>(HIDDEN) *
                           static_cast<int64_t>(INTERMEDIATE);

    // --- L1 GEMM: l1[j] = sum_h A[h] * W13[e, j, h]  for j in [0, 2*I).
    // We compute gate (j in [0, I)) and up (j in [I, 2*I)) into
    // gate_buf / up_buf with one thread per output index, sequential K.
    // Naive O(2*I*H/T) per CTA.
    for (int j = threadIdx.x; j < TWO_I; j += NUM_THREADS) {
      int64_t const w_row = w13_off +
                            static_cast<int64_t>(j) *
                                static_cast<int64_t>(HIDDEN);
      float acc = 0.0f;
      for (int h = 0; h < HIDDEN; ++h) {
        float a = static_cast<float>(A[h]);
        float b = static_cast<float>(W13[w_row + h]);
        acc += a * b;
      }
      if (j < INTERMEDIATE) {
        // gate
        if (activation_clamp > 0.0f) {
          // gate clamp: gate <- min(gate, clamp)
          acc = acc > activation_clamp ? activation_clamp : acc;
        }
        gate_buf[j] = acc;
      } else {
        // up
        int jj = j - INTERMEDIATE;
        if (activation_clamp > 0.0f) {
          // up clip: [-clamp, +clamp]
          if (acc > activation_clamp) {
            acc = activation_clamp;
          } else if (acc < -activation_clamp) {
            acc = -activation_clamp;
          }
        }
        up_buf[jj] = acc;
      }
    }
    __syncthreads();

    // --- SwiGLU: h_buf[j] = silu(gate_buf[j]) * up_buf[j]
    for (int j = threadIdx.x; j < INTERMEDIATE; j += NUM_THREADS) {
      float g = gate_buf[j];
      float u = up_buf[j];
      h_buf[j] = fp8_fp4_mega_moe_v4_detail::silu(g) * u;
    }
    __syncthreads();

    // --- L2 GEMM: l2[h] = sum_j h_buf[j] * W2[e, h, j]   for h in [0, HIDDEN).
    // Accumulate into y_acc with router weight.
    for (int h = threadIdx.x; h < HIDDEN; h += NUM_THREADS) {
      int64_t const w_row = w2_off +
                            static_cast<int64_t>(h) *
                                static_cast<int64_t>(INTERMEDIATE);
      float acc = 0.0f;
      for (int j = 0; j < INTERMEDIATE; ++j) {
        float hh = h_buf[j];
        float bb = static_cast<float>(W2[w_row + j]);
        acc += hh * bb;
      }
      y_acc[h] += w * acc;
    }
    __syncthreads();
  }

  // Store y.
  for (int h = threadIdx.x; h < HIDDEN; h += NUM_THREADS) {
    Y[h] = static_cast<bf16>(y_acc[h]);
  }
}

} // namespace kernel
