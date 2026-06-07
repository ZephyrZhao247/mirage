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
#include <cstdint>

// V4-Flash prepare_megamoe_inputs (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/prepare_megamoe_inputs.md
// vLLM reference: vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py
//   (_prepare_megamoe_inputs_kernel, Triton @jit).
//
// Fuses three things into a single per-token kernel:
//   1. bf16 -> FP8 E4M3 quantization with per-group (GROUP_K=32) UE8M0
//      scales, walking the hidden vector in BLOCK_K=128 chunks. Per
//      BLOCK_K = 128 we have 4 groups of size 32 each, and we pack 4
//      UE8M0 bytes into one int32 word (one packed int32 per BLOCK_K
//      chunk per token).
//   2. int -> int64 cast on topk_ids.
//   3. fp32 byte-copy on topk_weights.
//
// UE8M0 packing convention (matches the Triton reference exactly):
//   amax_group  = max(|hidden[group]|, eps=1e-4)
//   scale       = amax_group / 448.0          (fp8_e4m3_max = 448)
//   exp_bits    = ((scale.view(int32) >> 23) & 0xFF)
//                  + ((scale.view(int32) & 0x7FFFFF) != 0)
//   exp_bits    = clamp(exp_bits, 1, 254)
//   rounded_scale = (exp_bits << 23).view(float)    (= 2^k for some k)
//   fp8[t, h]   = (hidden[t, h] / rounded_scale).to(fp8_e4m3)
//   packed_int32 = exp_bits[0] | (exp_bits[1] << 8) | (exp_bits[2] << 16)
//                                                    | (exp_bits[3] << 24)
//
// The topk repack runs on EVERY token (we always include the kb==0 work
// in our per-token CTA layout -- see grid below).
//
// Naive design (correctness only, no perf):
//   * One CTA per token. grid = (num_tokens, 1, 1).
//   * NUM_THREADS = 256 (Blackwell default WORKER_NUM_THREADS).
//   * The CTA iterates K-chunks (NUM_BLOCKS = HIDDEN_SIZE / BLOCK_K)
//     sequentially. Inside each chunk a single thread (threadIdx.x == 0)
//     computes the 4 group absmax + scale + pack; threads 0..BLOCK_K-1
//     each emit one fp8 element with the just-computed group scale.
//     Per-thread fp8 emit uses the same rounded_scale shared via
//     __syncthreads().
//   * Topk repack: thread 0 of each CTA also copies the topk_ids
//     (int -> int64) and topk_weights (fp32 copy) for this token.
//
// Tensor contract (matches the spec):
//   * hidden_states:    bf16 [T, H]   (H = HIDDEN_SIZE, multiple of BLOCK_K=128)
//   * topk_ids:         int32 [T, K]  (the spec accepts int32 or int64; we
//                                       fix the input to int32 in the naive
//                                       port and always upcast to int64)
//   * topk_weights:     fp32 [T, K]
//   * x_fp8 (out):      fp8 e4m3 [T, H]
//   * x_sf  (out):      int32 [T, H/BLOCK_K]   (packed UE8M0)
//   * topk_idx_out:     int64 [T, K]
//   * topk_weights_out: fp32 [T, K]
//
// The TBGraph partitions all token-keyed tensors on dim 0. The runtime
// preoffsets per-token slices.

namespace kernel {

namespace prepare_megamoe_inputs_v4_detail {

__device__ __forceinline__ uint32_t ue8m0_exponent(float scale) {
  // Reference Triton path (round up to next power of two when mantissa != 0):
  //   bits = scale.view(int32)
  //   exp  = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0)
  //   exp  = clamp(exp, 1, 254)
  // Match this bit-for-bit so the packed int32 layout DeepGEMM reads is
  // identical.
  uint32_t bits = __float_as_uint(scale);
  uint32_t exp = (bits >> 23) & 0xFFu;
  if ((bits & 0x7FFFFFu) != 0u) {
    exp += 1u;
  }
  if (exp < 1u) {
    exp = 1u;
  }
  if (exp > 254u) {
    exp = 254u;
  }
  return exp;
}

__device__ __forceinline__ float exponent_to_scale(uint32_t exp_bits) {
  // 2^(exp - 127) as a float, via direct exponent bit-pack.
  uint32_t bits = (exp_bits & 0xFFu) << 23;
  return __uint_as_float(bits);
}

} // namespace prepare_megamoe_inputs_v4_detail

// HIDDEN_SIZE must be a multiple of BLOCK_K (= 128 by spec).
// BLOCK_K must be a multiple of GROUP_K (= 32 by spec); the spec hard-codes
// BLOCK_K=128, GROUP_K=32 (4 groups per block).
template <
    int HIDDEN_SIZE,
    int TOPK,
    int BLOCK_K = 128,
    int GROUP_K = 32,
    int NUM_THREADS = 256>
__device__ __forceinline__ void prepare_megamoe_inputs_v4_sm100_impl(
    void const *hidden_states_ptr,   // bf16 [HIDDEN_SIZE]  (this token)
    void const *topk_ids_ptr,        // int32 [TOPK]        (this token)
    void const *topk_weights_ptr,    // fp32 [TOPK]         (this token)
    void *x_fp8_ptr,                 // fp8_e4m3 [HIDDEN_SIZE] (this token)
    void *x_sf_ptr,                  // int32 [HIDDEN_SIZE / BLOCK_K] (this token)
    void *topk_idx_out_ptr,          // int64 [TOPK]        (this token)
    void *topk_weights_out_ptr) {    // fp32 [TOPK]         (this token)
  static_assert(HIDDEN_SIZE > 0, "HIDDEN_SIZE must be positive");
  static_assert(BLOCK_K > 0, "BLOCK_K must be positive");
  static_assert(GROUP_K > 0, "GROUP_K must be positive");
  static_assert(BLOCK_K % GROUP_K == 0, "BLOCK_K must be a multiple of GROUP_K");
  static_assert(HIDDEN_SIZE % BLOCK_K == 0,
                "HIDDEN_SIZE must be a multiple of BLOCK_K");
  static_assert(NUM_THREADS > 0, "NUM_THREADS must be positive");
  static_assert(TOPK > 0, "TOPK must be positive");

  using bf16 = type::bfloat16_t;
  namespace dt = prepare_megamoe_inputs_v4_detail;

  constexpr int NUM_GROUPS_PER_BLOCK = BLOCK_K / GROUP_K;     // 4
  constexpr int NUM_BLOCKS = HIDDEN_SIZE / BLOCK_K;
  constexpr float FP8_E4M3_MAX = 448.0f;
  constexpr float EPS = 1e-4f;

  bf16 const *__restrict__ hidden =
      static_cast<bf16 const *>(hidden_states_ptr);
  int const *__restrict__ topk_ids =
      static_cast<int const *>(topk_ids_ptr);
  float const *__restrict__ topk_weights =
      static_cast<float const *>(topk_weights_ptr);
  __nv_fp8_e4m3 *__restrict__ x_fp8 =
      static_cast<__nv_fp8_e4m3 *>(x_fp8_ptr);
  int32_t *__restrict__ x_sf = static_cast<int32_t *>(x_sf_ptr);
  int64_t *__restrict__ topk_idx_out =
      static_cast<int64_t *>(topk_idx_out_ptr);
  float *__restrict__ topk_weights_out =
      static_cast<float *>(topk_weights_out_ptr);

  // Shared scratch for the 4 inv_scales of the current block (broadcast
  // from thread 0 to all threads that emit fp8 values).
  __shared__ float inv_scale_smem[NUM_GROUPS_PER_BLOCK];

#pragma unroll 1
  for (int kb = 0; kb < NUM_BLOCKS; ++kb) {
    int base = kb * BLOCK_K;
    if (threadIdx.x == 0) {
      uint32_t packed = 0u;
#pragma unroll
      for (int g = 0; g < NUM_GROUPS_PER_BLOCK; ++g) {
        float amax = 0.0f;
#pragma unroll
        for (int h = 0; h < GROUP_K; ++h) {
          float v = static_cast<float>(hidden[base + g * GROUP_K + h]);
          float a = fabsf(v);
          if (a > amax) {
            amax = a;
          }
        }
        amax = fmaxf(amax, EPS);
        float scale = amax / FP8_E4M3_MAX;
        uint32_t exp_bits = dt::ue8m0_exponent(scale);
        float rounded_scale = dt::exponent_to_scale(exp_bits);
        inv_scale_smem[g] = 1.0f / rounded_scale;
        packed |= (exp_bits & 0xFFu) << (g * 8);
      }
      x_sf[kb] = static_cast<int32_t>(packed);
    }
    __syncthreads();

    // Thread-strided emission of the BLOCK_K fp8 elements for this block.
    for (int i = threadIdx.x; i < BLOCK_K; i += NUM_THREADS) {
      int g = i / GROUP_K;
      float v = static_cast<float>(hidden[base + i]);
      float scaled = v * inv_scale_smem[g];
      x_fp8[base + i] = static_cast<__nv_fp8_e4m3>(scaled);
    }
    __syncthreads();
  }

  // Topk repack (int->int64, fp32 copy). Runs in every CTA.
  for (int j = threadIdx.x; j < TOPK; j += NUM_THREADS) {
    int eid = topk_ids[j];
    topk_idx_out[j] = static_cast<int64_t>(eid);
    topk_weights_out[j] = topk_weights[j];
  }
}

} // namespace kernel
