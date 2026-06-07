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

// V4-Flash topk_softplus_sqrt (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/topk_softplus_sqrt.md
// vLLM reference: csrc/moe/topk_softplus_sqrt_kernels.cu
// Model reference: deps/.../inference/model.py:565-584 (Gate.forward).
//
// Two USE_HASH template branches in a single kernel (matches vLLM's dispatch):
//
//   USE_HASH == false (scored branch, V4-Flash layers 3..42):
//     1. scores_unbiased[e] = sqrt(softplus(gating_output[t, e]))  in fp32
//     2. scores_for_choice[e] = scores_unbiased[e] + correction_bias[e]
//     3. iterate k_idx in [0, K): expert = argmax(scores_for_choice)
//        max_val_unbiased = scores_unbiased[expert]  (== max - bias)
//        topk_weights[t, k_idx] = max_val_unbiased
//        topk_indices[t, k_idx] = expert (or NUM_EXPERTS if out of range)
//        token_expert_indices[t, k_idx] = k_idx * M + t
//        scores_for_choice[expert] = -1e4   (mask for next iter)
//     4. if renormalize: weights *= scale / sum(weights); else *= scale.
//
//   USE_HASH == true (hash branch, V4-Flash layers 0..2):
//     1. expert_ids = tid2eid[input_ids[t], 0..K-1]  (precomputed lookup)
//     2. scores_unbiased[e] = sqrt(softplus(gating_output[t, e]))
//     3. for k_idx in [0, K): weight = scores_unbiased[expert_ids[k_idx]]
//        topk_weights[t, k_idx] = weight  (NO bias subtraction; bias is null)
//        topk_indices[t, k_idx] = expert_ids[k_idx]  (global id, no offset)
//        token_expert_indices is NOT written on the hash branch (matches
//        vLLM kernel's early-return).
//     4. if renormalize: weights *= scale / sum(weights); else *= scale.
//
// Numerical / fusion notes (mirrors the spec):
//   * softplus stability: x > 20 -> x; else log1p(exp(x)).  (beta=1)
//   * Bias semantics: bias added pre-selection, NOT included in weight.
//     The kernel's weight write uses scores_unbiased[expert] directly --
//     algebraically equivalent to (max + bias) - bias.
//   * tie-break: lower expert index wins (matches vLLM kernel; we do this
//     via argmax-with-min-index iteration order, not warp tie-break).
//   * NUM_EXPERTS sentinel: out-of-range experts get index NUM_EXPERTS.
//     V4-Flash uses start_expert=0, end_expert=NUM_EXPERTS, so this never
//     triggers in practice; we still emit it for spec correctness.
//
// Naive design (correctness only, no perf):
//   * One CTA per token. grid = (num_tokens, 1, 1).
//   * NUM_THREADS = 256 (Blackwell default WORKER_NUM_THREADS).
//   * Single thread (threadIdx.x == 0) does the per-token work; the
//     remaining 255 threads idle. With at most 256 experts and k=6 this
//     is well under 1 us per token even on B200; performance is not a goal.
//
// Tensor contract (matches the spec):
//   * gating_output:        fp32 [T, E]   (V4-Flash: fp32 from GateLinear)
//   * correction_bias:      fp32 [E]      (scored branch only; null otherwise)
//   * input_ids:            int32 [T]     (hash branch only; null otherwise)
//   * tid2eid:              int32 [vocab, K] (hash branch only)
//   * topk_weights:         fp32 [T, K]
//   * topk_indices:         int32 [T, K]
//   * token_expert_indices: int32 [T, K]  (scored branch only)
//
// The kernel sees ONE token's slice for [gating_output, topk_weights,
// topk_indices, token_expert_indices, input_ids]; the bias/tid2eid pointers
// are broadcast (no partition).

namespace kernel {

namespace topk_softplus_sqrt_v4_detail {

__device__ __forceinline__ float softplus_sqrt_stable(float x) {
  // softplus(x) = log1p(exp(x))  for |x| < 20
  // softplus(x) ~ x              for x > 20  (within fp32 epsilon)
  float sp;
  if (x > 20.0f) {
    sp = x;
  } else {
    sp = __logf(1.0f + __expf(x));
  }
  return __fsqrt_rn(fmaxf(sp, 0.0f));
}

} // namespace topk_softplus_sqrt_v4_detail

// USE_HASH == false branch: scored top-k argmax with bias.
//
// All template parameters are compile-time:
//   NUM_EXPERTS    : E (V4-Flash: 256)
//   TOPK           : k (V4-Flash: 6)
//   NUM_THREADS    : block dim (256)
template <int NUM_EXPERTS, int TOPK, int NUM_THREADS = 256>
__device__ __forceinline__ void topk_softplus_sqrt_v4_sm100_scored_impl(
    void const *gating_output_ptr,        // fp32 [E] (this token's row)
    void const *correction_bias_ptr,      // fp32 [E] (broadcast)
    void *topk_weights_ptr,               // fp32 [K] (this token)
    void *topk_indices_ptr,               // int32 [K] (this token)
    void *token_expert_indices_ptr,       // int32 [K] (this token)
    int token_idx,                        // = thread_row in the spec
    int num_tokens,                       // M  (for token_expert_indices)
    int start_expert,                     // V4-Flash: 0
    int end_expert,                       // V4-Flash: NUM_EXPERTS
    bool renormalize,                     // V4-Flash: true
    float routed_scaling_factor) {        // V4-Flash: 1.5
  static_assert(NUM_EXPERTS > 0, "NUM_EXPERTS must be positive");
  static_assert(TOPK > 0, "TOPK must be positive");
  static_assert(TOPK <= NUM_EXPERTS, "TOPK must be <= NUM_EXPERTS");
  static_assert(NUM_THREADS > 0, "NUM_THREADS must be positive");

  namespace dt = topk_softplus_sqrt_v4_detail;

  float const *__restrict__ gating =
      static_cast<float const *>(gating_output_ptr);
  float const *__restrict__ bias =
      static_cast<float const *>(correction_bias_ptr);
  float *__restrict__ weights = static_cast<float *>(topk_weights_ptr);
  int *__restrict__ indices = static_cast<int *>(topk_indices_ptr);
  int *__restrict__ src_rows = static_cast<int *>(token_expert_indices_ptr);

  // Single-thread per-token implementation (naive). The CTA has 256
  // threads; only thread 0 does the work for this token.
  if (threadIdx.x != 0) {
    return;
  }

  // Stack-resident scratch of size NUM_EXPERTS. V4-Flash NUM_EXPERTS=256
  // fp32 floats = 1 KiB per thread; fits within local-memory budget.
  float scores_unbiased[NUM_EXPERTS];
  float scores_for_choice[NUM_EXPERTS];

#pragma unroll 1
  for (int e = 0; e < NUM_EXPERTS; ++e) {
    float x = gating[e];
    float s = dt::softplus_sqrt_stable(x);
    scores_unbiased[e] = s;
    float b = (bias != nullptr) ? bias[e] : 0.0f;
    scores_for_choice[e] = s + b;
  }

  float selected_sum = 0.0f;
#pragma unroll 1
  for (int k_idx = 0; k_idx < TOPK; ++k_idx) {
    // Argmax with tie-break on lowest expert index (matches vLLM).
    int best_e = 0;
    float best_v = scores_for_choice[0];
#pragma unroll 1
    for (int e = 1; e < NUM_EXPERTS; ++e) {
      float v = scores_for_choice[e];
      if (v > best_v) {
        best_v = v;
        best_e = e;
      }
    }
    bool in_range = (best_e >= start_expert) && (best_e < end_expert);
    float weight = scores_unbiased[best_e];  // bias-stripped
    weights[k_idx] = weight;
    indices[k_idx] = in_range ? (best_e - start_expert) : NUM_EXPERTS;
    src_rows[k_idx] = k_idx * num_tokens + token_idx;
    if (renormalize) {
      selected_sum += weight;
    }
    // Mask out the winner so next iter can't pick it again.
    scores_for_choice[best_e] = -1.0e4f;
  }

  float scale;
  if (renormalize) {
    float denom = (selected_sum > 0.0f) ? selected_sum : 1.0f;
    scale = routed_scaling_factor / denom;
  } else {
    scale = routed_scaling_factor;
  }
#pragma unroll
  for (int k_idx = 0; k_idx < TOPK; ++k_idx) {
    weights[k_idx] *= scale;
  }
}

// USE_HASH == true branch: hash-driven topk (layers 0..num_hash_layers-1).
//
// tid2eid is dtype `IndType` (int32 in V4-Flash); we template it as int.
template <int NUM_EXPERTS, int TOPK, int NUM_THREADS = 256>
__device__ __forceinline__ void topk_softplus_sqrt_v4_sm100_hash_impl(
    void const *gating_output_ptr, // fp32 [E] (this token's row)
    void const *input_ids_ptr,     // int32 scalar (this token's input_id)
    void const *tid2eid_ptr,       // int32 [vocab, K] (broadcast)
    void *topk_weights_ptr,        // fp32 [K] (this token)
    void *topk_indices_ptr,        // int32 [K] (this token)
    bool renormalize,
    float routed_scaling_factor) {
  static_assert(NUM_EXPERTS > 0, "NUM_EXPERTS must be positive");
  static_assert(TOPK > 0, "TOPK must be positive");
  static_assert(NUM_THREADS > 0, "NUM_THREADS must be positive");

  namespace dt = topk_softplus_sqrt_v4_detail;

  float const *__restrict__ gating =
      static_cast<float const *>(gating_output_ptr);
  int const *__restrict__ input_ids =
      static_cast<int const *>(input_ids_ptr);
  int const *__restrict__ tid2eid = static_cast<int const *>(tid2eid_ptr);
  float *__restrict__ weights = static_cast<float *>(topk_weights_ptr);
  int *__restrict__ indices = static_cast<int *>(topk_indices_ptr);

  if (threadIdx.x != 0) {
    return;
  }

  int token_id = input_ids[0];
  int const *__restrict__ expert_row = tid2eid + token_id * TOPK;

  float selected_sum = 0.0f;
#pragma unroll
  for (int k_idx = 0; k_idx < TOPK; ++k_idx) {
    int expert = expert_row[k_idx];
    float weight = dt::softplus_sqrt_stable(gating[expert]);
    weights[k_idx] = weight;
    indices[k_idx] = expert;
    if (renormalize) {
      selected_sum += weight;
    }
  }

  float scale;
  if (renormalize) {
    float denom = (selected_sum > 0.0f) ? selected_sum : 1.0f;
    scale = routed_scaling_factor / denom;
  } else {
    scale = routed_scaling_factor;
  }
#pragma unroll
  for (int k_idx = 0; k_idx < TOPK; ++k_idx) {
    weights[k_idx] *= scale;
  }
}

} // namespace kernel
