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

// V4-Flash MTP input RMSNorm (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/fused_mtp_input_rmsnorm.md
// Triton ref: vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py
//
// Joint, per-token kernel that runs HC_MULT+1 RMS norms back-to-back:
//   slot 0          : enorm(inputs_embeds[t, :])  with pos==0 zero-mask
//   slot 1..HC_MULT : hnorm(prev_hidden[t, slot-1, :]), unconditional
//
// Naive design (Blackwell, single CTA, no TMA / no UMMA / no warp-spec):
//   * One CTA per token. Each CTA processes the (HC_MULT+1) norms
//     sequentially in a single thread block.
//   * NUM_THREADS = 256 (Blackwell default WORKER_NUM_THREADS).
//   * fp32 accumulator throughout, bf16 input/weight, bf16 store.
//   * Per-row reduction: partial sum-of-squares with thread-strided
//     loop, then in-warp shfl_xor reduction, then a single shared-
//     memory cross-warp combine (8 warps -> 1).
//   * NO shared-memory staging for the input/weight: we re-read the
//     bf16 row from global memory in the write pass. Correct and
//     simple. Performance is NOT a goal.
//   * Pos==0 masking is implemented as a "skip the squared accumulate
//     and write zero" branch on the enorm slot (equivalent to
//     torch.where(pos!=0, x, 0) followed by RMSNorm with weight, since
//     variance(0)+eps -> rsqrt(eps), and 0 * rsqrt(eps) * w = 0).
//
// Inputs (in TBGraph order, matches register_..._task() codegen):
//   input_ptrs[0] = inputs_embeds  (bf16)  [num_tokens, HIDDEN]
//   input_ptrs[1] = positions      (int64) [num_tokens]
//   input_ptrs[2] = prev_hidden    (bf16)  [num_tokens, HC_MULT, HIDDEN]
//   input_ptrs[3] = enorm_weight   (bf16)  [HIDDEN]
//   input_ptrs[4] = hnorm_weight   (bf16)  [HIDDEN]
//
// Outputs:
//   output_ptrs[0] = enorm_out  (bf16) [num_tokens, HIDDEN]
//   output_ptrs[1] = hnorm_out  (bf16) [num_tokens, HC_MULT, HIDDEN]
//
// task_metadata.request_id -> which token this CTA owns. Grid is
// (num_tokens, 1, 1). The runtime pre-offsets per-task pointers based
// on the TBGraph partition spec (dim-0 partition on the bf16 tensors
// and the int64 positions vector), so each task already sees its
// per-token slice; the kernel itself does NOT need to multiply by the
// token index.

namespace kernel {

namespace fused_mtp_input_rmsnorm_v4_detail {

template <int NUM_THREADS>
__device__ __forceinline__ float warp_block_reduce_sum(float val,
                                                       float *smem) {
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
  // Intra-warp reduction.
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
  // First warp combines the per-warp partial sums.
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

// Compute one RMSNorm row in fp32 with bf16 store.
//   x_in:        bf16 input row (HIDDEN contiguous)
//   w_in:        bf16 weight row (HIDDEN contiguous, shared across calls)
//   x_out:       bf16 output row (HIDDEN contiguous)
//   reduce_smem: cross-warp combine buffer (>= NUM_WARPS fp32 slots)
//   force_zero:  if true, write 0 and skip the multiply (pos==0 mask)
template <int HIDDEN, int NUM_THREADS>
__device__ __forceinline__ void
rmsnorm_row_naive(type::bfloat16_t const *__restrict__ x_in,
                  type::bfloat16_t const *__restrict__ w_in,
                  type::bfloat16_t *__restrict__ x_out,
                  float *reduce_smem,
                  float eps,
                  bool force_zero) {
  if (force_zero) {
    // pos==0 path: torch.where(pos==0, 0, x) makes variance == 0, so the
    // output is also 0 regardless of weight (rsqrt(eps) finite, times 0).
    // Skip the reduction entirely and just clear the row.
#pragma unroll
    for (int i = threadIdx.x; i < HIDDEN; i += NUM_THREADS) {
      x_out[i] = type::bfloat16_t(0.0f);
    }
    __syncthreads();
    return;
  }

  // Pass 1: partial sum-of-squares in fp32.
  float partial = 0.0f;
#pragma unroll
  for (int i = threadIdx.x; i < HIDDEN; i += NUM_THREADS) {
    float v = static_cast<float>(x_in[i]);
    partial += v * v;
  }
  float sumsq =
      warp_block_reduce_sum<NUM_THREADS>(partial, reduce_smem);

  float inv_rms = rsqrtf(sumsq / static_cast<float>(HIDDEN) + eps);

  // Pass 2: write back y = (x * inv_rms * w) cast to bf16. We re-read x
  // and w from gmem here -- this is the "naive" path; a perf pass
  // would stage them in smem during pass 1.
#pragma unroll
  for (int i = threadIdx.x; i < HIDDEN; i += NUM_THREADS) {
    float v = static_cast<float>(x_in[i]);
    float w = static_cast<float>(w_in[i]);
    x_out[i] = type::bfloat16_t(v * inv_rms * w);
  }
  __syncthreads();
}

} // namespace fused_mtp_input_rmsnorm_v4_detail

// Per-token NAIVE entry point. Called once per task by the generated
// _execute_task() dispatch (one task == one token).
//
// Pointers are pre-offset by the runtime so they already point at this
// token's row (the codegen emits dim-0 partition for the bf16 tensors
// and the int64 positions vector).
template <int HIDDEN, int HC_MULT, int NUM_THREADS = 256>
__device__ __forceinline__ void fused_mtp_input_rmsnorm_v4_sm100_impl(
    void const *inputs_embeds_ptr, // bf16  [HIDDEN]  (this token's row)
    void const *positions_ptr,     // int64 [1]       (this token's pos)
    void const *prev_hidden_ptr,   // bf16  [HC_MULT, HIDDEN]
    void const *enorm_weight_ptr,  // bf16  [HIDDEN]
    void const *hnorm_weight_ptr,  // bf16  [HIDDEN]
    void *enorm_out_ptr,           // bf16  [HIDDEN]
    void *hnorm_out_ptr,           // bf16  [HC_MULT, HIDDEN]
    float eps) {
  static_assert(HIDDEN > 0, "HIDDEN must be positive");
  static_assert(HC_MULT > 0, "HC_MULT must be positive");
  static_assert(NUM_THREADS > 0, "NUM_THREADS must be positive");
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a warp multiple");

  using bf16 = type::bfloat16_t;

  // Shared mem layout: NUM_WARPS fp32 slots for the cross-warp combine.
  // NUM_WARPS = 8 on Blackwell (256/32).
  extern __shared__ char smem[];
  float *reduce_smem = reinterpret_cast<float *>(smem);

  bf16 const *__restrict__ x_in =
      static_cast<bf16 const *>(inputs_embeds_ptr);
  bf16 const *__restrict__ prev_hidden =
      static_cast<bf16 const *>(prev_hidden_ptr);
  bf16 const *__restrict__ enorm_w =
      static_cast<bf16 const *>(enorm_weight_ptr);
  bf16 const *__restrict__ hnorm_w =
      static_cast<bf16 const *>(hnorm_weight_ptr);
  bf16 *__restrict__ enorm_out = static_cast<bf16 *>(enorm_out_ptr);
  bf16 *__restrict__ hnorm_out = static_cast<bf16 *>(hnorm_out_ptr);

  int64_t pos = *static_cast<int64_t const *>(positions_ptr);
  bool pos_zero = (pos == 0);

  // ---- enorm: inputs_embeds[this_tok, :] -> enorm_out[this_tok, :].
  // Per spec: when pos==0, mask the input to zero before the norm; the
  // helper handles that as a direct zero-write (variance=0 -> out=0).
  fused_mtp_input_rmsnorm_v4_detail::rmsnorm_row_naive<HIDDEN, NUM_THREADS>(
      x_in, enorm_w, enorm_out, reduce_smem, eps, pos_zero);

  // ---- hnorm: prev_hidden[this_tok, slot, :] for slot in [0, HC_MULT).
  // Unconditional (no pos mask). Process slots sequentially in this
  // CTA; performance is not a goal.
#pragma unroll
  for (int slot = 0; slot < HC_MULT; ++slot) {
    bf16 const *__restrict__ src = prev_hidden + slot * HIDDEN;
    bf16 *__restrict__ dst = hnorm_out + slot * HIDDEN;
    fused_mtp_input_rmsnorm_v4_detail::rmsnorm_row_naive<HIDDEN,
                                                         NUM_THREADS>(
        src, hnorm_w, dst, reduce_smem, eps, /*force_zero=*/false);
  }
}

} // namespace kernel
