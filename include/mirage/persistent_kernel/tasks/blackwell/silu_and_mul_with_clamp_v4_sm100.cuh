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

// V4-Flash silu_and_mul_with_clamp (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/silu_and_mul_with_clamp.md
// vLLM reference: csrc/libtorch_stable/activation_kernels.cu (silu_and_mul_clamp).
// Model reference: deps/deepseek_v4/.../inference/model.py:596-606 (Expert.forward).
//
// Fused element-wise op: for each (token t, channel h):
//   gate = input[t, h];   up = input[t, d + h];
//   gate_c = min(gate, limit);
//   up_c   = clamp(up, -limit, limit);
//   out[t, h] = (silu(gate_c) * up_c) cast to bf16
// where silu(x) = x / (1 + exp(-x)) in fp32.
//
// Naive design (correctness only, no perf):
//   * One CTA per token-tile slab. The runtime preoffsets pointers so each
//     CTA sees its own slab of "intermediate_size" elements; we just walk
//     [0, INTERMEDIATE_SIZE) with threadIdx.x stride.
//   * NUM_THREADS = 256 (Blackwell default WORKER_NUM_THREADS).
//   * fp32 internal math. bf16 in, bf16 out (matches input.dtype).
//   * No vectorized loads / no 256b loads / no warp-spec.
//
// Tensor contract (matches the spec):
//   * input:  bf16 [T, 2 * INTERMEDIATE_SIZE]  (gate || up halved per row)
//   * out:    bf16 [T, INTERMEDIATE_SIZE]      (silu(clamp(gate, max=L)) *
//                                               clamp(up, -L, L))
//   * limit:  fp32 scalar (V4-Flash: 10.0)
//
// The TBGraph partitions input + out on the token dim (dim 0). The runtime
// codegen preoffsets the bf16 per-token slice; the kernel sees one token's
// flattened gateup row (length 2*INTERMEDIATE_SIZE) and writes the
// corresponding INTERMEDIATE_SIZE-long out row.

namespace kernel {

template <int INTERMEDIATE_SIZE, int NUM_THREADS = 256>
__device__ __forceinline__ void silu_and_mul_with_clamp_v4_sm100_impl(
    void const *gateup_ptr, // bf16 [2 * INTERMEDIATE_SIZE] (this token's row)
    void *out_ptr,          // bf16 [INTERMEDIATE_SIZE]     (this token's row)
    float limit) {
  static_assert(INTERMEDIATE_SIZE > 0, "INTERMEDIATE_SIZE must be positive");
  static_assert(NUM_THREADS > 0, "NUM_THREADS must be positive");

  using bf16 = type::bfloat16_t;
  bf16 const *__restrict__ gateup = static_cast<bf16 const *>(gateup_ptr);
  bf16 *__restrict__ out = static_cast<bf16 *>(out_ptr);

  bf16 const *__restrict__ gate_ptr = gateup;
  bf16 const *__restrict__ up_ptr = gateup + INTERMEDIATE_SIZE;

  for (int h = threadIdx.x; h < INTERMEDIATE_SIZE; h += NUM_THREADS) {
    float gate = static_cast<float>(gate_ptr[h]);
    float up = static_cast<float>(up_ptr[h]);
    // One-sided clamp on gate; two-sided clamp on up.
    float gate_c = fminf(gate, limit);
    float up_c = fmaxf(fminf(up, limit), -limit);
    // SiLU via x / (1 + exp(-x)) -- algebraically identical to x * sigmoid(x);
    // the form used here keeps the expf argument positive-bounded (since
    // gate_c <= limit), so no overflow.
    float silu = gate_c / (1.0f + __expf(-gate_c));
    float result = silu * up_c;
    out[h] = static_cast<bf16>(result);
  }
}

} // namespace kernel
