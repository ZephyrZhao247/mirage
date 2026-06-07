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
#pragma once
#include "tasks/common/common_header.cuh"

// Naive V4-Flash ``apply_rotary_emb`` — GPT-J / interleaved RoPE.
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/apply_rotary_emb.md.
// Reference: deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:232-242
// and the Triton kernel at vllm/vllm_flash_attn/ops/triton/rotary.py.
//
// V4-Flash uses ``is_neox_style=False`` (interleaved / GPT-J style;
// vllm/models/deepseek_v4/nvidia/model.py:719): for each rotary pair
// ``(even, odd) = (x[..., 2i], x[..., 2i+1])`` we compute
//   out[..., 2i]   = even * cos[..., i] - odd * sin[..., i]
//   out[..., 2i+1] = even * sin[..., i] + odd * cos[..., i]
// Trailing lanes [rotary_dim : head_dim) are copied through unchanged.
//
// This kernel is the generic-vLLM-fallback RoPE (the V4 production path
// fuses RoPE into other kernels); it is included for catalog
// completeness and for tests that exercise the naive contract.
//
// Naive single-CTA partitioning:
//   - Grid: (num_rows, 1, 1) where num_rows = NUM_TOKENS * NUM_HEADS.
//     Each CTA rotates one (token, head) pair.
//   - Block: (256, 1, 1) (Blackwell default).
//   - Per-CTA work: plain loop over rotary_dim/2 pairs (or head_dim - rotary_dim
//     pass-through lanes), each thread strides by blockDim.x. Arithmetic
//     in fp32, stored back in bf16. No TMA, no UMMA, no warp specialization.
//
// Tensor layout assumed by the catalog wrapper:
//   x:   [num_rows, head_dim] bf16, contiguous (catalog flattens token*head).
//   cos: [num_rows, rotary_dim/2] bf16, contiguous — caller has already
//        gathered per-(token,head)-row cos values (typically: cos table
//        indexed by ``positions[token_row]`` and broadcast over heads).
//   sin: [num_rows, rotary_dim/2] bf16, contiguous — same layout as cos.
//   out: [num_rows, head_dim] bf16, contiguous.
//
// The codegen template baker (task_register.cc) sets HEAD_DIM and
// ROTARY_DIM at compile time from the catalog's compile() parameters.

namespace kernel {

template <typename T, int HEAD_DIM, int ROTARY_DIM>
__device__ __forceinline__ void apply_rotary_emb_v4_sm100_task_impl(
    void const *input_ptr,
    void const *cos_ptr,
    void const *sin_ptr,
    void *output_ptr) {
  static_assert(ROTARY_DIM % 2 == 0,
                "ROTARY_DIM must be even (rotary pairs)");
  static_assert(ROTARY_DIM <= HEAD_DIM,
                "ROTARY_DIM cannot exceed HEAD_DIM");

  T const *__restrict__ x = static_cast<T const *>(input_ptr);
  T const *__restrict__ cos = static_cast<T const *>(cos_ptr);
  T const *__restrict__ sin = static_cast<T const *>(sin_ptr);
  T *__restrict__ out = static_cast<T *>(output_ptr);

  constexpr int NUM_PAIRS = ROTARY_DIM / 2;

  // Each CTA rotates one (token, head) row. The catalog wires a grid of
  // (NUM_TOKENS * NUM_HEADS, 1, 1) CTAs and the task launcher dispatches
  // task->input_ptrs[0] / output_ptrs[0] pre-offset to the row this CTA
  // owns. Cos / sin are also gathered per-row by the catalog.
  //
  // Naive loop: each thread takes pair index ``p = threadIdx.x`` and
  // strides by blockDim.x.
#pragma unroll
  for (int p = threadIdx.x; p < NUM_PAIRS; p += blockDim.x) {
    int even_idx = 2 * p;
    int odd_idx = 2 * p + 1;

    float even = static_cast<float>(x[even_idx]);
    float odd = static_cast<float>(x[odd_idx]);
    float c = static_cast<float>(cos[p]);
    float s = static_cast<float>(sin[p]);

    float out_even = even * c - odd * s;
    float out_odd = even * s + odd * c;

    out[even_idx] = static_cast<T>(out_even);
    out[odd_idx] = static_cast<T>(out_odd);
  }

  // Pass-through copy for trailing lanes [ROTARY_DIM, HEAD_DIM).
  // V4-Flash sets rotary_dim == head_dim (qk_rope_head_dim=64; this whole
  // tensor IS the rope head), so this loop is typically empty — but we
  // keep it for the generic-fallback contract.
  if constexpr (ROTARY_DIM < HEAD_DIM) {
#pragma unroll
    for (int i = ROTARY_DIM + threadIdx.x; i < HEAD_DIM; i += blockDim.x) {
      out[i] = x[i];
    }
  }
}

} // namespace kernel
