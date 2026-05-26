/* Copyright 2025 Mirage Team
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
namespace kernel {

template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int I_STRIDE,
          int O_STRIDE,
          bool WITH_CLAMP = false>
__device__ __forceinline__ void silu_mul_task_impl(void const *input_ptr,
                                                   void *output_ptr,
                                                   int num_active_tokens,
                                                   float swiglu_limit = 0.0f) {
  // V4-Flash clamped SwiGLU: when WITH_CLAMP=true, the gate is clamped to
  // (-inf, L] (max-only, asymmetric) and the up is clamped to [-L, L] (full)
  // before silu(gate) * up. See model.py:596-606 in DeepSeek-V4-Flash.
  // When WITH_CLAMP=false the kernel is byte-identical to the V3 path.
  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_mul = static_cast<T const *>(input_ptr) + OUTPUT_SIZE;
  T *__restrict__ d_output = static_cast<T *>(output_ptr);

#pragma unroll
  for (int i = threadIdx.x; i < num_active_tokens * OUTPUT_SIZE;
       i += blockDim.x) {
    int batch_idx = i / OUTPUT_SIZE;
    int offset = i % OUTPUT_SIZE;
    float input_val = float(d_input[batch_idx * I_STRIDE + offset]);
    if (WITH_CLAMP) {
      // gate: asymmetric upper clamp only (model.py:602)
      input_val = fminf(input_val, swiglu_limit);
      float mul_val = float(d_mul[batch_idx * I_STRIDE + offset]);
      // up: full clamp to [-L, L] (model.py:601)
      mul_val = fminf(fmaxf(mul_val, -swiglu_limit), swiglu_limit);
      d_output[batch_idx * O_STRIDE + offset] =
          T(input_val / (1.0f + expf(-input_val))) * T(mul_val);
    } else {
      T mul_val = d_mul[batch_idx * I_STRIDE + offset];
      d_output[batch_idx * O_STRIDE + offset] =
          T(input_val / (1.0f + expf(-input_val))) * mul_val;
    }
  }
}

} // namespace kernel
