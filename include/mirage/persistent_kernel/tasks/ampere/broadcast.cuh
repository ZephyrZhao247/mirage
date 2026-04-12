/* Copyright 2026 CMU
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

#ifdef USE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

namespace kernel {

#ifdef USE_NVSHMEM

/**
 * NVSHMEM-based broadcast using put operations (root GPU only).
 * Root puts its input directly into each target GPU's output tensor.
 * Each (bid.x, bid.y, bid.z) has (num_gpus - 1) subtasks on root.
 *
 * input:  2D tensor (batch_size, hidden_size) on root
 * output: 2D tensor (batch_size, hidden_size) in NVSHMEM memory on target
 */
template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE>
__device__ __forceinline__ void
    nvshmem_broadcast_put(void *output_ptr,
                          void *input_ptr,
                          void *sig_addr,
                          size_t event_index,
                          int target_gpu_id,
                          int active_tokens) {
#pragma unroll
  for (int i = 0; i < active_tokens; i++) {
    nvshmemx_putmem_nbi_block(reinterpret_cast<char *>(output_ptr) +
                                  i * OUTPUT_STRIDE * sizeof(T),
                              reinterpret_cast<char *>(input_ptr) +
                                  i * OUTPUT_STRIDE * sizeof(T),
                              OUTPUT_SIZE * sizeof(T),
                              target_gpu_id);
  }

  nvshmem_quiet();
  __syncthreads();
  if (threadIdx.x == 0) {
    nvshmemx_signal_op(reinterpret_cast<uint64_t *>(sig_addr),
                       1,
                       NVSHMEM_SIGNAL_ADD,
                       target_gpu_id);
  }
}

/**
 * No-op broadcast receive task (non-root GPUs).
 * Data is already written directly by the root via NVSHMEM put.
 * This task exists solely as an event dependency node in the task graph.
 */
__device__ __forceinline__ void nvshmem_broadcast_recv() {
  // No-op
}

#endif // USE_NVSHMEM

} // namespace kernel
