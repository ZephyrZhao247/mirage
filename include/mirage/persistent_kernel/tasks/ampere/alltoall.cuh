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
 * NVSHMEM-based alltoall using put operations.
 * GPU i sends input[target_gpu_id, :, :] to output[MY_GPU_ID, :, :] on
 * target_gpu_id. Each (bid.x, bid.y, bid.z) has (num_gpus - 1) subtasks.
 *
 * input:  3D tensor (world_size, batch_size, hidden_size)
 * output: 3D tensor (world_size, batch_size, hidden_size) in NVSHMEM memory
 *
 * The local chunk (input[my_id] -> output[my_id]) is handled separately
 * via an identity task registered from the Python layer.
 */
template <typename T,
          int NUM_GPUS,
          int MY_GPU_ID,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE>
__device__ __forceinline__ void
    nvshmem_alltoall_put(void *output_ptr,
                         void *input_ptr,
                         void *sig_addr,
                         size_t event_index,
                         int target_gpu_id,
                         int active_tokens) {
  size_t chunk_stride = BATCH_SIZE * OUTPUT_STRIDE * sizeof(T);
  // Source: input[target_gpu_id, :, :] on this GPU (chunk destined for target)
  char *src_base =
      reinterpret_cast<char *>(input_ptr) + target_gpu_id * chunk_stride;
  // Dest: output[MY_GPU_ID, :, :] on target_gpu_id
  // The output base offset for my_gpu_id slot is already applied by runtime.cc.
  char *dst_base = reinterpret_cast<char *>(output_ptr);

#pragma unroll
  for (int i = 0; i < active_tokens; i++) {
    nvshmemx_putmem_nbi_block(dst_base + i * OUTPUT_STRIDE * sizeof(T),
                              src_base + i * OUTPUT_STRIDE * sizeof(T),
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

#endif // USE_NVSHMEM

} // namespace kernel
