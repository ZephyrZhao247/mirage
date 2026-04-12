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
#include "device_host/nvshmem_types.h"
#include "tasks/common/common_header.cuh"

#ifdef USE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

namespace kernel {

#ifdef USE_NVSHMEM

/**
 * NVSHMEM tile-based allgather for Hopper+ (SM >= 90).
 * Each PE contributes its local (BATCH_SIZE, OUTPUT_SIZE) tile and the result
 * is gathered into a (NUM_GPUS * BATCH_SIZE, OUTPUT_SIZE) buffer on all PEs.
 *
 * input:  per-PE local data, shape (BATCH_SIZE, OUTPUT_SIZE), stride OUTPUT_STRIDE
 * output: gathered data, shape (NUM_GPUS * BATCH_SIZE, OUTPUT_SIZE), stride OUTPUT_STRIDE
 */
template <typename T,
          int NUM_GPUS,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int OUTPUT_STRIDE>
__device__ __forceinline__ void nvshmem_tile_allgather(void *input_ptr,
                                                       void *output_ptr,
                                                       void *_teams,
                                                       int task_offset,
                                                       int active_tokens) {
  using c_hidden = ConstInt<OUTPUT_STRIDE>;
  using c_output = ConstInt<OUTPUT_SIZE>;
  using c_1 = ConstInt<1>;

  // Source tile: (OUTPUT_SIZE, active_tokens) with stride (1, OUTPUT_STRIDE)
  auto src_shape =
      nvshmemx::make_shape<c_output, int>(c_output{}, active_tokens);
  auto src_stride = nvshmemx::make_stride<c_1, c_hidden>(c_1{}, c_hidden{});
  auto src_layout = nvshmemx::make_layout(src_shape, src_stride);
  auto src_tensor = nvshmemx::Tensor<T, decltype(src_layout)>(
      reinterpret_cast<T *>(input_ptr), src_layout);

  // Destination tile: (OUTPUT_SIZE, active_tokens * NUM_GPUS) with stride (1, OUTPUT_STRIDE)
  // The allgather concatenates along the batch (rows) dimension
  auto dst_shape =
      nvshmemx::make_shape<c_output, int>(c_output{}, active_tokens * NUM_GPUS);
  auto dst_stride = nvshmemx::make_stride<c_1, c_hidden>(c_1{}, c_hidden{});
  auto dst_layout = nvshmemx::make_layout(dst_shape, dst_stride);
  auto dst_tensor = nvshmemx::Tensor<T, decltype(dst_layout)>(
      reinterpret_cast<T *>(output_ptr), dst_layout);

  nvshmem_team_t *teams = reinterpret_cast<nvshmem_team_t *>(_teams);

  struct empty {};
  nvshmemx::tile_allgather_block<
      decltype(src_tensor),
      decltype(dst_tensor),
      empty,
      nvshmemx::tile_coll_algo_t::NVLS_ONE_SHOT_PUSH_NBI>(
      teams[task_offset], src_tensor, dst_tensor, empty{}, empty{}, 0, 0);
}

#endif // USE_NVSHMEM

} // namespace kernel
