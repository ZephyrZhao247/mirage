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

// =============================================================================
// compressor_state_update_sm100.cuh — DeepSeek V4-Flash Compressor state
// update kernel (sub-batch C2 of Wave-2).
//
// Migrated from the vLLM Triton kernel
// `deps/vllm/vllm/model_executor/layers/deepseek_compressor.py::
//  _save_partial_states_kernel` (lines 380-433).
//
// Per-token (one CTA per token) write into a ring buffer of "partial
// state" rows. Every row is `2 * HEAD_DIM` elements wide and packs
// [kv | score+ape] back-to-back along the last dim. The sibling
// `compressor_compress_sm100` task reads from this same ring every
// `compress_ratio` tokens to produce one compressed-KV entry.
//
// Math (v1 — coff = 1 in this task; the spec's coff>1 case is folded into
// HEAD_DIM at the Python catalog layer so the kernel sees a flat row):
//
//   slot_id = slot_mapping[t]                  // skip if slot_id < 0
//   row     = state_cache + slot_id * (2 * HEAD_DIM)
//   ape_row = positions[t] % COMPRESS_RATIO
//   row[                 0 :   HEAD_DIM] = kv[t]
//   row[          HEAD_DIM : 2*HEAD_DIM] = score[t] + ape[ape_row]
//
// Layout notes:
//   - All of kv, score, ape, state_cache are bf16. The Triton reference
//     reads them at full width and writes back at full width; we follow
//     that path with bf16 loads + an fp32 add + a bf16 store, which is
//     numerically equivalent for the simple addition here.
//   - `state_cache` is flat `[num_slots, 2 * HEAD_DIM]`. The ring index
//     (per spec: `block_idx = slot_id / block_size_state`,
//     `pos_in_block = slot_id % block_size_state`) is collapsed to a
//     single multiply because v1 uses `block_size_state == 1`. This
//     matches the Triton kernel's behaviour for that degenerate paging
//     parameter without complicating the kernel surface.
//
// blockIdx-agnostic: this CTA reads its `token_idx` exclusively from
// `task_desc->task_metadata.token_offset` (see runtime.cc dispatch).
// =============================================================================

#pragma once
#include "tasks/common/common_header.cuh"
#include <cstdint>

namespace kernel {

template <int HEAD_DIM,
          int COMPRESS_RATIO,
          bool OVERLAP,
          int NUM_THREADS,
          typename IN_T>
__device__ __forceinline__ void compressor_state_update_task_impl(
    void const *__restrict__ kv_ptr,         // bf16 [T, HEAD_DIM]
    void const *__restrict__ score_ptr,      // bf16 [T, HEAD_DIM]
    void const *__restrict__ ape_ptr,        // bf16 [COMPRESS_RATIO, HEAD_DIM]
    void const *__restrict__ positions_ptr,  // int32 [T]
    void const *__restrict__ slot_mapping_ptr, // int32 [T]
    void *__restrict__ state_cache_ptr,      // bf16 [num_slots, 2*HEAD_DIM] (in/out)
    int token_offset,
    int num_tokens_per_task,
    int num_tokens_total) {
  // Gate inactive lanes — the worker block is launched with
  // WORKER_NUM_THREADS lanes regardless of this kernel's needs.
  if (threadIdx.x >= NUM_THREADS) {
    return;
  }

  static_assert(HEAD_DIM > 0, "HEAD_DIM must be positive");
  static_assert(COMPRESS_RATIO > 0, "COMPRESS_RATIO must be positive");
  // OVERLAP is forwarded as a constexpr to keep the templated codegen
  // surface stable with the spec; v1's write logic is identical for both
  // overlap modes since the per-token write is determined entirely by
  // `slot_mapping[t]` upstream (the Python catalog layer computes the
  // overlap-aware slot id and hands it to this kernel). The flag is kept
  // for future use by the in-kernel slot-id derivation path.
  (void)OVERLAP;

  IN_T const *__restrict__ kv = static_cast<IN_T const *>(kv_ptr);
  IN_T const *__restrict__ score = static_cast<IN_T const *>(score_ptr);
  IN_T const *__restrict__ ape = static_cast<IN_T const *>(ape_ptr);
  int const *__restrict__ positions =
      static_cast<int const *>(positions_ptr);
  int const *__restrict__ slot_mapping =
      static_cast<int const *>(slot_mapping_ptr);
  IN_T *__restrict__ state_cache = static_cast<IN_T *>(state_cache_ptr);

  // Per-CTA loop (v1: num_tokens_per_task == 1).
  for (int local = 0; local < num_tokens_per_task; ++local) {
    int const t = token_offset + local;
    if (t >= num_tokens_total) {
      break;
    }

    int const slot_id = slot_mapping[t];
    // Skip padded / invalid slots — matches the Triton kernel's
    // `if slot_id < 0: return` guard at line 407-408.
    if (slot_id < 0) {
      continue;
    }

    int const pos = positions[t];
    // Python catalog passes `pos` directly; ape_row = pos % COMPRESS_RATIO.
    // Use a non-negative modulo (pos is the absolute position; always >= 0
    // in real V4-Flash inputs).
    int ape_row = pos % COMPRESS_RATIO;
    if (ape_row < 0) {
      ape_row += COMPRESS_RATIO;
    }

    IN_T const *__restrict__ kv_row = kv + (size_t)t * HEAD_DIM;
    IN_T const *__restrict__ score_row = score + (size_t)t * HEAD_DIM;
    IN_T const *__restrict__ ape_row_ptr = ape + (size_t)ape_row * HEAD_DIM;
    // Row width in the state cache is 2 * HEAD_DIM (kv | score+ape).
    constexpr int ROW_WIDTH = 2 * HEAD_DIM;
    IN_T *__restrict__ out_row =
        state_cache + (size_t)slot_id * ROW_WIDTH;

    // Strided cooperative copy of the kv half (write `kv_row` -> `out_row[0..HEAD_DIM)`).
    for (int i = (int)threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
      out_row[i] = kv_row[i];
    }

    // Strided cooperative `score + ape` write to the second half.
    for (int i = (int)threadIdx.x; i < HEAD_DIM; i += NUM_THREADS) {
      float s = static_cast<float>(score_row[i]);
      float a = static_cast<float>(ape_row_ptr[i]);
      float v = s + a;
      // bf16 store via the IN_T cast — matches the input dtype.
      out_row[HEAD_DIM + i] = static_cast<IN_T>(v);
    }
  }
}

} // namespace kernel
