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

// V4-Flash fp8_fp4_paged_mqa_logits (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/fp8_fp4_paged_mqa_logits.md
//
// FP8 path only in the naive port (MXFP4 variant would require block-
// scaled MMA-like dequant; we use the FP8 path matching the
// `use_fp4_cache=False` branch of DeepGEMM).
//
// Computes MQA logits for ONE query atom (bb, nn) — i.e., one row of
// the output `[B * NEXT_N, MAX_MODEL_LEN]` logits matrix. The CTA
// iterates over all kv_pos in [0, MAX_MODEL_LEN), checking the mask
// `kv_pos < context_len`, and writes:
//
//   logits[kv_pos] = sum_h weights[h] * (q_fp32[h] . k_fp32(kv_pos))
//
//   where  q_fp32[h, d] = (fp8 -> fp32) q[h, d]
//          k_fp32[d]    = (fp8 -> fp32) k_packed[block_idx_phys, slot, 0, d] * k_scale
//          block_idx_phys = block_tables[block_idx_logical]
//                          (block_idx_logical = kv_pos / BLOCK_SIZE)
//          slot           = kv_pos % BLOCK_SIZE
//          k_scale        = *(float*)&kv_cache[..., HEAD_DIM:HEAD_DIM+4]
//
// Naive design (correctness only, no perf):
//   * One CTA per query atom (bb, nn). grid = (B * NEXT_N, 1, 1).
//     The catalog preoffsets `q_ptr`, `weights_ptr`, `block_table_ptr`,
//     `context_len_ptr`, and `logits_out_ptr` to this query atom.
//   * Each CTA loops over kv_pos in [0, MAX_MODEL_LEN). For each
//     unmasked kv_pos the threads of the block cooperatively compute
//     the MQA reduction (sum over (h, d) of N_HEADS*HEAD_DIM elements).
//     Masked-out slots are left untouched (clean_logits=False).
//   * NUM_THREADS = 256 (Blackwell default).
//   * No TMA / no UMMA / no warp-spec / no scheduler.
//
// Tensor layouts (this CTA's view; the catalog preoffsets where noted):
//   q_in        : fp8   [N_HEADS, HEAD_DIM]       (this query atom)
//   kv_cache    : uint8 [num_blocks, BLOCK_SIZE, 1, KV_HEAD_WIDTH]
//                       (FULL buffer; kernel does block-table lookup)
//   weights     : fp32  [N_HEADS]                 (this query atom)
//   block_table : int32 [MAX_BLOCKS]              (this batch row, preoffset)
//   context_len : int32 [1] scalar                (this (bb, nn), preoffset)
//   logits_out  : fp32  [MAX_MODEL_LEN]           (this query atom's row)
//
// Template params:
//   N_HEADS = 64, HEAD_DIM = 128, BLOCK_SIZE = paged block size,
//   KV_HEAD_WIDTH = HEAD_DIM + 4 (per-token fp8 + fp32 scale).
//
// Runtime param: `max_model_len` (int) — extent of the kv_pos loop.

namespace kernel {

namespace fp8_fp4_paged_mqa_logits_v4_detail {

template <int NUM_THREADS>
__device__ __forceinline__ float
warp_block_reduce_sum_paged(float val, float *smem) {
  constexpr int NUM_WARPS = NUM_THREADS / NUM_THREADS_PER_WARP;
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

} // namespace fp8_fp4_paged_mqa_logits_v4_detail

template <int N_HEADS,
          int HEAD_DIM,
          int BLOCK_SIZE,
          int KV_HEAD_WIDTH,
          int NUM_THREADS = 256>
__device__ __forceinline__ void fp8_fp4_paged_mqa_logits_v4_sm100_impl(
    void const *q_in_ptr,        // fp8  [N_HEADS, HEAD_DIM]
    void const *kv_cache_ptr,    // uint8 [num_blocks, BLOCK_SIZE, 1, KV_HEAD_WIDTH]
    void const *weights_ptr,     // fp32 [N_HEADS]
    void const *block_table_ptr, // int32 [MAX_BLOCKS]
    void const *context_len_ptr, // int32 [1]
    void *logits_out_ptr,        // fp32 [MAX_MODEL_LEN]
    int max_model_len) {
  static_assert(N_HEADS > 0, "N_HEADS must be positive");
  static_assert(HEAD_DIM > 0, "HEAD_DIM must be positive");
  static_assert(BLOCK_SIZE > 0, "BLOCK_SIZE must be positive");
  static_assert(KV_HEAD_WIDTH >= HEAD_DIM + 4,
                "KV_HEAD_WIDTH must accommodate HEAD_DIM fp8 + 4-byte fp32 scale");
  static_assert(NUM_THREADS > 0, "NUM_THREADS must be positive");

  using fp8 = __nv_fp8_e4m3;

  fp8 const *__restrict__ q = static_cast<fp8 const *>(q_in_ptr);
  uint8_t const *__restrict__ kv_cache =
      static_cast<uint8_t const *>(kv_cache_ptr);
  float const *__restrict__ w = static_cast<float const *>(weights_ptr);
  int32_t const *__restrict__ block_table =
      static_cast<int32_t const *>(block_table_ptr);
  int32_t const *__restrict__ context_len_arr =
      static_cast<int32_t const *>(context_len_ptr);
  float *__restrict__ logits_out = static_cast<float *>(logits_out_ptr);

  int const ctx = context_len_arr[0];

  extern __shared__ char smem_raw[];
  float *reduce_buf = reinterpret_cast<float *>(smem_raw);

  for (int kv_pos = 0; kv_pos < max_model_len; ++kv_pos) {
    if (kv_pos >= ctx) {
      // clean_logits=False: leave the slot untouched.
      continue;
    }

    int const block_idx_logical = kv_pos / BLOCK_SIZE;
    int const slot = kv_pos % BLOCK_SIZE;
    int const block_idx_phys = block_table[block_idx_logical];

    // Row offset within kv_cache (bytes):
    //   ((block * BLOCK_SIZE + slot) * 1 + 0) * KV_HEAD_WIDTH
    size_t k_row_byte_offset =
        static_cast<size_t>(block_idx_phys) * BLOCK_SIZE * KV_HEAD_WIDTH +
        static_cast<size_t>(slot) * KV_HEAD_WIDTH;
    uint8_t const *__restrict__ k_row_bytes = kv_cache + k_row_byte_offset;
    fp8 const *__restrict__ k_fp8 =
        reinterpret_cast<fp8 const *>(k_row_bytes);
    float const k_scale =
        *reinterpret_cast<float const *>(k_row_bytes + HEAD_DIM);

    // Block-strided reduction over (h, d).
    float local = 0.0f;
    int const total = N_HEADS * HEAD_DIM;
    for (int idx = threadIdx.x; idx < total; idx += NUM_THREADS) {
      int h = idx / HEAD_DIM;
      int d = idx - h * HEAD_DIM;
      float qv = static_cast<float>(q[h * HEAD_DIM + d]);
      float kv_v = static_cast<float>(k_fp8[d]) * k_scale;
      local += w[h] * qv * kv_v;
    }
    float reduced = fp8_fp4_paged_mqa_logits_v4_detail::
        warp_block_reduce_sum_paged<NUM_THREADS>(local, reduce_buf);

    if (threadIdx.x == 0) {
      logits_out[kv_pos] = reduced;
    }
    __syncthreads();
  }
}

} // namespace kernel
