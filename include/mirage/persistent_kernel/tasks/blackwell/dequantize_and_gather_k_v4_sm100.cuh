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
#include <cuda_fp8.h>

// V4-Flash dequantize_and_gather_k_kernel (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/dequantize_and_gather_k_kernel.md
// (Triton variant; CuteDSL sibling at
// `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` is the
// locked alternative -- NOT spec'd here per D2.)
//
// Inverse of quantize_and_insert_k_v4_sm100: dequantize FP8 NoPE (7 blocks
// of 64 lanes, UE8M0 scales) and copy bf16 RoPE verbatim.
//
// Naive design: one CTA per request (partitioned by dim 0 on `out`).
// Each CTA serially loops over its gather range and writes
// `gather_len` tokens to `out[batch_idx, offset:offset+gather_len, :512]`.
// `gather_lens` may be a null pointer -- then we gather all `seq_len`
// tokens (compressed-K call site).
//
// I/O layout (see spec):
//   - out         : bf16 [num_reqs, max_num_tokens, head_size=576]
//   - k_cache     : uint8 [num_blocks, block_stride] (block_stride bytes per
//                   paged block; 656 bytes/token packed inside)
//   - seq_lens    : int32 [num_reqs]
//   - gather_lens : int32 [num_reqs] OR null
//   - block_table : int32 [num_reqs, max_blocks_per_seq]
//   - offset, cache_block_size: scalars
//
// Kernel writes only the first 512 lanes (448 dequant + 64 bf16 copy);
// lanes [512, 576) of `out` are left untouched (matches the Triton
// reference).

namespace kernel {

template <int FP8_DIM = 448,
          int BF16_DIM = 64,
          int QUANT_BLOCK = 64,
          int N_QUANT_BLOCKS_REAL = 7,
          int OUTPUT_DIM = 512,
          int NUM_THREADS = 256>
__device__ __forceinline__ void dequantize_and_gather_k_v4_sm100_impl(
    void *out_ptr,                 // bf16 [num_reqs, max_num_tokens, 576] -- this batch's slice
    void const *k_cache_ptr,       // uint8 [num_blocks, block_stride]
    void const *seq_lens_ptr,      // int32 [1] -- this batch's seq_len
    void const *gather_lens_ptr,   // int32 [1] OR nullptr
    void const *block_table_ptr,   // int32 [max_blocks_per_seq] -- this batch's row
    int out_stride1,               // bytes between tokens in `out` (= head_size * 2)
    int block_stride,              // bytes per paged block
    int cache_block_size,          // tokens per paged block (64 SWA or 256 compressed)
    int offset,                    // output base-row offset (in tokens)
    int max_blocks_per_seq         // for bounds; not strictly needed beyond docs
) {
  static_assert(FP8_DIM == 448 && BF16_DIM == 64 && QUANT_BLOCK == 64 &&
                    N_QUANT_BLOCKS_REAL == 7 && OUTPUT_DIM == 512,
                "V4-Flash compressed-K cache layout locked.");
  static_assert(NUM_THREADS > 0 && NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a positive warp multiple");

  constexpr int SCALE_DIM = 8;
  constexpr int TOKEN_DATA_SIZE = FP8_DIM + 2 * BF16_DIM; // 576

  using bf16 = type::bfloat16_t;

  bf16 *__restrict__ out = static_cast<bf16 *>(out_ptr);
  unsigned char const *__restrict__ k_cache_bytes =
      static_cast<unsigned char const *>(k_cache_ptr);
  int const *__restrict__ seq_lens_in =
      static_cast<int const *>(seq_lens_ptr);
  int const *__restrict__ gather_lens_in =
      static_cast<int const *>(gather_lens_ptr);
  int const *__restrict__ block_table =
      static_cast<int const *>(block_table_ptr);

  int const seq_len = seq_lens_in[0];
  int const gather_len = (gather_lens_in != nullptr) ? gather_lens_in[0] : seq_len;
  int const start_pos = seq_len - gather_len;

  // Each CTA handles one batch row; serially scan its gather range.
  for (int i = 0; i < gather_len; ++i) {
    int pos = start_pos + i;
    int block_in_seq = pos / cache_block_size;
    int pos_in_block = pos % cache_block_size;
    int physical_block_id = block_table[block_in_seq];
    long long base = static_cast<long long>(physical_block_id) *
                     static_cast<long long>(block_stride);

    unsigned char const *token_data =
        k_cache_bytes + base +
        static_cast<long long>(pos_in_block) *
            static_cast<long long>(TOKEN_DATA_SIZE);
    unsigned char const *token_scale =
        k_cache_bytes + base +
        static_cast<long long>(cache_block_size) *
            static_cast<long long>(TOKEN_DATA_SIZE) +
        static_cast<long long>(pos_in_block) *
            static_cast<long long>(SCALE_DIM);

    // Destination row in `out`. out_stride1 is in BYTES; bf16 is 2 bytes.
    bf16 *out_row = reinterpret_cast<bf16 *>(
        reinterpret_cast<unsigned char *>(out) +
        static_cast<long long>(offset + i) *
            static_cast<long long>(out_stride1));

    // -------- Dequant 7 FP8 blocks of 64 lanes each --------
#pragma unroll 1
    for (int qb = 0; qb < N_QUANT_BLOCKS_REAL; ++qb) {
      unsigned char enc = token_scale[qb];
      float scale = exp2f(static_cast<float>(static_cast<int>(enc) - 127));
      // Lanes 0..63: each thread handles one lane.
      if (threadIdx.x < QUANT_BLOCK) {
        unsigned char byte = token_data[qb * QUANT_BLOCK + threadIdx.x];
        __nv_fp8_e4m3 packed;
        *reinterpret_cast<unsigned char *>(&packed) = byte;
        float v = static_cast<float>(packed);
        float dequant = v * scale;
        out_row[qb * QUANT_BLOCK + threadIdx.x] = static_cast<bf16>(dequant);
      }
      __syncthreads();
    }

    // -------- Copy bf16 RoPE (64 lanes, 128 bytes) --------
    if (threadIdx.x < BF16_DIM) {
      unsigned char b0 = token_data[FP8_DIM + 2 * threadIdx.x + 0];
      unsigned char b1 = token_data[FP8_DIM + 2 * threadIdx.x + 1];
      unsigned short bits =
          static_cast<unsigned short>(b0) |
          (static_cast<unsigned short>(b1) << 8);
      bf16 v = *reinterpret_cast<bf16 *>(&bits);
      out_row[FP8_DIM + threadIdx.x] = v;
    }
    __syncthreads();
  }
}

} // namespace kernel
