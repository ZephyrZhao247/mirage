/* Copyright 2026 Mirage Team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once
#include "hc_prenorm_gemm_v4_sm100.cuh"

// V4-Flash hc_prenorm_gemm_block_m (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/hc_prenorm_gemm_block_m_tilelang.md
//
// The TileLang block_m variant only differs from hc_prenorm_gemm by its
// M-tiling strategy (block_m=2 vs per-token CTA). Outputs are IDENTICAL
// at n_splits=1: `gemm_out_mul: [1, T, HC_MULT3]` fp32 and
// `gemm_out_sqrsum: [1, T]` fp32.
//
// For the naive port both variants share the SAME __device__ impl and
// the SAME per-token CTA partition. This file exists only to give the
// block_m variant its own __device__ entry point so the codegen can
// thread the variant-specific task name through to the runtime; the
// actual work is delegated to hc_prenorm_gemm_v4_sm100_impl.

namespace kernel {

template <int HC_MULT, int HIDDEN, int NUM_THREADS = 256>
__device__ __forceinline__ void hc_prenorm_gemm_block_m_v4_sm100_impl(
    void const *x_ptr,
    void const *fn_ptr,
    void *gemm_out_ptr,
    void *gemm_sqrsum_ptr) {
  hc_prenorm_gemm_v4_sm100_impl<HC_MULT, HIDDEN, NUM_THREADS>(
      x_ptr, fn_ptr, gemm_out_ptr, gemm_sqrsum_ptr);
}

} // namespace kernel
