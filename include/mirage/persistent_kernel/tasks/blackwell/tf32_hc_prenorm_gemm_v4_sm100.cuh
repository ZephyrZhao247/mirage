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

// V4-Flash tf32_hc_prenorm_gemm (NAIVE Blackwell SM100 impl).
//
// Spec: docs/mpk/deepseek_v4/vllm_kernels/tf32_hc_prenorm_gemm.md
//
// The DeepGEMM upstream uses TF32 UMMA (`SM100_MMA_TF32_TS`) for the
// GEMM and a separate fp32 sum-of-squares pass. The naive port collapses
// both into a single bf16->fp32->multiply-accumulate loop, producing the
// SAME outputs as hc_prenorm_gemm_*_tilelang at n_splits=1:
//   `gemm_out_mul: [1, T, HC_MULT3]` fp32
//   `gemm_out_sqrsum: [1, T]`        fp32
//
// Per the integration brief (user D5): focus on sm_100a (B200) only;
// SM90 (H100) is a separate DeepGEMM `sm90_tf32_hc_prenorm_gemm.cuh`
// upstream and a one-line pointer in the spec -- not implemented here.
//
// This file thin-wraps the shared hc_prenorm_gemm_v4_sm100_impl so the
// catalog/codegen can keep a distinct task name (`tf32_hc_prenorm_gemm_v4_sm100`)
// even though the numerical behavior at the naive level is byte-identical
// to the two TileLang variants.

namespace kernel {

template <int HC_MULT, int HIDDEN, int NUM_THREADS = 256>
__device__ __forceinline__ void tf32_hc_prenorm_gemm_v4_sm100_impl(
    void const *x_ptr,
    void const *fn_ptr,
    void *gemm_out_ptr,
    void *gemm_sqrsum_ptr) {
  hc_prenorm_gemm_v4_sm100_impl<HC_MULT, HIDDEN, NUM_THREADS>(
      x_ptr, fn_ptr, gemm_out_ptr, gemm_sqrsum_ptr);
}

} // namespace kernel
