/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 *
 * Standalone driver for kernel::fp8_gemm_dense_smallm::
 * fp8_gemm_dense_smallm_sm100_task_impl<BN=128, NS=3> — the EXACT template
 * instantiation the MPK codegen emits (see src/kernel/task_register.cc
 * register_fp8_gemm_dense_variant: `<128, 3>`). It lets us drive the kernel
 * directly with controllable per-128x128-block weight scales to isolate the
 * scale-handling defect that real DeepSeek weights expose.
 *
 * A: FP8 e4m3 [M, K] row-major, 1x128-group activation scale sa[M, K/128].
 * B: FP8 e4m3 [N, K] row-major, 128x128-block weight scale sb[N/128, K/128].
 * C: BF16     [M, N] row-major.
 */

#include "blackwell/fp8_gemm_dense_smallm_sm100.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace {

void check_driver(CUresult r, char const *what) {
  if (r == CUDA_SUCCESS) {
    return;
  }
  char const *en = nullptr;
  char const *es = nullptr;
  cuGetErrorName(r, &en);
  cuGetErrorString(r, &es);
  TORCH_CHECK(false, what, " failed: ", (en ? en : "?"), ": ", (es ? es : "?"));
}

// 2D TMA descriptor, row-major source, 128B swizzle. gmem layout is
// (outer_dim, inner_dim) with inner contiguous; box = (box_outer, box_inner).
CUtensorMap make_desc(void *ptr,
                      int gmem_outer,
                      int gmem_inner,
                      int box_outer,
                      int box_inner) {
  CUtensorMap desc{};
  // tensor dims are listed inner-first for cuTensorMapEncodeTiled.
  cuuint64_t const gdims[2] = {static_cast<cuuint64_t>(gmem_inner),
                               static_cast<cuuint64_t>(gmem_outer)};
  // global stride (bytes) of the OUTER dim; inner stride is implicit (1 elem).
  cuuint64_t const gstride[1] = {static_cast<cuuint64_t>(gmem_inner) *
                                 sizeof(uint8_t)};
  cuuint32_t const bdims[2] = {static_cast<cuuint32_t>(box_inner),
                               static_cast<cuuint32_t>(box_outer)};
  cuuint32_t const estride[2] = {1, 1};
  check_driver(cuTensorMapEncodeTiled(&desc,
                                      CU_TENSOR_MAP_DATA_TYPE_UINT8,
                                      2,
                                      ptr,
                                      gdims,
                                      gstride,
                                      bdims,
                                      estride,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
               "cuTensorMapEncodeTiled");
  return desc;
}

__global__ void run_kernel(CUtensorMap const *ta,
                           CUtensorMap const *tb,
                           float const *sa,
                           float const *sb,
                           __nv_bfloat16 *C,
                           int M,
                           int N,
                           int K,
                           int worker_idx,
                           int num_workers) {
  kernel::fp8_gemm_dense_smallm::fp8_gemm_dense_smallm_sm100_task_impl<128, 3>(
      ta, tb, sa, sb, C, M, N, K, worker_idx, num_workers);
}

} // namespace

void dense_gemm_smallm(torch::Tensor a_fp8,     // (M, K) uint8
                       torch::Tensor sa,        // (M, K/128) f32
                       torch::Tensor b_fp8,     // (N, K) uint8
                       torch::Tensor sb,        // (N/128, K/128) f32
                       torch::Tensor c,         // (M, N) bf16
                       int num_workers) {
  TORCH_CHECK(a_fp8.dim() == 2 && b_fp8.dim() == 2 && c.dim() == 2);
  TORCH_CHECK(a_fp8.is_contiguous() && b_fp8.is_contiguous() &&
              c.is_contiguous());
  TORCH_CHECK(a_fp8.scalar_type() == at::kByte, "a must be uint8");
  TORCH_CHECK(b_fp8.scalar_type() == at::kByte, "b must be uint8");
  TORCH_CHECK(c.scalar_type() == at::kBFloat16, "c must be bf16");
  TORCH_CHECK(sa.scalar_type() == at::kFloat && sb.scalar_type() == at::kFloat);

  int const M = static_cast<int>(a_fp8.size(0));
  int const K = static_cast<int>(a_fp8.size(1));
  int const N = static_cast<int>(b_fp8.size(0));
  TORCH_CHECK(b_fp8.size(1) == K, "b/K mismatch");
  TORCH_CHECK(c.size(0) == M && c.size(1) == N, "c shape mismatch");
  TORCH_CHECK(N % 128 == 0 && K % 128 == 0, "N/K must be multiples of 128");

  constexpr int BM = 128, BN = 128, BK = 128, NS = 3, NE = 2;

  // A: gmem (M, K); box (BM, BK). B: gmem (N, K); box (BN, BK).
  CUtensorMap h_ta = make_desc(a_fp8.data_ptr(), M, K, BM, BK);
  CUtensorMap h_tb = make_desc(b_fp8.data_ptr(), N, K, BN, BK);

  auto opts = at::TensorOptions().dtype(at::kByte).device(a_fp8.device());
  auto d_ta = torch::empty({(long)sizeof(CUtensorMap)}, opts);
  auto d_tb = torch::empty({(long)sizeof(CUtensorMap)}, opts);
  C10_CUDA_CHECK(cudaMemcpy(d_ta.data_ptr(), &h_ta, sizeof(CUtensorMap),
                            cudaMemcpyHostToDevice));
  C10_CUDA_CHECK(cudaMemcpy(d_tb.data_ptr(), &h_tb, sizeof(CUtensorMap),
                            cudaMemcpyHostToDevice));

  int const smem =
      kernel::fp8_gemm_dense_smallm::fp8_gemm_dense_smallm_smem_size<BN, NS>();
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      run_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  for (int w = 0; w < num_workers; w++) {
    run_kernel<<<1, 256, smem, stream>>>(
        reinterpret_cast<CUtensorMap const *>(d_ta.data_ptr()),
        reinterpret_cast<CUtensorMap const *>(d_tb.data_ptr()),
        sa.data_ptr<float>(),
        sb.data_ptr<float>(),
        reinterpret_cast<__nv_bfloat16 *>(c.data_ptr()),
        M, N, K, w, num_workers);
  }
  C10_CUDA_CHECK(cudaGetLastError());
  C10_CUDA_CHECK(cudaStreamSynchronize(stream));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dense_gemm_smallm", &dense_gemm_smallm,
        "fp8_gemm_dense_smallm_sm100<128,3> standalone driver");
}
