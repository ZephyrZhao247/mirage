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

// Standalone kernel-wrapper test for the single-GPU (tp=1) MLA MTP decode
// + reduce kernels for DeepSeek V3 on B200 (SM100a).
//
// This invokes
//   kernel::mla_mtp_decode_sm100_task_impl<SINGLE_TILE,false>   (NUM_HEADS=128)
//   kernel::mla_mtp_reduce_sm100_task_impl<512>
// directly through thin __global__ shims that forward blockIdx, exactly as a
// normal CUDA kernel would be launched -- bypassing the MPK persistent-kernel
// runtime / scheduler / codegen entirely. The TMA descriptors for Q/KV are
// built host-side, mirroring the TP=2 standalone wrapper but adapted for
// NUM_HEADS=128 and head-group partitioning (hpb = NUM_HEADS / num_head_groups).

#include "blackwell/mla_mtp_decode_sm100.cuh"

#include <cuda.h>
#include <cudaTypedefs.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>

using namespace kernel::mla_mtp;

// ===== Thin __global__ shims forwarding blockIdx to the __device__ funcs =====
// Grid: (num_head_groups * num_splits, batch_size).  blockIdx.x decodes into
//   gi = blockIdx.x / num_splits   (head group)
//   si = blockIdx.x % num_splits   (split index)
// blockIdx.y = bi (batch index).  This matches the MPK task-metadata mapping
// (request_id=gi, kv_idx=si) but is supplied directly here.
template <bool SINGLE_TILE>
__global__ __launch_bounds__(TB) void shim_decode(
    __grid_constant__ const CUtensorMap Q_tm,
    __grid_constant__ const CUtensorMap KV_tm,
    nv_bfloat16 *Oa,
    float *La,
    float ss,
    int kv_len,
    int sk,
    int num_head_groups,
    int Q_LEN) {
  int gi = blockIdx.x / sk;
  int si = blockIdx.x % sk;
  int bi = blockIdx.y;
  kernel::mla_mtp_decode_sm100_task_impl<SINGLE_TILE, /*WRITE_FINAL=*/false>(
      &Q_tm,
      &KV_tm,
      Oa,
      La,
      ss,
      kv_len,
      sk,
      num_head_groups,
      Q_LEN,
      gi,
      si,
      bi,
      /*local_num_heads=*/NUM_HEADS);
}

// Reduce grid: (ceil(D_V/RD_DV), num_head_groups, batch_size), 512 threads.
__global__ __launch_bounds__(RD_TB) void shim_reduce(nv_bfloat16 const *Oa,
                                                     float const *La,
                                                     nv_bfloat16 *O,
                                                     int sk,
                                                     int num_head_groups,
                                                     int Q_LEN) {
  int dv_base = blockIdx.x * RD_DV;
  int gi = blockIdx.y;
  int bi = blockIdx.z;
  kernel::mla_mtp_reduce_sm100_task_impl</*NUM_THREADS_=*/RD_TB>(
      Oa,
      La,
      O,
      sk,
      num_head_groups,
      Q_LEN,
      dv_base,
      gi,
      bi,
      /*local_num_heads=*/NUM_HEADS);
}

static void check_cu(CUresult e, char const *what) {
  if (e != CUDA_SUCCESS) {
    char const *s = nullptr;
    cuGetErrorString(e, &s);
    TORCH_CHECK(false, what, " failed: ", (s ? s : "unknown"));
  }
}

// DeepSeek V3 MLA softmax scale (matches task_register.cc / pytorch_reference).
static float deepseek_softmax_scale() {
  float const mscale = 0.1f * 1.0f * logf(40.0f) + 1.0f;
  return (1.0f / sqrtf(192.0f)) * mscale * mscale;
}

// Build the KV TMA descriptor (3D), identical geometry to the working path:
//   gmem_dims  = {64, B*KV_LEN, K_ITERS}
//   gmem_strides (in elements, dims 1..2) = {D_K, 128}  (bytes: *2)
//   box_dims   = {64, TILE_S, 1}, 128B swizzle
static CUtensorMap make_kv_desc(void *dKV, int B, int kv_len) {
  CUtensorMap tm{};
  uint64_t gd[3] = {64, (uint64_t)B * kv_len, (uint64_t)K_ITERS};
  uint64_t gs[2] = {(uint64_t)D_K * 2, 128};
  uint32_t bd[3] = {64, (uint32_t)TILE_S, 1};
  uint32_t es[3] = {1, 1, 1};
  check_cu(cuTensorMapEncodeTiled(&tm,
                                  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                  3,
                                  dKV,
                                  gd,
                                  gs,
                                  bd,
                                  es,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  CU_TENSOR_MAP_SWIZZLE_128B,
                                  CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
           "cuTensorMapEncodeTiled(KV)");
  return tm;
}

// Build the Q TMA descriptor (3D). Q is laid out [B*Q_LEN*NUM_HEADS, D_K].
//   gmem_dims  = {64, B*Q_LEN*NUM_HEADS, K_ITERS}
//   gmem_strides = {D_K*2, 128}
//   box_dims   = {64, hpb, 1}, 128B swizzle
// Box height is hpb (heads per group) because the single-GPU kernel loads hpb
// rows per (query, group) TMA slice -- different from TP2 (which loads all
// NUM_HEADS per query). hpb varies with q_len so the descriptor is per-config.
static CUtensorMap make_q_desc(void *dQ, int B, int q_len, int hpb) {
  CUtensorMap tm{};
  uint64_t gd[3] = {64, (uint64_t)B * q_len * NUM_HEADS, (uint64_t)K_ITERS};
  uint64_t gs[2] = {(uint64_t)D_K * 2, 128};
  uint32_t bd[3] = {64, (uint32_t)hpb, 1};
  uint32_t es[3] = {1, 1, 1};
  check_cu(cuTensorMapEncodeTiled(&tm,
                                  CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
                                  3,
                                  dQ,
                                  gd,
                                  gs,
                                  bd,
                                  es,
                                  CU_TENSOR_MAP_INTERLEAVE_NONE,
                                  CU_TENSOR_MAP_SWIZZLE_128B,
                                  CU_TENSOR_MAP_L2_PROMOTION_NONE,
                                  CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE),
           "cuTensorMapEncodeTiled(Q)");
  return tm;
}

// ===== Decode entry: returns (output_partial, output_lse) =====
// q  : bf16 [B*q_len*NUM_HEADS, D_K]
// kv : bf16 [B*kv_len, D_K]
// Returns:
//   output_partial : bf16 [B*num_head_groups*num_splits, D_V*128]
//   output_lse     : fp32 [B*num_head_groups*num_splits, 128]
std::vector<torch::Tensor> mtp_decode(torch::Tensor q,
                                      torch::Tensor kv,
                                      int64_t batch_size,
                                      int64_t q_len,
                                      int64_t kv_len,
                                      int64_t num_head_groups,
                                      int64_t num_splits,
                                      bool force_false = false) {
  TORCH_CHECK(q.is_cuda() && kv.is_cuda(), "q/kv must be CUDA tensors");
  TORCH_CHECK(q.scalar_type() == at::kBFloat16, "q must be bf16");
  TORCH_CHECK(kv.scalar_type() == at::kBFloat16, "kv must be bf16");
  TORCH_CHECK(q.is_contiguous() && kv.is_contiguous(), "q/kv must be contiguous");
  TORCH_CHECK(q.size(0) == batch_size * q_len * NUM_HEADS && q.size(1) == D_K,
              "q shape mismatch");
  TORCH_CHECK(kv.size(0) == batch_size * kv_len && kv.size(1) == D_K,
              "kv shape mismatch");
  TORCH_CHECK(NUM_HEADS % num_head_groups == 0, "NUM_HEADS % groups != 0");

  int B = (int)batch_size;
  int QL = (int)q_len;
  int KL = (int)kv_len;
  int ng = (int)num_head_groups;
  int sk = (int)num_splits;
  int hpb = NUM_HEADS / ng;

  cuInit(0);

  int64_t nblocks = (int64_t)B * ng * sk;
  auto out_part = torch::zeros(
      {nblocks, (int64_t)D_V * 128}, q.options().dtype(torch::kBFloat16));
  // Initialize lse to -inf so inactive splits reduce away cleanly. The kernel
  // only writes La for active splits (t0<t1); the reduce reads all sk slots.
  auto out_lse = torch::full(
      {nblocks, 128}, -1e30f, q.options().dtype(torch::kFloat32));

  CUtensorMap KVtm = make_kv_desc(kv.data_ptr(), B, KL);
  CUtensorMap Qtm = make_q_desc(q.data_ptr(), B, QL, hpb);

  // single_tile is true when every split covers exactly one TILE_S tile.
  int kvt = (KL + TILE_S - 1) / TILE_S;
  int tps = (kvt + sk - 1) / sk;
  // The MPK codegen path ALWAYS instantiates <false,false> (never SINGLE_TILE).
  // force_false lets the test exercise that exact instantiation even for
  // single-tile configs (where the standalone wrappers would pick <true>).
  bool single_tile = (tps == 1) && !force_false;

  float ss = deepseek_softmax_scale();

  cudaFuncSetAttribute(shim_decode<true>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       MTP_SMEM_SIZE);
  cudaFuncSetAttribute(shim_decode<false>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       MTP_SMEM_SIZE);

  dim3 grid(ng * sk, B);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  if (single_tile) {
    shim_decode<true><<<grid, TB, MTP_SMEM_SIZE, stream>>>(
        Qtm,
        KVtm,
        (nv_bfloat16 *)out_part.data_ptr(),
        (float *)out_lse.data_ptr(),
        ss,
        KL,
        sk,
        ng,
        QL);
  } else {
    shim_decode<false><<<grid, TB, MTP_SMEM_SIZE, stream>>>(
        Qtm,
        KVtm,
        (nv_bfloat16 *)out_part.data_ptr(),
        (float *)out_lse.data_ptr(),
        ss,
        KL,
        sk,
        ng,
        QL);
  }
  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "decode launch: ", cudaGetErrorString(err));
  err = cudaStreamSynchronize(stream);
  TORCH_CHECK(err == cudaSuccess, "decode sync: ", cudaGetErrorString(err));

  return {out_part, out_lse};
}

// ===== Reduce entry: LSE-weighted merge across splits =====
// output_partial : bf16 [B*num_head_groups*num_splits, D_V*128]
// output_lse     : fp32 [B*num_head_groups*num_splits, 128]
// Returns out: bf16 [B, q_len, NUM_HEADS, D_V]
torch::Tensor mtp_reduce(torch::Tensor output_partial,
                         torch::Tensor output_lse,
                         int64_t batch_size,
                         int64_t q_len,
                         int64_t num_head_groups,
                         int64_t num_splits) {
  TORCH_CHECK(output_partial.is_cuda() && output_lse.is_cuda(),
              "inputs must be CUDA tensors");
  TORCH_CHECK(output_partial.scalar_type() == at::kBFloat16, "partial bf16");
  TORCH_CHECK(output_lse.scalar_type() == at::kFloat, "lse fp32");

  int B = (int)batch_size;
  int QL = (int)q_len;
  int ng = (int)num_head_groups;
  int sk = (int)num_splits;
  TORCH_CHECK(sk <= MAX_SK, "num_splits exceeds MAX_SK=", (int)MAX_SK);

  auto out = torch::zeros({B, QL, (int64_t)NUM_HEADS, (int64_t)D_V},
                          output_partial.options().dtype(torch::kBFloat16));

  dim3 grid((D_V + RD_DV - 1) / RD_DV, ng, B);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  shim_reduce<<<grid, RD_TB, 0, stream>>>(
      (nv_bfloat16 const *)output_partial.data_ptr(),
      (float const *)output_lse.data_ptr(),
      (nv_bfloat16 *)out.data_ptr(),
      sk,
      ng,
      QL);
  auto err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "reduce launch: ", cudaGetErrorString(err));
  err = cudaStreamSynchronize(stream);
  TORCH_CHECK(err == cudaSuccess, "reduce sync: ", cudaGetErrorString(err));

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mtp_decode",
        &mtp_decode,
        "MLA MTP decode (tp=1) standalone",
        pybind11::arg("q"),
        pybind11::arg("kv"),
        pybind11::arg("batch_size"),
        pybind11::arg("q_len"),
        pybind11::arg("kv_len"),
        pybind11::arg("num_head_groups"),
        pybind11::arg("num_splits"),
        pybind11::arg("force_false") = false);
  m.def("mtp_reduce", &mtp_reduce, "MLA MTP reduce (tp=1) standalone");
}
