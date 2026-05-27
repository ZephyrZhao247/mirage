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
// indexer_q_transform_sm100.cuh — DeepSeek V4-Flash Indexer per-step Q
// transform (Wave-2 sub-batch C2).
//
// Per token ``t``, per index head ``h`` (H = INDEX_N_HEADS = 64, D =
// INDEX_HEAD_DIM = 128 in V4-Flash):
//
//   1. **Q expand via wq_b** — fp32 GEMV:
//        q_idx[t, h, d] = sum_k q_lora[t, k] * wq_b[h*D + d, k]
//      where ``q_lora`` is the q-LoRA-A normalized output and ``wq_b`` is
//      ASSUMED PRE-HADAMARD-ABSORBED at convert time (i.e. the stored
//      weight is ``H * wq_b`` so the produced Q is already Hadamard-
//      rotated; no runtime Hadamard pass).
//
//   2. **GPT-J interleaved RoPE** on the last ROPE_DIM channels:
//        new q[2k]   = q[2k]   * cos[k] - q[2k+1] * sin[k]
//        new q[2k+1] = q[2k+1] * cos[k] + q[2k]   * sin[k]
//      ``cos_sin_cache`` stores cos in ``[:, :ROPE_DIM/2]`` and sin in
//      ``[:, ROPE_DIM/2:]`` (vLLM Triton convention, matches the
//      ``inv_rope_fp8_quant_o_sm100`` and Compressor kernels).
//
//   3. **Per-block MXFP4 quantization** with UE8M0 exponent scales over
//      the full HEAD_DIM, in blocks of BLOCK_SIZE = 32 elements:
//        amax    = max(|q_block|)
//        exp     = ceil(log2(max(amax, eps) / 6.0))     # 6 = E2M1 max
//        scale   = 2^exp                                 # power-of-two
//        ue8m0   = clamp(exp + 127, [0, 255])            # uint8 byte
//        v_e2m1  = quantize_e2m1(q / scale)              # to {±0, ±0.5,
//                                                        # ±1, ±1.5, ±2,
//                                                        # ±3, ±4, ±6}
//      Two E2M1 nibbles pack into one uint8 byte: bit [7:4] is the
//      element at odd index (d=2k+1), bit [3:0] is the element at even
//      index (d=2k) — i.e. the byte index is ``d/2``. Each nibble layout
//      is the IEEE-style ``S EE M``: bit 3 = sign, bits 2..1 = exponent,
//      bit 0 = mantissa. This matches the PyTorch reference in the
//      catalog module.
//
//   Output layout:
//     q_fp4   [T, H, D/2]   uint8   — packed E2M1 nibbles
//     q_scale [T, H, D/32]  uint8   — UE8M0 exponent bytes (one per block)
//
// MPK convention:
//   * ``__device__ __forceinline__`` task impl.
//   * blockIdx-agnostic: ``t`` is read from
//     ``task_desc->task_metadata.token_offset``.
//   * v1: one CTA per token (``num_tokens_per_task == 1``); the loop
//     below is written generically so larger slices Just Work.
//   * NUM_THREADS is a template parameter; threadIdx.x >= NUM_THREADS
//     lanes early-out so a worker block with WORKER_NUM_THREADS=256 can
//     host a smaller-CTA task without wasted work.
//   * Default NUM_THREADS == INDEX_HEAD_DIM (one thread per output channel
//     of one head); the registration function picks 128 for V4-Flash and
//     downgrades when D < 128 (test mode).
//   * Heads are processed serially inside the CTA so a single shmem tile
//     of q_lora is reused across all H heads.
//
// FP32 accumulators. The kernel currently requires:
//   * NUM_THREADS == INDEX_HEAD_DIM  (one thread per d)
//   * INDEX_HEAD_DIM % 32 == 0       (block-size 32 MXFP4)
//   * ROPE_DIM % 2 == 0
//   * NUM_THREADS % 32 == 0          (warp shuffle for partner exchange)
// =============================================================================

#pragma once
#include "tasks/common/common_header.cuh"
#include <cstdint>
#include <cuda_bf16.h>

namespace kernel {

namespace indexer_q_transform_detail {

// Quantize a fp32 ``x`` to the nearest E2M1 value in
// {±0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} and return its 4-bit code.
//
// E2M1 layout (used by NVIDIA MXFP4 spec):
//   bit 3 : sign
//   bits 2..1 : exponent (biased = 1)
//   bit 0 : mantissa (implicit leading 1 except for subnormal exp==0)
//
//  | code | bits | value |
//  |------|------|-------|
//  | 0    | 0000 |  +0   |
//  | 1    | 0001 |  +0.5 |
//  | 2    | 0010 |  +1.0 |
//  | 3    | 0011 |  +1.5 |
//  | 4    | 0100 |  +2.0 |
//  | 5    | 0101 |  +3.0 |
//  | 6    | 0110 |  +4.0 |
//  | 7    | 0111 |  +6.0 |
//  | 8..15: same magnitudes with sign bit set.
//
// Round-to-nearest, ties-to-even is approximated by rounding the
// magnitude to the closest representable value via a midpoint table —
// simple comparisons suffice for the 8 magnitudes above and match the
// PyTorch reference in the catalog module.
__device__ __forceinline__ uint8_t quantize_e2m1(float x) {
  uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
  float a = fabsf(x);
  uint8_t mag;
  // Midpoints between consecutive E2M1 magnitudes: 0.25, 0.75, 1.25,
  // 1.75, 2.5, 3.5, 5.0. Above 5.0 the closest is 6 (saturating; the
  // caller is expected to have applied per-block scaling so values
  // already lie roughly in [-6, 6], but we still clamp here for safety).
  if (a < 0.25f) {
    mag = 0; // 0
  } else if (a < 0.75f) {
    mag = 1; // 0.5
  } else if (a < 1.25f) {
    mag = 2; // 1
  } else if (a < 1.75f) {
    mag = 3; // 1.5
  } else if (a < 2.5f) {
    mag = 4; // 2
  } else if (a < 3.5f) {
    mag = 5; // 3
  } else if (a < 5.0f) {
    mag = 6; // 4
  } else {
    mag = 7; // 6 (saturating)
  }
  return static_cast<uint8_t>(sign | mag);
}

} // namespace indexer_q_transform_detail

template <int Q_LORA_RANK,
          int INDEX_N_HEADS,
          int INDEX_HEAD_DIM,
          int ROPE_DIM,
          int NUM_THREADS = 128>
__device__ __forceinline__ void indexer_q_transform_task_impl(
    void const *__restrict__ q_lora_ptr,        // bf16 [T, Q_LORA_RANK]
    void const *__restrict__ wq_b_ptr,          // bf16 [H*D, Q_LORA_RANK]
    void const *__restrict__ cos_sin_cache_ptr, // bf16 [max_pos, ROPE_DIM]
    void const *__restrict__ positions_ptr,     // int32 [T]
    void *__restrict__ q_fp4_ptr,               // uint8 [T, H, D/2]
    void *__restrict__ q_scale_ptr,             // uint8 [T, H, D/32]
    int token_offset,
    int num_tokens_per_task,
    int num_tokens_total) {
  static_assert(INDEX_HEAD_DIM == NUM_THREADS,
                "v1: NUM_THREADS must equal INDEX_HEAD_DIM (one thread per d)");
  static_assert(NUM_THREADS % NUM_THREADS_PER_WARP == 0,
                "NUM_THREADS must be a multiple of warp size");
  static_assert(INDEX_HEAD_DIM % 32 == 0,
                "INDEX_HEAD_DIM must be a multiple of 32 (MXFP4 block size)");
  static_assert(ROPE_DIM % 2 == 0,
                "ROPE_DIM must be even (GPT-J interleaved pairs)");
  static_assert(ROPE_DIM <= INDEX_HEAD_DIM,
                "ROPE_DIM cannot exceed INDEX_HEAD_DIM");

  constexpr int BLOCK_SIZE = 32;
  constexpr int NUM_BLOCKS_PER_ROW = INDEX_HEAD_DIM / BLOCK_SIZE;
  constexpr int NOPE_DIM = INDEX_HEAD_DIM - ROPE_DIM;
  constexpr int HALF_ROPE = ROPE_DIM / 2;
  constexpr float kE2M1Max = 6.0f;
  constexpr float kEps = 1e-30f;

  // Worker thread blocks may have up to WORKER_NUM_THREADS threads. Only
  // the first NUM_THREADS lanes participate.
  if (threadIdx.x >= NUM_THREADS) {
    return;
  }

  // Typed pointers.
  __nv_bfloat16 const *__restrict__ q_lora =
      reinterpret_cast<__nv_bfloat16 const *>(q_lora_ptr);
  __nv_bfloat16 const *__restrict__ wq_b =
      reinterpret_cast<__nv_bfloat16 const *>(wq_b_ptr);
  __nv_bfloat16 const *__restrict__ cs_cache =
      reinterpret_cast<__nv_bfloat16 const *>(cos_sin_cache_ptr);
  int const *__restrict__ positions =
      reinterpret_cast<int const *>(positions_ptr);
  uint8_t *__restrict__ q_fp4 = reinterpret_cast<uint8_t *>(q_fp4_ptr);
  uint8_t *__restrict__ q_scale = reinterpret_cast<uint8_t *>(q_scale_ptr);

  // Shared workspace.
  //   * q_lora_smem: [Q_LORA_RANK] bf16 — the per-token q_lora row,
  //     loaded once and reused across all H heads. Pair-packing of
  //     E2M1 nibbles is done via warp shuffle (same-warp partner), so
  //     no additional shmem is needed.
  __shared__ __nv_bfloat16 q_lora_smem[Q_LORA_RANK];

  for (int local = 0; local < num_tokens_per_task; ++local) {
    int const t = token_offset + local;
    if (t >= num_tokens_total) {
      break;
    }

    // ----- Load q_lora row into shared memory. -----
    __nv_bfloat16 const *q_row =
        q_lora + static_cast<size_t>(t) * Q_LORA_RANK;
    for (int k = threadIdx.x; k < Q_LORA_RANK; k += NUM_THREADS) {
      q_lora_smem[k] = q_row[k];
    }
    __syncthreads();

    // ----- Per-token RoPE table lookup. -----
    int const pos = positions[t];
    __nv_bfloat16 const *cos_base = cs_cache + pos * ROPE_DIM;
    __nv_bfloat16 const *sin_base = cos_base + HALF_ROPE;

    // ----- Per-head loop. -----
    for (int h = 0; h < INDEX_N_HEADS; ++h) {
      int const d = threadIdx.x; // each lane owns one channel.

      // Step 1: GEMV — acc = sum_k q_lora_smem[k] * wq_b[h*D + d, k].
      // ``wq_b`` is row-major [H*D, K]; row h*D+d is contiguous along k.
      __nv_bfloat16 const *w_row =
          wq_b + static_cast<size_t>(h * INDEX_HEAD_DIM + d) * Q_LORA_RANK;
      float acc = 0.f;
      for (int k = 0; k < Q_LORA_RANK; ++k) {
        float qv = __bfloat162float(q_lora_smem[k]);
        float wv = __bfloat162float(w_row[k]);
        acc += qv * wv;
      }

      // Step 2: GPT-J RoPE on the last ROPE_DIM channels. Pair partner
      // is ``d ^ 1`` (NOPE_DIM is even by static_assert, so toggling
      // bit 0 stays within the rope tail). Each thread fetches the
      // partner's pre-RoPE value via warp shuffle (same warp because
      // pairs are adjacent in d). The shuffle is executed by ALL active
      // lanes (uniformly) — the rotation result is then conditionally
      // adopted only by the rope-tail lanes (d >= NOPE_DIM). Doing the
      // shuffle from inside a divergent branch with a full warp mask is
      // undefined; this restructure keeps participation uniform.
      float partner = __shfl_xor_sync(0xffffffff, acc, 1,
                                      NUM_THREADS_PER_WARP);
      if (d >= NOPE_DIM) {
        int rope_local = d - NOPE_DIM;     // [0, ROPE_DIM)
        int cs_idx = rope_local >> 1;       // pair index in [0, HALF_ROPE)
        float cos_v = __bfloat162float(cos_base[cs_idx]);
        float sin_v = __bfloat162float(sin_base[cs_idx]);
        bool is_even = ((rope_local & 1) == 0);
        // Forward GPT-J RoPE:
        //   x_even' = x_even * cos - x_odd  * sin
        //   x_odd'  = x_odd  * cos + x_even * sin
        float rotated = is_even ? (acc * cos_v - partner * sin_v)
                                : (acc * cos_v + partner * sin_v);
        acc = rotated;
      }

      // Step 3: Per-block MXFP4 quant. Each block of 32 contiguous d
      // channels reduces a block-wise amax via warp shuffle (NUM_THREADS
      // >= 32 always; BLOCK_SIZE == 32 lines up with the warp width on
      // NVIDIA), then quantizes to E2M1 with a UE8M0 power-of-two scale.
      int blk = d / BLOCK_SIZE;
      // Reduce amax across the 32 lanes of this block.
      float local_max = fabsf(acc);
#pragma unroll
      for (int offset = BLOCK_SIZE / 2; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(0xffffffff, local_max, offset,
                                      BLOCK_SIZE);
        local_max = fmaxf(local_max, other);
      }
      float block_amax = local_max;

      // UE8M0 exponent: exp = ceil(log2(max(amax, eps) / 6.0)), clamped
      // into [-127, 128] (8-bit unsigned biased; byte = exp + 127).
      float amax_safe = fmaxf(block_amax, kEps);
      float exp_f = ceilf(log2f(amax_safe / kE2M1Max));
      int exp_i = (int)exp_f;
      if (exp_i < -127) exp_i = -127;
      if (exp_i > 128) exp_i = 128;
      int ue8m0 = exp_i + 127;
      if (ue8m0 < 0) ue8m0 = 0;
      if (ue8m0 > 255) ue8m0 = 255;
      float scale = exp2f(static_cast<float>(exp_i));
      float inv_scale = 1.0f / scale;

      // Write per-block scale (one lane per block).
      if ((d % BLOCK_SIZE) == 0) {
        size_t s_off = (static_cast<size_t>(t) * INDEX_N_HEADS + h)
                           * NUM_BLOCKS_PER_ROW + blk;
        q_scale[s_off] = static_cast<uint8_t>(ue8m0);
      }

      // Step 4: Quantize this thread's element to a 4-bit E2M1 code and
      // park it in shmem so the lower of each pair packs both nibbles
      // into one byte.
      float scaled = acc * inv_scale;
      // Clamp into [-6, 6] for safety; the scale was chosen so this is
      // typically already satisfied modulo rounding.
      scaled = fminf(fmaxf(scaled, -kE2M1Max), kE2M1Max);
      uint8_t code =
          indexer_q_transform_detail::quantize_e2m1(scaled);

      // Each pair (d_even, d_odd) → one byte at index d/2.
      //   bits [3:0] = code for d_even (lower d of the pair)
      //   bits [7:4] = code for d_odd
      // Both lanes of the pair live in the same warp, so we use a
      // warp shuffle to fetch the partner's nibble code and only the
      // even lane writes the packed byte. This avoids any
      // __syncthreads() inside the per-head loop (which would otherwise
      // be tricky given lanes >= NUM_THREADS have already returned).
      uint32_t code32 = static_cast<uint32_t>(code) & 0xFu;
      uint32_t partner_code32 = __shfl_xor_sync(
          0xffffffff, code32, 1, NUM_THREADS_PER_WARP);
      bool is_odd = ((d & 1) != 0);
      if (!is_odd) {
        // d is even: own nibble in low bits, partner (odd) in high bits.
        uint8_t byte = static_cast<uint8_t>(
            code32 | (partner_code32 << 4));
        int const num_bytes = INDEX_HEAD_DIM / 2;
        size_t row_off = (static_cast<size_t>(t) * INDEX_N_HEADS + h)
                             * num_bytes;
        q_fp4[row_off + (d >> 1)] = byte;
      }
    }
  }
}

} // namespace kernel
