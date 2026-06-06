# fused_moe_kernel

## Identity
- Source file: `vllm/model_executor/layers/fused_moe/fused_moe.py:292-553` (Triton JIT body `fused_moe_kernel`).
- Launch wrappers: `invoke_fused_moe_triton_kernel` at lines 706-814; dispatch entry `dispatch_fused_moe_kernel` at lines 817-917.
- Language/DSL: **Triton** (`@triton.jit`).
- Third-party dep: none (pure Triton + PyTorch).
- Registered as: Python-level function (no `torch.ops` op); consumed indirectly through `fused_experts` / `fused_experts_impl` (line 1474, 1537) and the `TritonExperts` modular path (`vllm/model_executor/layers/fused_moe/experts/triton_moe.py`).

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/model_executor/layers/fused_moe/fused_moe.py:1672` | `fused_experts_impl` (L1 gate+up GEMM) | `A: [T, H]`, `w1: [E, 2*I, H]`, `C: [T, top_k, 2*I]` | bf16/fp16/fp32 acts; FP8/INT8/INT4 weights | active when `dispatch_fused_moe_kernel` falls through to the non-wna16 branch (fused_moe.py:895-917) |
| `vllm/model_executor/layers/fused_moe/fused_moe.py:1711` | `fused_experts_impl` (L2 down GEMM, `top_k=1`) | `A: [T*top_k, I]`, `w2: [E, H, I]`, `C: [T, 1, H]` | same as L1 | same |
| `vllm/model_executor/layers/fused_moe/experts/triton_moe.py:~572` | `TritonExperts.apply` | as above (per-layer L1/L2) | per quant config | active when `dispatch_fused_moe_kernel` selects this path |
| `vllm/models/deepseek_v4/nvidia/model.py:573` (transitively) | `DeepseekV4MoE._forward_fused_moe → FusedMoE.forward → TritonExperts.apply` | V4-Flash-Base: H=7168, I=2048, top_k=8, E=128 | bf16 acts; FP8 W8A8 (`block_shape=(1,32)` per V4-Flash) | active when `moe_backend != "deep_gemm_mega_moe"` (the default path for V4-Flash-Base which ships `expert_dtype="fp8"`) |

Two launches per MoE-layer forward (L1 and L2), gated by `dispatch_fused_moe_kernel` falling through to `invoke_fused_moe_triton_kernel`. The `(use_int8_w8a16 or use_int4_w4a16) and block_shape[1] > 0` branch routes to `fused_moe_kernel_gptq_awq` instead — see `fused_moe_kernel_gptq_awq.md`.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `a_ptr` (A) | `[num_valid_tokens // top_k, K]` after `// top_k` index reshape (L1: `[T, H]`; L2: `[T*top_k, I]`) | bf16/fp16/fp32 (or FP8/INT8 if pre-quantized via `moe_kernel_quantize_input`) | row-major; strides `(stride_am, stride_ak)` | Per-token activation; FP8/INT8 cases pass the quantized buffer + a separate `a_scale_ptr` |
| `b_ptr` (B) | `[E, N, K]` (L1: N = 2*I; L2: N = H) | bf16/fp16/fp32 / `float8_e4m3fn` / `int8` | row-major over `[E, N, K]`; strides `(stride_be, stride_bk, stride_bn)`. The wrapper passes `stride_bk = B.stride(2)` and `stride_bn = B.stride(1)` to expose K as the GEMM contracting dim (fused_moe.py:790-791). | Per-expert weight stack |
| `c_ptr` (C) | `[T, top_k, N]` (L1: `[T, top_k, 2*I]`; L2: `[T, 1, H]`) | matches `compute_type` | row-major; strides `(stride_cm, stride_cn)` for inner two dims | Output buffer (per-token, per-topk slot) |
| `b_bias_ptr` (optional) | `[E, N]` | matches `compute_type` | row-major; `stride_bbe`, `stride_bbn` | Per-expert bias added after dequant (V4-Flash: bias is `None`) |
| `a_scale_ptr` (optional) | block-wise: `[T, K // group_k]` per-token; channel-wise: `[T]` per-token; tensor-wise: `[1]` scalar | fp32 | strides `(stride_asm, stride_ask)` | FP8/INT8 activation scale |
| `b_scale_ptr` (optional) | block-wise: `[E, N // group_n, K // group_k]`; channel-wise: `[E, N]`; tensor-wise: `[E]`; w8a16: `[E, N]` | fp32 | strides `(stride_bse, stride_bsk, stride_bsn)` | Weight scale (per quant strategy) |
| `topk_weights_ptr` (optional) | `[T*top_k]` flat | fp32 | contiguous | Router weights, multiplied into accumulator iff `MUL_ROUTED_WEIGHT == True` |
| `sorted_token_ids_ptr` (optional) | `[EM]` (EM = padded token count post `moe_align_block_size`) | int32 → cast to int64 in-kernel (line 410) | contiguous; padding entries `>= num_valid_tokens` | Pre-sorted token-id-per-block from `moe_align_block_size`; `None` switches kernel to `naive_block_assignment` mode (line 399-407) |
| `expert_ids_ptr` | `[EM // BLOCK_SIZE_M]` | int32 → cast to int64 (line 414) | contiguous | Expert id for each M-block; `-1` means "expert not on this TP rank" → early-out via `write_zeros_to_output` (line 415-431) |
| `num_tokens_post_padded_ptr` | `[1]` | int32 | scalar | Used to early-exit blocks past the post-pad fence (line 397-398) |
| Constants | `N, K, EM, num_valid_tokens, group_n, group_k, naive_block_assignment, BLOCK_SIZE_{M,N,K}, GROUP_SIZE_M, SPLIT_K, MUL_ROUTED_WEIGHT, top_k, compute_type, use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, per_channel_quant, HAS_BIAS` | mixed (`tl.constexpr`) | — | Compile-time switches — Triton specializes the PTX per combination. |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `C` (in-place via `c_ptr`) | `[T, top_k, N]` | `compute_type` (bf16 for V4-Flash) | row-major; strides `(stride_cm, stride_cn)` over the inner two dims | L1: per-(token, topk) `[2*I]` gate+up output. L2: per-(token, 1) `[H]` down-projection output. The activation/SwiGLU is NOT in this kernel — `apply_moe_activation` (fused_moe.py:1696-1698) runs between L1 and L2. |

For tokens whose pid_m corresponds to an out-of-rank expert (`off_experts == -1`), the kernel writes zeros via `write_zeros_to_output` (line 419-430). See `write_zeros_to_output.md`.

## Grid / Block

- `grid = lambda META: (triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),)` — 1-D grid over `(M_block, N_block)` pairs (fused_moe.py:761-764).
- `pid → (pid_m, pid_n)` decoded via "grouped ordering" (lines 379-387) with `GROUP_SIZE_M` M-blocks per L2-friendly group: `pid_m = first_pid_m + (pid_in_group % group_size_m)`, `pid_n = pid_in_group // group_size_m`. Improves L2 reuse of A.
- `BLOCK_SIZE_{M,N,K}, GROUP_SIZE_M, num_warps, num_stages` are auto-tuned via `try_get_optimal_moe_config` (fused_moe.py:1303-1332) — looks up a JSON config under `vllm/model_executor/layers/fused_moe/configs/E={E},N={N},...,dtype=...json` or falls back to `get_default_config`.
- `SPLIT_K = 1` hard-coded in the launcher (line 768).
- Per-CTA work: one CTA computes a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` tile of C for a single expert (the one identified by `expert_ids[pid_m]`); iterates K in `BLOCK_SIZE_K`-wide chunks, accumulating fp32. After the K loop, applies dequant scaling (per quant strategy), optional bias, optional router-weight multiplication, casts to `compute_type`, and stores with `c_mask = token_mask[:, None] & (offs_cn[None, :] < N)`.
- **`naive_block_assignment` fast path** (line 399-407): when `sorted_token_ids is None` (set by `_prepare_expert_assignment` at fused_moe.py:1453-1463 for small batches where `num_tokens * top_k * 4 <= global_num_experts`), `offs_token = [pid_m, num_valid_tokens, num_valid_tokens, ...]` — only the first lane of each M-block holds a valid token id. This skips the `moe_align_block_size` launch entirely for decode-shaped batches.
- Int64 index trick: `offs = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)` (line 395) and `offs_token = offs_token.to(tl.int64)` (line 410). Comment at lines 408-409 explains: `stride_cm * offs_token` can exceed int32 for large token counts. `offs_bn` also casts to int64 (line 433).

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:596-606` (`Expert.forward` — SwiGLU FFN composed of `w1`, `w3` gate/up and `w2` down) and `model.py:630-644` (`MoE.forward` — top-k dispatch + per-expert evaluation + weight-multiplied sum). The reference loops over experts in Python; this kernel runs each `(M_block, N_block)` GEMM tile as a separate program-id, with `expert_ids[pid_m]` selecting `w1` (L1 launch) or `w2` (L2 launch). The SwiGLU activation is NOT in this kernel — it lives in `apply_moe_activation` between the two launches.

```python
# PyTorch-operator equivalent of one CTA's work for (pid_m, pid_n):
#
# Inputs (L1 launch — analogous for L2 with N=H, K=I):
#   A:         [T, K]  bf16 (or [T, K] fp8 + a_scale)
#   B:         [E, N, K] fp8/int8/bf16 (+ b_scale per quant strategy)
#   sorted_token_ids: [EM] int32 (or None for naive path)
#   expert_ids:        [EM // BLOCK_SIZE_M] int32  (-1 means "not on this TP rank")
#
# Per CTA:
m_start = pid_m * BLOCK_SIZE_M
if m_start >= num_tokens_post_padded:
    return                                     # tail block past the fence

if sorted_token_ids is not None:
    offs_token = sorted_token_ids[m_start : m_start + BLOCK_SIZE_M].long()
else:                                          # naive_block_assignment
    offs_token = torch.full((BLOCK_SIZE_M,), num_valid_tokens, dtype=torch.long)
    offs_token[0] = pid_m
token_mask = offs_token < num_valid_tokens     # valid rows in the M-block

off_experts = expert_ids[pid_m].long()
if off_experts == -1:
    write_zeros_to_output(C, pid_n, ...)       # see write_zeros_to_output.md
    return

# offs_token // top_k recovers the source-row in A (since C is indexed by t*top_k+k).
a_rows = offs_token // top_k                                                    # [BLOCK_SIZE_M]
n_cols = (pid_n * BLOCK_SIZE_N + torch.arange(BLOCK_SIZE_N).long()) % N         # [BLOCK_SIZE_N]

acc = torch.zeros(BLOCK_SIZE_M, BLOCK_SIZE_N, dtype=torch.float32)
for k in range(cdiv(K, BLOCK_SIZE_K)):
    k_lo = k * BLOCK_SIZE_K
    k_hi = min(k_lo + BLOCK_SIZE_K, K)
    a = A[a_rows[:, None], k_lo:k_hi]                                            # masked
    b = B[off_experts, n_cols[None, :], k_lo:k_hi]                               # [BLOCK_K, BLOCK_N]

    if use_fp8_w8a8 or use_int8_w8a8:
        if group_k > 0 and group_n > 0:                                          # block-wise
            offs_ks = k_lo // group_k
            a_s = a_scale[a_rows, offs_ks]                                       # [BLOCK_SIZE_M]
            b_s = b_scale[off_experts, n_cols // group_n, offs_ks]               # [BLOCK_SIZE_N]
            acc += (a.float() @ b.float().T).T * a_s[:, None] * b_s[None, :]
        elif per_channel_quant:                                                  # per-channel
            acc += (a.float() @ b.float().T).T                                   # scale applied after loop
        else:                                                                    # tensor-wise
            acc = tl.dot(a, b, acc=acc)                                          # fp8 fast-accum
    elif use_int8_w8a16:
        acc += a.float() @ b.float().T                                           # B dequant after loop
    else:
        acc += a.float() @ b.float().T

# Post-loop dequant:
if use_int8_w8a16:
    acc *= b_scale[off_experts, n_cols]                                          # [BLOCK_SIZE_N]
elif (use_fp8_w8a8 or use_int8_w8a8) and not (group_k > 0 and group_n > 0):
    acc *= a_scale * b_scale                                                     # scalar per-channel

if HAS_BIAS:
    acc += b_bias[off_experts, n_cols]                                           # [BLOCK_SIZE_N]

if MUL_ROUTED_WEIGHT:
    acc *= topk_weights[offs_token][:, None]                                     # fp32 before cast

C[offs_token, pid_n*BLOCK_SIZE_N : (pid_n+1)*BLOCK_SIZE_N] = acc.to(compute_type)
```

Notes on fusion / quant:
- **3 quant strategies, one kernel**: `use_fp8_w8a8` / `use_int8_w8a8` / `use_int8_w8a16` are `tl.constexpr` flags — Triton specializes the PTX per combination. The block-wise (`group_k > 0 and group_n > 0`) branch multiplies scales inside the K-loop; the channel/tensor-wise branch defers to a post-loop scalar multiply. `use_fp8_w8a8` uses `tl.dot(a, b, acc=accumulator)` which routes to the fp8 fast-accum MMA instruction (line 506 comment).
- **W4A16 / W8A16 with `group_n > 0`** is excluded — `dispatch_fused_moe_kernel` (fused_moe.py:847-893) routes those to `fused_moe_kernel_gptq_awq` instead.
- **Router weight fusion (`MUL_ROUTED_WEIGHT`)**: the topk weight from the router is multiplied into the L2 accumulator (not L1) when `apply_router_weight_on_input=False` (the default; fused_moe.py:1722 passes `not apply_router_weight_on_input` for L2). For L1 it is True only when input pre-weighting is requested. The multiply happens in fp32 BEFORE casting to `compute_type`, per the comment at lines 532-535 — critical for numerical stability.
- **Bias placement** (lines 525-530 comment): bias is applied AFTER dequantization but BEFORE the router-weight multiply, since bias is not quantized and should not be scaled by quant factors. V4-Flash does not use bias on the routed experts.
- **`-1` expert short-circuit**: when the assigned expert is not on the current TP rank, `write_zeros_to_output` writes zeros to that M-block of C and returns early. This preserves the all-reduce semantics — adding zero from this rank doesn't disturb peers' contributions. See `write_zeros_to_output.md`.

## Config-dependent dispatch

- Activation condition: **active when `moe_backend != "deep_gemm_mega_moe"`** (the default for V4-Flash-Base, since its checkpoint ships `expert_dtype="fp8"` which is incompatible with the MegaMoE FP4-only path; see `fp8_fp4_mega_moe.md` for the dispatch).
- Within the FusedMoE path, `dispatch_fused_moe_kernel` (fused_moe.py:817-917) further branches:
  - If `(use_int8_w8a16 or use_int4_w4a16) and block_shape is not None and block_shape[1] > 0` → first checks `should_moe_wna16_use_cuda` (line 852), then either `invoke_fused_moe_wna16_cuda_kernel` (CUDA path) or `invoke_fused_moe_wna16_triton_kernel` → `fused_moe_kernel_gptq_awq` (see `fused_moe_kernel_gptq_awq.md`).
  - Otherwise → `invoke_fused_moe_triton_kernel` → `fused_moe_kernel` (this spec).
- For V4-Flash-Base on NVIDIA, the FP8 W8A8 block-wise (`block_shape=(1, 32)`) quant config flows through this kernel for both L1 and L2 (with `group_n=1, group_k=32`).
- Hardware: pure Triton, works on SM80+ with sufficient shared memory. The FP8 fast-accum path requires SM90+ (Hopper) or SM100+ (Blackwell) to hit native FP8 tensor cores; falls back to upcast-then-fp16/bf16 MMA on Ampere.
- Preconditions enforced by the wrapper (fused_moe.py:728-745):
  - `topk_weights.stride(1) == 1` (line 729).
  - `sorted_token_ids.stride(0) == 1` (line 730).
  - `use_fp8_w8a8 / use_int8_w8a8` requires `B_scale is not None` and block-shape consistency with `B_scale` dims.
  - `(use_int8_w8a16 or use_int4_w4a16)` requires `B_scale is not None` and `block_shape[0] == 0` (token-axis is dense).
  - `MUL_ROUTED_WEIGHT == True` requires `topk_weights is not None`.
- Downstream consumer: L1 output `C` is reshaped to `[T*top_k, 2*I]` and fed to `apply_moe_activation` (SwiGLU + optional `swiglu_limit` clamp; fused_moe.py:1696-1698). L2 output `C` (shape `[T, 1, H]`) is summed across topk via `moe_sum` / `topk_weight_and_reduce` to produce the final `[T, H]` bf16 MoE output. Any spec change must preserve the `[T, top_k, N]` output layout and the `c_mask` semantics that gate stores by `token_mask`.
