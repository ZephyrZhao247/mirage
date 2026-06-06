# fused_moe_kernel_gptq_awq

## Identity
- Source file: `vllm/model_executor/layers/fused_moe/fused_moe.py:58-289` (Triton JIT body `fused_moe_kernel_gptq_awq`).
- Launch wrapper: `invoke_fused_moe_wna16_triton_kernel` at lines 616-703.
- Language/DSL: **Triton** (`@triton.jit`).
- Third-party dep: none (pure Triton + PyTorch).
- Registered as: Python-level function (no `torch.ops` op).

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/model_executor/layers/fused_moe/fused_moe.py:667` | `invoke_fused_moe_wna16_triton_kernel` | `A: [T, K]` bf16/fp16, `B: [E, N, K//pack]` int32 (4-bit packed) or `[E, N, K]` int8, `C: [T, top_k, N]`, `B_scale: [E, N//1, K//group_size]`, `B_zp: [E, N, K//group_size]` (optional) | bf16/fp16 acts; INT4 W4A16 or INT8 W8A16 weights | `dispatch_fused_moe_kernel` selects this when `(use_int8_w8a16 or use_int4_w4a16) and block_shape is not None and block_shape[1] > 0 and not use_moe_wna16_cuda` (fused_moe.py:847-893) |
| `vllm/model_executor/layers/fused_moe/fused_moe.py:1672, 1711` (transitively) | `fused_experts_impl` → `dispatch_fused_moe_kernel` (L1 and L2) | same | per config | as above |

Two launches per MoE-layer forward (L1 and L2) when the active path is `(W8A16 or W4A16) + group quant`. For V4-Flash-Base (FP8 W8A8) this kernel is NOT on the path — `fused_moe_kernel` runs instead. This spec documents the older GPTQ/AWQ INT4/INT8 alternative kept for non-FP8 checkpoints.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `a_ptr` (A) | `[T, K]` (or `[T*top_k, K]` for L2) | bf16/fp16 | row-major; `(stride_am, stride_ak)` | Per-token activation, dequantized |
| `b_ptr` (B) | `use_int4_w4a16`: `[E, N, K // 2]` int8 (two 4-bit weights per byte); `use_int8_w8a16`: `[E, N, K]` int8 | int8 | row-major over `[E, N, K_packed]`; strides `(stride_be, stride_bk, stride_bn)` (note: wrapper passes `stride_bk = B.stride(2)`, `stride_bn = B.stride(1)`, lines 684-685) | Per-expert quantized weight |
| `c_ptr` (C) | `[T, top_k, N]` | matches `compute_type` (bf16/fp16) | row-major; `(stride_cm, stride_cn)` | Output |
| `b_scale_ptr` (B_scale) | `[E, N, K // group_size]` | fp16/bf16 (matches A) | row-major over `[E, N, K_blocks]`; strides `(stride_bse, stride_bsk, stride_bsn)` (line 688-690) | Per-(expert, n, k-group) weight scale |
| `b_zp_ptr` (B_zp, optional) | `use_int4_w4a16`: `[E, N // 2, K // group_size]` int8; `use_int8_w8a16`: `[E, N, K // group_size]` int8 | int8 | row-major | Per-(expert, n, k-group) zero-point. `None` ⇒ uses fixed `b_zp_num = 8` (W4) or `128` (W8) as if no zero-point was learned (line 201-204). |
| `topk_weights_ptr` (optional) | `[T*top_k]` | fp32 | contiguous | Router weights, multiplied when `MUL_ROUTED_WEIGHT == True` |
| `sorted_token_ids_ptr` | `[EM]` | int32 → cast to int64 (line 157) | contiguous | From `moe_align_block_size` |
| `expert_ids_ptr` | `[EM // BLOCK_SIZE_M]` | int32 → cast to int64 (line 160) | contiguous | Expert id per M-block; `-1` ⇒ write zeros |
| `num_tokens_post_padded_ptr` | `[1]` | int32 | scalar | Fence past which M-blocks are skipped |
| Constants | `N, K` (constexpr), `EM, num_valid_tokens, block_k_diviable, group_size, BLOCK_SIZE_{M,N,K}, GROUP_SIZE_M, SPLIT_K, MUL_ROUTED_WEIGHT, top_k, compute_type, has_zp, use_int4_w4a16, use_int8_w8a16` | mixed (`tl.constexpr`) | — | Compile-time switches. Note `N, K` are `tl.constexpr` here (line 71-72), unlike `fused_moe_kernel` which takes them as runtime ints. |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `C` (in-place via `c_ptr`) | `[T, top_k, N]` | `compute_type` | row-major | L1: per-(token, topk) `[2*I]` gate+up. L2: per-(token, 1) `[H]` down-projection. SwiGLU runs in `apply_moe_activation` between L1 and L2 (not in this kernel). |

`-1` experts route to `write_zeros_to_output` (line 165-176); see `write_zeros_to_output.md`.

## Grid / Block

- `grid = lambda META: (triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(B.size(1), META["BLOCK_SIZE_N"]),)` (fused_moe.py:648-651). 1-D grouped grid over `(M_block, N_block)`.
- `pid → (pid_m, pid_n)` decoded the same way as `fused_moe_kernel` via `GROUP_SIZE_M` super-grouping (lines 136-144).
- `BLOCK_SIZE_{M,N,K}, GROUP_SIZE_M` etc. auto-tuned via `get_moe_wna16_block_config` (line 653-665) which can override the optimal-config lookup based on `num_valid_tokens`, `group_size`, `real_top_k`.
- Per-CTA work: same shape as `fused_moe_kernel` — one `[BLOCK_SIZE_M, BLOCK_SIZE_N]` output tile per CTA. K loop iterates `BLOCK_SIZE_K`-wide chunks, dequantizing B inside each iteration.
- Int64 index trick: `offs_token_id = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)` (line 155), and `offs_token = tl.load(...).to(tl.int64)` (line 157); same overflow rationale as `fused_moe_kernel`. `off_experts = tl.load(expert_ids_ptr + pid_m).to(tl.int64)` (line 160). `offs_bn` is also int64 (line 179).
- `block_k_diviable` constexpr: `True` ⇒ skip K-mask load (line 218-223). Set by wrapper from `A.size(1) % config["BLOCK_SIZE_K"] == 0` (line 694).

## Math

Reference: same as `fused_moe_kernel` — `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:596-606` (`Expert.forward`) and `model.py:630-644` (`MoE.forward`). The quantization-dequantization recipe (`b_dequant = (b - b_zp) * b_scale`) is GPTQ/AWQ convention applied per K-group; the reference uses bf16 weights so this dequant has no analog in the reference.

```python
# PyTorch-operator equivalent of one CTA's work for (pid_m, pid_n):
#
# Inputs (use_int4_w4a16 case shown — W8 differs only in pack width):
#   A:        [T, K] bf16/fp16
#   B:        [E, N, K // 2] int8  (two int4 per byte, low nibble = even k)
#   B_scale:  [E, N, K // group_size] fp16/bf16
#   B_zp:     [E, N // 2, K // group_size] int8  (two int4 per byte, optional)
#
# Per CTA:
m_start = pid_m * BLOCK_SIZE_M
if m_start >= num_tokens_post_padded:
    return

offs_token = sorted_token_ids[m_start : m_start + BLOCK_SIZE_M].long()
token_mask = offs_token < num_valid_tokens
off_experts = expert_ids[pid_m].long()
if off_experts == -1:
    write_zeros_to_output(C, pid_n, ...)
    return

a_rows = offs_token // top_k                                                  # [BLOCK_SIZE_M]
n_cols = (pid_n * BLOCK_SIZE_N + torch.arange(BLOCK_SIZE_N).long()) % N       # [BLOCK_SIZE_N]

acc = torch.zeros(BLOCK_SIZE_M, BLOCK_SIZE_N, dtype=torch.float32)
for k in range(cdiv(K, BLOCK_SIZE_K)):
    k_lo = k * BLOCK_SIZE_K
    k_hi = min(k_lo + BLOCK_SIZE_K, K)
    a = A[a_rows[:, None], k_lo:k_hi]                                          # masked

    if use_int4_w4a16:
        b_packed = B[off_experts, n_cols, k_lo//2 : k_hi//2]                   # [BLOCK_K//2, BLOCK_N] int8
        b_shift  = (torch.arange(BLOCK_SIZE_K) % 2) * 4                        # even nibble = 0, odd = 4
        b_q      = (b_packed[k_lo:k_hi] >> b_shift) & 0xF                      # [BLOCK_K, BLOCK_N] int8
    else:                                                                       # use_int8_w8a16
        b_q = B[off_experts, n_cols, k_lo:k_hi]                                # [BLOCK_K, BLOCK_N] int8

    # K-group scale lookup:
    ks = (torch.arange(BLOCK_SIZE_K) + k * BLOCK_SIZE_K) // group_size
    b_s = B_scale[off_experts, n_cols[None, :], ks[:, None]].float()            # [BLOCK_K, BLOCK_N]

    if has_zp:
        if use_int4_w4a16:
            zp_packed  = B_zp[off_experts, n_cols // 2, ks]
            zp_shifter = (n_cols % 2) * 4
            b_zp_val   = ((zp_packed >> zp_shifter) & 0xF).float()
        else:
            b_zp_val = B_zp[off_experts, n_cols, ks].float()
        b = ((b_q.float() - b_zp_val) * b_s).to(compute_type)
    else:
        b_zp_num = 8 if use_int4_w4a16 else 128                                # symmetric quant midpoint
        b = ((b_q.float() - b_zp_num) * b_s).to(compute_type)

    acc = tl.dot(a, b, acc=acc)                                                # fp16/bf16 MMA

if MUL_ROUTED_WEIGHT:
    moe_w = topk_weights[offs_token]
    acc *= moe_w[:, None]

C[offs_token, pid_n*BLOCK_SIZE_N : (pid_n+1)*BLOCK_SIZE_N] = acc.to(compute_type)
```

Notes on fusion / quant:
- **Per-iteration dequant**: unlike `fused_moe_kernel` which can amortize block-wise scales to the outside of the K-loop, `fused_moe_kernel_gptq_awq` always dequantizes B inside the K loop because the scale grid is `K // group_size`-fine (line 234-240). The dequant cast to `compute_type` (bf16/fp16) lets the MMA run at native compute precision — there's no native INT4/INT8 MMA path being exercised here.
- **Bit-packing**: W4A16 packs two 4-bit weights per int8 byte. The kernel uses `b_shifter = (offs_k[:, None] % 2) * 4` to extract the low (k even) or high (k odd) nibble (line 192, 232). Zero-points use the same nibble trick but along N: `b_zp_shifter = (offs_bn[None, :] % 2) * 4` (line 206, 252).
- **Default zero-points**: when `B_zp is None`, the kernel uses `b_zp_num = 8` (mid-range of `[0, 15]` for W4) or `b_zp_num = 128` (mid-range of `[0, 255]` for W8) as a symmetric-quant default (line 201-204, 269). This matches AWQ's symmetric-quant convention.
- **No bias, no FP8/INT8 W8A8**: this kernel does NOT support bias (no `b_bias_ptr` param) and does NOT support FP8/INT8 W8A8. For those, the dispatcher routes to `fused_moe_kernel` instead.
- **`MUL_ROUTED_WEIGHT` numerics**: the router weight multiply happens before the final cast (line 279-281), in fp32, matching `fused_moe_kernel`'s convention.

## Config-dependent dispatch

- Activation condition: **active when `moe_backend != "deep_gemm_mega_moe"`** AND `(use_int8_w8a16 or use_int4_w4a16) and block_shape is not None and block_shape[1] > 0` AND `should_moe_wna16_use_cuda(...) == False` (fused_moe.py:847-876).
- The decision between this Triton kernel and `invoke_fused_moe_wna16_cuda_kernel` (CUDA path, `ops.moe_wna16_gemm`) is made by `should_moe_wna16_use_cuda` (fused_moe.py:1192-…) based on `num_valid_tokens`, `group_size`, `num_experts`, and bit width. The CUDA path is preferred for small batches on SM75 (Turing); the Triton path covers SM80+ and large batches.
- **NOT applicable to V4-Flash-Base**: V4-Flash-Base ships `expert_dtype="fp8"`, which surfaces in vLLM as the FP8 W8A8 block-scaled quant config (`block_shape=(1, 32)`). That config routes through `fused_moe_kernel` (FP8 path), not this kernel. This spec documents the canonical FusedMoE INT4/INT8 alternative.
- Preconditions enforced by the wrapper (fused_moe.py:634-636):
  - `B_scale is not None and B_scale.ndim == 3`.
  - `B_zp is None or B_zp.ndim == 3`.
  - `block_shape is not None and block_shape[0] == 0` (token-axis dense).
- Downstream consumer: same as `fused_moe_kernel` — L1 output feeds `apply_moe_activation` (SwiGLU), L2 output is reduced via `moe_sum`/`topk_weight_and_reduce`. Shape `[T, top_k, N]` must be preserved.
