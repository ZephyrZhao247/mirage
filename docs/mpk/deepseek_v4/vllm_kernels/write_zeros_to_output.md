# write_zeros_to_output

## Identity
- Source file: `vllm/model_executor/layers/fused_moe/fused_moe.py:38-55` (Triton JIT body `write_zeros_to_output`).
- Language/DSL: **Triton** (`@triton.jit`) — declared as a device-callable helper, not a top-level launchable kernel.
- Third-party dep: none (pure Triton).
- Registered as: in-kernel helper; only ever invoked from `fused_moe_kernel` (line 165) and `fused_moe_kernel_gptq_awq` (line 419).

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/model_executor/layers/fused_moe/fused_moe.py:165` | `fused_moe_kernel_gptq_awq` (in-kernel `@triton.jit` call when `off_experts == -1`) | `C: [T, top_k, N]`, `offs_token: [BLOCK_SIZE_M] int64`, `token_mask: [BLOCK_SIZE_M] bool` | matches `compute_type` (bf16 for V4-Flash) | always invoked at `off_experts == -1` |
| `vllm/model_executor/layers/fused_moe/fused_moe.py:419` | `fused_moe_kernel` (in-kernel call when `off_experts == -1`) | same | same | same |

No standalone launch — this is a `@triton.jit` device function inlined by the Triton compiler at each call site. It is not separately autotuned.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `c_ptr` | (base pointer to) `[T, top_k, N]` | `compute_type` | row-major; strides `(stride_cm, stride_cn)` for the inner two dims | C tensor of the parent kernel — same buffer the GEMM would write |
| `stride_cm` | scalar | int | — | C stride along the (combined `T*top_k`) row dim |
| `stride_cn` | scalar | int | — | C stride along the column dim |
| `pid_n` | scalar | int | — | Parent kernel's N-block id |
| `N` | scalar | `tl.constexpr` int (gptq_awq) or int (fused_moe_kernel) | — | Output N dimension (per the parent kernel) |
| `offs_token` | `[BLOCK_SIZE_M]` | int64 (cast in parent before call) | — | Token offsets the parent kernel decoded from `sorted_token_ids` |
| `token_mask` | `[BLOCK_SIZE_M]` | bool | — | `offs_token < num_valid_tokens` — guards against padding tokens |
| `BLOCK_SIZE_M` | scalar | `tl.constexpr` int | — | Parent kernel's M-block size |
| `BLOCK_SIZE_N` | scalar | `tl.constexpr` int | — | Parent kernel's N-block size |
| `compute_type` | dtype | `tl.constexpr` | — | bf16/fp16/fp32 (parent's accumulator dtype) |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `C` (in-place) | `[BLOCK_SIZE_M, BLOCK_SIZE_N]` tile of `[T, top_k, N]` | `compute_type` | row-major | The `(pid_m, pid_n)` tile is zero-filled (masked by `token_mask`). |

No explicit return value; modifies C in place and returns control to the parent kernel which then `return`s.

## Grid / Block

- No own grid — this helper runs on the parent kernel's CTA (one CTA per `(pid_m, pid_n)` of the parent's grouped grid).
- Per-CTA work: allocates a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` zero accumulator (held in registers / shared memory by Triton, dtype `compute_type`), computes per-lane store pointers and a mask, and issues a single masked `tl.store`. No K loop, no dependencies on quant scales or expert weights.
- Threads/CTA: inherits from the parent — `num_warps` set by the parent's autotune config (typically 4 or 8).
- No `tl.constexpr` autotune of its own.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:636-641` (`MoE.forward`):

```python
for i in range(self.experts_start_idx, self.experts_end_idx):
    if counts[i] == 0:
        continue
    expert = self.experts[i]
    idx, top = torch.where(indices == i)
    y[idx] += expert(x[idx], weights[idx, top, None])
```

The reference initializes `y = torch.zeros_like(x, dtype=torch.float32)` (model.py:634) and only ADDS contributions from experts owned by the current TP rank — non-owned experts contribute zero by virtue of being skipped. This kernel achieves the same semantics by writing explicit zeros for the `expert_ids[pid_m] == -1` slots: downstream the per-(token, k) outputs are summed via `topk_weights * C` + `moe_sum`, so any block that should not contribute MUST be zero (not garbage).

```python
# PyTorch-operator equivalent of the device function:
#
# Inputs:
#   C:           [T, top_k, N]  (already allocated; viewed as [T*top_k, N] by row strides)
#   offs_token:  [BLOCK_SIZE_M] int64
#   token_mask:  [BLOCK_SIZE_M] bool
#   pid_n:       int
#
acc = torch.zeros(BLOCK_SIZE_M, BLOCK_SIZE_N, dtype=compute_type)
offs_cn = pid_n * BLOCK_SIZE_N + torch.arange(BLOCK_SIZE_N)                # [BLOCK_SIZE_N]
c_mask  = token_mask[:, None] & (offs_cn[None, :] < N)                     # [BLOCK_M, BLOCK_N]

# C indexing: C_flat[offs_token, offs_cn] = 0 where mask is True.
C_flat = C.view(-1, N)
C_flat[offs_token[:, None], offs_cn[None, :]] = torch.where(
    c_mask, acc, C_flat[offs_token[:, None], offs_cn[None, :]]
)
```

Notes on fusion / quant:
- **Why explicit zeros and not skip?**: the downstream reduction (`moe_sum` / `topk_weight_and_reduce`) sums across the `top_k` dim. If a block were left uninitialized, the sum would pick up garbage from the `torch.empty` allocation in `fused_experts_impl` (fused_moe.py:1622-1638). Zeroing the entire C ahead of time would also work but at the cost of an extra launch and full-tensor write; this in-kernel branch only zeros the blocks that actually need it (typically `1/world_size` of total blocks under EP).
- **`expert_map is not None` interaction**: when EP is enabled, `expert_ids` is built by `moe_align_block_size` and then remapped via `expert_map[expert_ids]` (`moe_align_block_size.py:100-101`). Experts outside the local TP rank land as `-1` in `expert_ids`, triggering this branch. Without EP, `expert_map is None` and no block ever has `off_experts == -1` — the helper is dead code in that mode. (Note: there's a separate `intermediate_cache3.zero_()` fence at fused_moe.py:1708-1709 for the L2 launch under EP, which means the in-kernel zero is somewhat belt-and-suspenders for L2 but still essential for L1.)
- **Masked store**: the mask `token_mask[:, None] & (offs_cn[None, :] < N)` excludes both padding rows (`offs_token >= num_valid_tokens`) and over-N tile lanes (when `BLOCK_SIZE_N` doesn't evenly divide `N`). Lanes outside the mask are not stored, preserving whatever value was there (irrelevant since those bytes are also out of the valid C region).
- **Address arithmetic**: `c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]`. The parent kernel passes `offs_token` already cast to int64 (fused_moe.py:410 and 157), so the `stride_cm * offs_token` product stays in int64.

## Config-dependent dispatch

- Activation condition: invoked unconditionally from `fused_moe_kernel` and `fused_moe_kernel_gptq_awq` whenever `expert_ids[pid_m] == -1`. **Active when `moe_backend != "deep_gemm_mega_moe"`** (the FusedMoE path); the MegaMoE path does its own expert-dispatch via NVLink and does not use this helper.
- Variants: none — pure Triton device function, no SM split, no quant-strategy split.
- Hard preconditions (implicit from parent kernels):
  - `offs_token` and `token_mask` arrays of length exactly `BLOCK_SIZE_M`.
  - `c_ptr` points to a contiguous `[T, top_k, N]` (or `[T*top_k, N]` flat view) buffer with strides matching what the parent kernel passes.
  - `compute_type` is one of `tl.bfloat16`, `tl.float16`, `tl.float32`.
- Downstream consumer constraint: after this helper writes zeros, the C tile must be summable across topk by `moe_sum` / `topk_weight_and_reduce` without poisoning the sum. Zero output is the additive identity, so the contract holds.
