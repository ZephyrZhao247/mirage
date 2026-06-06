# mtp_shared_head_rmsnorm

## Identity
- Source file: `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py:100-121` (Triton JIT body `_mtp_shared_head_rmsnorm_kernel`); shared helper `_rmsnorm_row` at lines 25-40; user-facing wrapper `mtp_shared_head_rmsnorm` at lines 124-150.
- Language/DSL: **Triton** (`@triton.jit`).
- Third-party dep: none (pure Triton + PyTorch).
- Registered as: Python-level function (no `torch.ops` op); imported from `vllm.models.deepseek_v4.common.ops` package.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/nvidia/mtp.py:247` | `DeepSeekV4MultiTokenPredictor.compute_logits` | `hidden_states: [T, H=4096]` (after `hc_head_fused_kernel_tilelang` at line 239-246 flattens `[T, HC_MULT, H]` → `[T, H]`) | bf16 in / bf16 out | always on per MTP draft step's `compute_logits` |

Single call site. Runs once per MTP draft step's logits computation, between the HC-head fuse and the lm_head GEMM:
1. `hc_head_fused_kernel_tilelang(...)` at `mtp.py:239` → `[T, H]` bf16
2. `mtp_shared_head_rmsnorm(hidden_states, mtp_layer.shared_head.norm.weight.data, eps)` at `mtp.py:247` → `[T, H]` bf16
3. `self.logits_processor(mtp_layer.shared_head.head, hidden_states)` at `mtp.py:252` → `[T, vocab_size]` bf16

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `hidden_states` (`x_ptr`) | `[num_tokens, HIDDEN]` (V4-Flash: `HIDDEN=4096`) | bf16 | row-major, **contiguous** (asserted at line 135) | HC-head-fused hidden state ready for the MTP `shared_head.head` (LM head) GEMM. |
| `weight` (`weight_ptr`) | `[HIDDEN]` | bf16 (RMSNorm default dtype) | contiguous (asserted line 136) | RMSNorm gain. `mtp_layer.shared_head.norm.weight.data`. |
| `eps` | scalar float | — | — | `mtp_layer.shared_head.norm.variance_epsilon` (== `config.rms_norm_eps`). |
| `HIDDEN` | scalar (`tl.constexpr`) | int | — | `hidden_states.shape[1]`, baked at JIT time. |
| `BLOCK_SIZE` | scalar (`tl.constexpr`) | int | — | `triton.next_power_of_2(HIDDEN)` — V4-Flash: `4096`. |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `out` (`out_ptr`) | `[num_tokens, HIDDEN]` | bf16 (matches `hidden_states.dtype`) | `torch.empty_like(hidden_states)` — contiguous, row-major | RMS-normalized hidden state ready for `shared_head.head` (ParallelLMHead) and the downstream `LogitsProcessor`. |

Early-return: when `num_tokens == 0` the wrapper returns the empty allocation without launching (lines 139-140).

## Grid / Block

- `grid_dim = (num_tokens,)` — one CTA per token (line 142).
- `block_dim`: default Triton `num_warps` (no explicit override). For `BLOCK_SIZE=4096`, Triton defaults to `num_warps=4` (128 threads/CTA).
- `num_stages`: default (no override).
- Autotune configs: **none** — `BLOCK_SIZE = next_power_of_2(HIDDEN)` is computed once at launch and baked as `tl.constexpr`. Single tile per CTA covers `HIDDEN` lanes with `mask = block < HIDDEN`.
- Per-CTA work: a single CTA handles one token end-to-end. Loads `HIDDEN` bf16 values, casts to fp32, computes variance via `tl.sum(x * x, axis=0) / HIDDEN`, applies `rsqrt(var + eps) * w`, stores back as bf16. Identical body (`_rmsnorm_row`, lines 25-40) to `fused_mtp_input_rmsnorm` — same PTX module covers both.
- Int64 index trick: `token_idx = tl.program_id(0).to(tl.int64)` (line 109) — same convention as `fused_q_kv_rmsnorm` and `_fused_mtp_input_rmsnorm_kernel`. Needed because `token_idx * HIDDEN` overflows int32 once `T ≳ 524K` for `HIDDEN=4096`; the cast also protects against unusually large prefills.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:722` (`MTPBlock.head` is `ParallelHead.forward` → `self.get_logits(norm(x))` where `norm` is the `shared_head.norm` RMSNorm; see `model.py:747` for `self.norm = RMSNorm(args.dim, args.norm_eps)` in `MTPBlock.__init__`, and `model.py:766` where the MTPBlock invokes `self.head(x, ..., self.norm)`). The reference `RMSNorm.forward` body is at `model.py:191-196`. This kernel matches the reference exactly: fp32 reduction, weight applied after `rsqrt`, dtype-preserving final cast.

```python
# PyTorch-operator equivalent of one CTA's work for token t:
#
# Inputs:
#   hidden_states:  [T, H]  bf16     (H = HIDDEN)
#   weight:         [H]     bf16
#   eps:            float
#
x_in = hidden_states[t, :].to(torch.float32)         # [H]
w    = weight.to(torch.float32)                      # [H]
variance = (x_in * x_in).sum() / H                   # scalar fp32
rrms = torch.rsqrt(variance + eps)                   # scalar fp32
out[t, :] = (x_in * rrms * w).to(torch.bfloat16)     # cast at store
```

Notes on fusion / quant:
- **Lightweight kernel**: this is the *plain* per-token RMSNorm without any extra fusion. It exists as a separate Triton kernel (rather than reusing `torch.ops._C.rms_norm`) for two reasons stated in the source docstring (lines 131-132):
  1. Consistency: shares `_rmsnorm_row` with `fused_mtp_input_rmsnorm`, so the MTP draft path runs ONE consistent RMSNorm implementation end-to-end (matches the fp32-reduction-then-bf16-store convention exactly).
  2. CUDA-graph friendliness: lives in the same module as the other MTP fused ops, simplifying graph capture.
- No quant. bf16 in / bf16 out. fp32 accumulator (`_rmsnorm_row` line 35: `x.to(tl.float32)`).
- Cast happens once per row at the store (`tl.store(out_row_ptr + block, y.to(out_row_ptr.dtype.element_ty), ...)`, line 40).

## Config-dependent dispatch

- Activation condition: always on for MTP `compute_logits`. Skipped only if MTP is not built (e.g., `num_speculative_tokens == 0`).
- Variants: none. Single Triton kernel, no SM90/SM100 split, no Class A/B branch.
- Downstream consumer constraint:
  - Output feeds `LogitsProcessor(self.shared_head.head, hidden_states)` at `mtp.py:252`. `shared_head.head` is a `ParallelLMHead` (instantiated inside `SharedHead`). The lm_head GEMM (`linear_cublas.md` row `shared_head.head`) requires the input to be `[T, H]` bf16 row-major contiguous — exactly what `torch.empty_like(hidden_states)` produces.
- Hard preconditions (wrapper asserts at lines 134-136):
  - `hidden_states.ndim == 2`
  - `hidden_states.is_contiguous()`
  - `weight.is_contiguous()`

## Relationship to `rms_norm`

This kernel is *semantically* equivalent to the generic `torch.ops._C.rms_norm` (see `rms_norm.md`). Both produce identical bit patterns on bf16 inputs with the same eps and weight (modulo accumulator-order rounding within the fp32 reduction — Triton's `tl.sum` and CUB's `BlockReduce` are both tree reductions; precision typically matches within 1 ULP at the bf16 store).

The MTP draft path uses this Triton variant instead of the CUDA `_C.rms_norm` to keep the entire MTP forward in a single Triton-friendly module (`fused_mtp_input_rmsnorm.py`) and to share JIT-compiled cache with the more-complex `_fused_mtp_input_rmsnorm_kernel` that lives in the same file.
