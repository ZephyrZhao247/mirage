# fused_mtp_input_rmsnorm

## Identity
- Source file: `vllm/models/deepseek_v4/common/ops/fused_mtp_input_rmsnorm.py:43-97` (Triton JIT body `_fused_mtp_input_rmsnorm_kernel`); shared helper `_rmsnorm_row` at lines 25-40; user-facing wrapper `fused_mtp_input_rmsnorm` at lines 153-203.
- Language/DSL: **Triton** (`@triton.jit`).
- Third-party dep: none (pure Triton + PyTorch).
- Registered as: Python-level function (no `torch.ops` op); imported from `vllm.models.deepseek_v4.common.ops` package.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/nvidia/mtp.py:144` | `DeepSeekV4MultiTokenPredictorLayer.forward` | `inputs_embeds: [T, H=4096]`, `positions: [T] int64`, `previous_hidden_states: [T, hc_mult, H]` (reshaped from `[T, hc_mult*H]` on line 140-142) | bf16 activations | always on per MTP draft step (single call per layer per `forward()`) |

Single call site. Runs once per MTP draft step, immediately after the trunk model's `_mtp_hidden_buffer` is read and reshaped, and feeds into `h_proj` / `e_proj` (line 153-155) which sum to the input of the MTP decoder block.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `inputs_embeds` (`inputs_embeds_ptr`) | `[num_tokens, HIDDEN]` (V4-Flash: `HIDDEN=4096`) | bf16 | row-major, **contiguous** (asserted at line 180) | Token embeddings for the next-step input IDs (output of `mtp.embed_tokens` at `mtp.py:217`). |
| `positions` (`positions_ptr`) | `[num_tokens]` | int64 | contiguous | Absolute token positions. Tokens at `position == 0` are mask-zeroed before enorm to match the reference semantics (no embedding contribution for the BOS token of a fresh sequence). |
| `previous_hidden_states` (`prev_hidden_ptr`) | `[num_tokens, HC_MULT, HIDDEN]` (V4-Flash: `HC_MULT=8`, `HIDDEN=4096`) | bf16 | row-major, contiguous (asserted line 180) | Last-layer hidden state from the *target* model, reshaped from the flat `[T, hc_mult*H]` buffer (`mtp.py:140-142`). |
| `enorm_weight` (`enorm_weight_ptr`) | `[HIDDEN]` | bf16 (V4-Flash RMSNorm default dtype) | contiguous (asserted line 181) | RMSNorm gain for `inputs_embeds` (the `enorm` stream). `self.enorm.weight.data`. |
| `hnorm_weight` (`hnorm_weight_ptr`) | `[HIDDEN]` | bf16 | contiguous (asserted line 181) | RMSNorm gain for `previous_hidden_states` (the `hnorm` stream). `self.hnorm.weight.data`. |
| `eps` | scalar float | — | — | `self.enorm.variance_epsilon` (== `config.rms_norm_eps`, also used by hnorm; both norms are instantiated with the same eps at `mtp.py:82-83`). |
| `HIDDEN` | scalar (`tl.constexpr`) | int | — | `inputs_embeds.shape[1]`, baked at JIT time. |
| `HC_MULT` | scalar (`tl.constexpr`) | int | — | `hc_mult` (8 for V4-Flash). |
| `BLOCK_SIZE` | scalar (`tl.constexpr`) | int | — | `triton.next_power_of_2(HIDDEN)` — V4-Flash: `4096`. |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `enorm_out` (`enorm_out_ptr`) | `[num_tokens, HIDDEN]` | bf16 (matches `inputs_embeds.dtype`) | `torch.empty_like(inputs_embeds)` — contiguous, row-major | Mask-zeroed + RMS-normalized token embedding. Ready for `self.e_proj(...)` at `mtp.py:154`. |
| `hnorm_out` (`hnorm_out_ptr`) | `[num_tokens, HC_MULT, HIDDEN]` | bf16 | `torch.empty_like(previous_hidden_states)` | Per-HC-slot RMS-normalized previous hidden state. Ready for `self.h_proj(...)` at `mtp.py:153`. |

Early-return: when `num_tokens == 0` the wrapper returns empty allocations without launching (lines 186-187).

## Grid / Block

- `grid_dim = (num_tokens, HC_MULT + 1)` — outer dim is the token, inner dim selects task: `pid_task == 0` → enorm; `pid_task ∈ [1, HC_MULT]` → hnorm at slot `pid_task - 1`. V4-Flash: `(T, 9)` CTAs.
- `block_dim`: default Triton `num_warps` (no explicit override). For `BLOCK_SIZE=4096`, Triton defaults to `num_warps=4` (128 threads/CTA).
- `num_stages`: default (no override).
- Autotune configs: **none** — `BLOCK_SIZE = next_power_of_2(HIDDEN)` is computed once at launch and baked as `tl.constexpr`. Single tile per CTA covers `HIDDEN` lanes with `mask = block < HIDDEN`.
- Per-CTA work: a single CTA handles one (token, task) pair end-to-end. Loads `HIDDEN` bf16 values, computes fp32 variance via `tl.sum(x * x, axis=0) / HIDDEN` (kernel line 36), applies `rsqrt(var + eps) * w` (line 37-39), stores back as `out_row_ptr.dtype.element_ty` (bf16). The shared body `_rmsnorm_row` (lines 25-40) is identical to the helper used by `mtp_shared_head_rmsnorm` — single PTX module covers both kernels' norm logic.
- Int64 index trick: `token_idx = tl.program_id(0).to(tl.int64)` (line 59) — same as `fused_q_kv_rmsnorm`; needed because `token_idx * HIDDEN * HC_MULT` overflows int32 once `T ≳ 65536` for `HIDDEN=4096, HC_MULT=8`.
- **enorm branch** (lines 65-83): loads `positions[token_idx]`, applies `keep = (pos != 0)`, loads `inputs_embeds[token_idx, :]`, zero-masks the row via `tl.where(keep, x, 0.0)`, then runs `_rmsnorm_row` writing to `enorm_out_ptr + token_idx * HIDDEN`. Position-0 → all-zero x → variance=0 → output is zero regardless of weight (kernel comment lines 67-68).
- **hnorm branch** (lines 84-97): computes `slot = pid_task - 1`, loads `prev_hidden[token, slot, :]` at flat offset `(token_idx * HC_MULT + slot) * HIDDEN`, runs `_rmsnorm_row` writing to `hnorm_out + (token_idx * HC_MULT + slot) * HIDDEN`. No position-based masking — every (token, slot) is normalized unconditionally.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:758-764` (`MTPBlock.forward`):
```
e = self.embed(input_ids)
e = self.enorm(e)
x = self.hnorm(x)
x = self.e_proj(e).unsqueeze(2) + self.h_proj(x)
```
The kernel jointly executes the `enorm(e)` and `hnorm(x)` steps **with an extra pos==0 mask on `e`** that the reference does NOT have. The mask is a vLLM-side semantic add to fix the BOS-token contribution issue (see file docstring lines 3-15: "Math is preserved: positions==0 → masked row → zero RMS output regardless of weight").

The reference's `RMSNorm.forward` is at `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:191-196`. This kernel matches the reference body: fp32 reduction, weight applied after `rsqrt`, dtype-preserving final cast.

```python
# PyTorch-operator equivalent of one CTA's work for token t, task ∈ {0, 1, ..., HC_MULT}:
#
# Inputs:
#   inputs_embeds:       [T, H]            bf16    (H = HIDDEN)
#   positions:           [T]               int64
#   previous_hidden:     [T, HC_MULT, H]   bf16
#   enorm_weight:        [H]               bf16
#   hnorm_weight:        [H]               bf16
#
if pid_task == 0:
    # enorm branch
    pos = positions[t]
    x_in = inputs_embeds[t, :].to(torch.float32)
    x_in = torch.where(pos != 0, x_in, torch.zeros_like(x_in))   # mask-zero at pos==0
    w = enorm_weight.to(torch.float32)
    variance = (x_in * x_in).sum() / H
    rrms = torch.rsqrt(variance + eps)
    enorm_out[t, :] = (x_in * rrms * w).to(torch.bfloat16)
else:
    # hnorm branch
    slot = pid_task - 1
    x_in = previous_hidden[t, slot, :].to(torch.float32)
    w = hnorm_weight.to(torch.float32)
    variance = (x_in * x_in).sum() / H
    rrms = torch.rsqrt(variance + eps)
    hnorm_out[t, slot, :] = (x_in * rrms * w).to(torch.bfloat16)
```

Notes on fusion / quant:
- The "joint" fusion is at the **grid** level: a single launch covers `HC_MULT + 1` norms per token with one PTX module, saving `HC_MULT` separate launches. The body `_rmsnorm_row` (lines 25-40) is shared with `mtp_shared_head_rmsnorm` (`_mtp_shared_head_rmsnorm_kernel`, lines 100-121) so MTP runs one consistent RMSNorm impl end-to-end (file docstring lines 131-132).
- **Pos==0 mask fusion**: the file docstring (lines 3-15) explains the motivation — without this kernel the eager path is `torch.where(pos==0, 0, x); enorm(x); reshape; hnorm(prev)` which lowers to ~6 small kernels on the breakable-cudagraph path. This kernel collapses all of them.
- No quant. bf16 in / bf16 out. fp32 accumulator throughout (`_rmsnorm_row` line 35 casts `x.to(tl.float32)` before reduction; weight cast to fp32 on load at line 38).
- The cast happens once per row at the store (`tl.store(out_row_ptr + block, y.to(out_row_ptr.dtype.element_ty), ...)`, line 40).

## Config-dependent dispatch

- Activation condition: always on for MTP. The MTP draft is the only consumer; if `num_speculative_tokens == 0`, the MTP module is not built and this kernel is never called.
- Variants: none. Single Triton kernel, no SM90/SM100 split, no Class A/B branch.
- Downstream consumer constraints:
  - `enorm_out` feeds `self.e_proj(inputs_embeds)` at `mtp.py:154` (ReplicatedLinear, bf16 → bf16). Layout requirement: `[T, H]` bf16 row-major contiguous (matches `torch.empty_like(inputs_embeds)`).
  - `hnorm_out` feeds `self.h_proj(previous_hidden_states)` at `mtp.py:153` (ReplicatedLinear). Layout requirement: `[T, HC_MULT, H]` bf16 contiguous; PyTorch's `nn.Linear` flattens the leading dims internally — no shape constraint beyond contiguity on the last dim.
- Hard preconditions (wrapper asserts at lines 168-181):
  - `inputs_embeds.ndim == 2`
  - `previous_hidden_states.ndim == 3` and `previous_hidden_states.shape[1] == hc_mult`
  - `inputs_embeds.shape[0] == previous_hidden_states.shape[0]` (token dim match)
  - All four feature dims equal (`inputs_embeds.shape[1] == previous_hidden_states.shape[2] == enorm_weight.shape[0] == hnorm_weight.shape[0]`)
  - All four tensors `is_contiguous()`
