# logits_processor

## Identity
- Source file (Python wrapper): `vllm/model_executor/layers/logits_processor.py:18-104` (`class LogitsProcessor(PluggableLayer)`).
- Key sub-method: `_get_logits` at lines 89-104 (the actual lm_head dispatch); `forward` at lines 54-73 (post-processing).
- Language/DSL: **Python orchestration only** — `LogitsProcessor` is not itself a GPU kernel; it is a thin layer that chains (1) an lm_head GEMM, (2) optional TP-gather collective, (3) vocab-trim slice, and (4) optional soft-cap + scale. Each of these dispatches to its own kernel.
- Registered as: Python `PluggableLayer` (`@PluggableLayer.register("logits_processor")`, line 18).

This spec documents the GPU ops invoked by V4-Flash's call at `vllm/models/deepseek_v4/nvidia/model.py:1301` (`logits = self.logits_processor(self.lm_head, hidden_states)`), not a single kernel.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/nvidia/model.py:1301` | `DeepseekV4ForCausalLM.compute_logits` | `hidden_states: [num_tokens, hidden_size=4096]` | bf16 in / fp32 out (the GEMM output is bf16 then gather + slice; numerical type at the boundary depends on the lm_head quant method — V4's default `ParallelLMHead` is unquantized bf16, but the sampler downstream casts to fp32) | always on (PP-last rank, called by EngineCore from `model_runner` per step) |
| `vllm/models/deepseek_v4/nvidia/mtp.py:252` | `DeepSeekV4MultiTokenPredictor.compute_logits` | `hidden_states: [num_tokens, hidden_size=4096]` after `mtp_shared_head_rmsnorm` | bf16 in / bf16 out | always on per MTP draft step |

Both call sites instantiate with `LogitsProcessor(config.vocab_size)` (`nvidia/model.py:1289`, `mtp.py:203`), `scale = 1.0`, `soft_cap = None`. With these defaults, lines 66-72 are no-ops — the `forward` collapses to: `_get_logits(hidden_states, lm_head, embedding_bias)` then return.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `lm_head` | `ParallelLMHead` module | — | — | The lm_head layer (V4 trunk: `nvidia/model.py:1282`; V4 MTP: `mtp_layer.shared_head.head` at `mtp.py:252`). |
| `lm_head.weight` | `[vocab_size_per_partition, hidden_size]` = `[129280, 4096]` for `tp_size=1`; sharded by `tp_size` otherwise | bf16 | contiguous, row-major | LM head weight matrix. For V4 the head is tied via `ParallelLMHead.tie_weights` (only if the model opts in; V4 does NOT tie — see `nvidia/model.py:1282-1287` which constructs a separate ParallelLMHead). |
| `hidden_states` | `[num_tokens, hidden_size=4096]` | bf16 (after final `RMSNorm` at `nvidia/model.py:1103`, or after `mtp_shared_head_rmsnorm` for MTP) | contiguous, row-major | Per-token last-layer hidden state. |
| `embedding_bias` | `None` for V4 | — | — | Optional bias for the lm_head GEMM. V4 does not pass one (default `None` at `logits_processor.py:58`). |
| `self.scale` | `1.0` (V4 default) | float | — | Logit scale; applied at line 72 only if `!= 1.0`. |
| `self.soft_cap` | `None` (V4 default) | — | — | Gemma-2 soft-cap; line 66-69 inactive for V4. |
| `self.vocab_size` | `129280` | int | — | Padded vocab size used to trim logits at line 103. |
| `self.org_vocab_size` | `129280` | int | — | Same as `vocab_size` for V4 (no LoRA added vocab). |
| `self.use_all_gather` | `current_platform.use_all_gather()` (line 52) | bool | — | NVIDIA: `True` → use `tensor_model_parallel_all_gather`; `False` → `tensor_model_parallel_gather`. SM100 (B200): typically `True`. |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `logits` | `[num_tokens, vocab_size = 129280]` | bf16 (V4 default; later cast to fp32 in the sampler) | contiguous, row-major | Pre-softmax logits over the full vocabulary, sliced to `org_vocab_size`. Returned to the sampler. |

For `tp_size > 1`, the per-rank `local_logits` of shape `[num_tokens, vocab_size / tp_size]` is the GEMM output; `_gather_logits` (line 75-87) reduces to the final `[num_tokens, vocab_size]` via `all_gather` (on the active rank, the gathered tensor; on other ranks under non-allgather mode, `None`).

For `tp_size == 1`, no collective; `local_logits` IS `logits`.

## Grid / Block

`LogitsProcessor.forward` itself launches no GPU kernels. The kernels launched in sequence are:

### 1. lm_head GEMM (the dominant op)
- Op: `lm_head.quant_method.apply(lm_head, hidden_states, bias=None)` at line 96.
- For V4-Flash NVIDIA (`ParallelLMHead` is the unquantized case — V4 ships bf16 lm_head): the quant method is `UnquantizedLinearMethod`, which dispatches `torch.nn.functional.linear(hidden_states, lm_head.weight)` → cuBLAS `gemmEx`/`hgemm` (bf16 GEMM).
- See `linear_cublas.md` row `lm_head`. Shapes: `[T, 4096] @ [4096, 129280] = [T, 129280]` (or `[T, 4096] @ [4096, 129280/tp_size]` per-rank under TP).
- Grid/block: managed by cuBLAS, not user-configurable.

### 2. TP collective (only if `tp_size > 1`)
- `tensor_model_parallel_all_gather(logits)` at line 83 OR `tensor_model_parallel_gather(logits)` at line 86.
- Dispatches to NCCL (`ncclAllGather` or `ncclGather`). Grid/block managed by NCCL.
- Tensor shape: gathered `[num_tokens, vocab_size]` (concat along `dim=-1`).

### 3. Vocab trim slice (always)
- `logits[..., : self.org_vocab_size]` at line 103.
- For V4 `vocab_size == org_vocab_size == 129280` → this slice is a no-op view (zero-cost; PyTorch returns a strided view, no copy).

### 4. Soft-cap + scale (skipped for V4)
- Lines 66-72: `logits = tanh(logits / soft_cap) * soft_cap` then `logits *= scale`. Each is a pointwise op (PyTorch CUDA generic pointwise kernel, grid `ceil(numel/256)`, block 256). **Inactive for V4** (`soft_cap=None`, `scale=1.0`).

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:716-727` (`class ParallelHead.get_logits` and `forward`). The reference does `logits = F.linear(x[:, -1].float(), self.weight)` then `all_gather` (line 723-726). vLLM's `LogitsProcessor` performs the same logical operation:
- Reference: bf16 input → fp32 cast → fp32 weight GEMM → all_gather concat.
- vLLM V4: bf16 input → bf16 GEMM (with bf16 weight) → all_gather concat (or just gather on non-tensor-parallel-aware paths) → fp32 cast happens downstream in the sampler.

```python
# PyTorch-operator equivalent of V4-Flash NVIDIA logits computation:
#
# Inputs:
#   hidden_states:   [T, H=4096]      bf16  (output of final self.norm at nvidia/model.py:1103)
#   lm_head.weight:  [V_local, H]     bf16  (V_local = 129280 / tp_size)
#
# 1) lm_head GEMM (per TP rank)
local_logits = F.linear(hidden_states, lm_head.weight)   # [T, V_local] bf16  → cuBLAS

# 2) TP gather (skipped for tp_size == 1)
if tp_size > 1:
    if use_all_gather:
        logits = all_gather(local_logits, dim=-1)         # [T, V] bf16
    else:
        logits = gather(local_logits, dim=-1)             # [T, V] on rank 0, None on others
else:
    logits = local_logits

# 3) Vocab trim (no-op for V4 since org_vocab_size == vocab_size == 129280)
logits = logits[..., : org_vocab_size]

# 4) Scale / soft-cap — inactive for V4
# if soft_cap: logits = tanh(logits / soft_cap) * soft_cap
# if scale != 1: logits *= scale

return logits  # downstream sampler casts to fp32 before softmax
```

Notes on fusion / quant:
- **No fusion across the 4 steps.** Each is its own kernel launch. The GEMM dominates (V × H ≈ 530M FMAs/token).
- **No quantization** in V4's default config (bf16 lm_head). If a future V4 checkpoint shipped fp8 lm_head, the quant method's `apply` would dispatch to `Fp8LinearMethod` (fp8 GEMM + dequant); the rest of `LogitsProcessor` is unchanged.
- **No fused soft-cap + scale**: scale and soft_cap (when present) are each a separate kernel. Combined, they are still memory-bound on `[T, V]` and not the bottleneck.

## Config-dependent dispatch

- Activation condition: always on (PP-last rank for the trunk; every MTP draft step for the speculative decoder).
- Variants:
  - `tp_size > 1` vs `tp_size == 1` — the gather kernel is skipped for tp1.
  - `use_all_gather=True` vs `False` — selected by `current_platform.use_all_gather()` (line 52). On NVIDIA SM100 (B200) this depends on whether the platform reports XLA-style strict-SPMD (typically False on B200, so `gather` is used). The behavior difference: `all_gather` returns full logits on every rank; `gather` returns full logits only on rank 0 (else `None`).
  - `soft_cap is not None` (Gemma-2 style) — V4 does NOT set this.
  - `scale != 1.0` — V4 does NOT set this.
  - `logits_as_input=True` — bypasses the lm_head GEMM. V4 does NOT use this.
- The downstream sampler (not part of this spec) selects greedy vs sampling per-request; it consumes the fp32-cast logits.

## Related kernels referenced

- **lm_head GEMM**: see `linear_cublas.md` (row `lm_head` and `shared_head.head` for MTP).
- **Final RMSNorm** that produces `hidden_states`: see `rms_norm.md` (the one standalone call site at `nvidia/model.py:1103`).
- **MTP path**: the MTP draft replaces final RMSNorm with `mtp_shared_head_rmsnorm` — see `mtp_shared_head_rmsnorm.md`.
- **`get_top_tokens` fast path** (`logits_processor.py:106-`): vocab-parallel local argmax for the case where only the argmax is needed (skips full all-gather). Not used by V4 in the standard sampler path; mentioned for completeness.
