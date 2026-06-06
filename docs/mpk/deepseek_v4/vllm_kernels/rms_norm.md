# rms_norm

## Identity
- Source file (Python wrapper): `vllm/model_executor/layers/layernorm.py:37-128` (`class RMSNorm(CustomOp)`).
- Source file (semantic op): `vllm/ir/ops/layernorm.py:9-21` (`@register_op def rms_norm(...)`).
- Source file (Python op binding): `vllm/_custom_ops.py:316-320` (`def rms_norm(out, input, weight, epsilon)` → `torch.ops._C.rms_norm`).
- Source file (CUDA kernel body): `csrc/libtorch_stable/layernorm_kernels.cu:13-85` (`__global__ rms_norm_kernel<scalar_t, VEC_SIZE, NUM_DIMS>`); launch wrapper at lines 191-237.
- Language/DSL: **CUDA C++** (templated, vectorized).
- Third-party dep: CUB (`cub::BlockReduce<float, 1024>` for the variance reduction at `layernorm_kernels.cu:61-63`).
- Registered as opaque custom op: `torch.ops._C.rms_norm` (residual-free) and `torch.ops._C.fused_add_rms_norm` (with residual; `csrc/libtorch_stable/layernorm_kernels.cu:91-187`). V4 NVIDIA only uses the residual-free path.
- Dispatch (V4-Flash NVIDIA): `RMSNorm.__call__` → `forward_cuda` (layernorm.py:104-116) → `forward_native` (line 116) → `ir.ops.rms_norm` (layernorm.py:89-94) which is registered with priority `[cuda, native]`; the CUDA implementation is the kernel above. `VLLM_BATCH_INVARIANT=1` would divert to `rms_norm_batch_invariant`; not used in the V4 production path.

## Call sites

All V4-Flash NVIDIA call sites that instantiate or invoke `RMSNorm` (residual-free, `residual=None`):

| Caller file:line | Module / function | Instance name | Input shape | Hidden size | Notes |
| --- | --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/nvidia/model.py:666` | `DeepseekV4Attention.__init__` | `q_norm` | `[T, q_lora_rank]` | 1536 | **Fused into `fused_q_kv_rmsnorm` Triton kernel** at `attention.py:422`; this `RMSNorm` instance owns the `weight` parameter but its `forward` is never called (weight is read directly as `self.q_norm.weight.data`). |
| `vllm/models/deepseek_v4/nvidia/model.py:676` | `DeepseekV4Attention.__init__` | `kv_norm` | `[T, head_dim]` | 512 | Same: weight read by `fused_q_kv_rmsnorm`, no standalone forward. |
| `vllm/models/deepseek_v4/nvidia/model.py:809` | `DeepseekV4DecoderLayer.__init__` | `attn_norm` | `[T, hc_mult, hidden_size]` | 4096 | **Fused into `mhc_pre_tilelang` / `mhc_fused_post_pre_tilelang`** (TileLang) at `nvidia/model.py:884-885,903-904`; weight read as `self.attn_norm.weight.data` only. |
| `vllm/models/deepseek_v4/nvidia/model.py:810` | `DeepseekV4DecoderLayer.__init__` | `ffn_norm` | `[T, hc_mult, hidden_size]` | 4096 | **Fused into `mhc_fused_post_pre_tilelang`** at `nvidia/model.py:927-928`; weight read only. |
| `vllm/models/deepseek_v4/nvidia/model.py:992,1103` | `DeepseekV4Model.__init__` / `.forward` | `norm` (final pre-logits norm) | `[T, hidden_size]` | 4096 | **Standalone call** at `nvidia/model.py:1103`: `hidden_states = self.norm(hidden_states)`. This is the only V4-trunk site that actually launches the `torch.ops._C.rms_norm` kernel. |
| `vllm/models/deepseek_v4/nvidia/mtp.py:82` | `DeepSeekV4MultiTokenPredictorLayer.__init__` | `enorm` | `[T, hidden_size]` | 4096 | **Fused into `fused_mtp_input_rmsnorm`** Triton kernel at `mtp.py:144` (pos==0 masking + dual norm). Weight read only. |
| `vllm/models/deepseek_v4/nvidia/mtp.py:83` | `DeepSeekV4MultiTokenPredictorLayer.__init__` | `hnorm` | `[T, hc_mult, hidden_size]` | 4096 | **Fused into `fused_mtp_input_rmsnorm`** (same kernel as enorm). Weight read only. |
| `vllm/models/deepseek_v4/nvidia/mtp.py` (instantiated inside `SharedHead`; called at `mtp.py:247-251`) | `DeepSeekV4MultiTokenPredictor.compute_logits` | `shared_head.norm` | `[T, hidden_size]` | 4096 | **Fused into `mtp_shared_head_rmsnorm`** Triton kernel at `mtp.py:247`. Weight read only. |
| `vllm/models/deepseek_v4/compressor.py:234` | `DeepseekCompressor.__init__` | `compressor.norm` | `[T, coff*compress_ratio, head_dim]` | 512 (MLA) / 128 (indexer) | **Fused into `compress_norm_rope_store_*`** (Triton/CuteDSL) — weight read only. |

**Effective production call count on V4-Flash NVIDIA**: exactly one standalone `torch.ops._C.rms_norm` launch per forward — the final trunk norm at `nvidia/model.py:1103`. All other `RMSNorm` instances are present as parameter containers whose `weight.data` (and `variance_epsilon`) feed downstream fused kernels (Triton, CuteDSL, TileLang, or the CUDA `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` op).

## Inputs

For the single live call site (`self.norm(hidden_states)` at `nvidia/model.py:1103`):

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `input` (kernel `input`) | `[num_tokens, hidden_size = 4096]` | bf16 | row-major, contiguous on the last dim (asserted at `layernorm_kernels.cu:196-199`) | Hidden state after `hc_head_fused_kernel_tilelang` (final HC head fuse) |
| `weight` | `[hidden_size = 4096]` | bf16 (matches `input.dtype`; see `RMSNorm.__init__` line 64 with default dtype) | contiguous | Learned gain; `self.norm.weight.data` |
| `epsilon` | scalar | float (double in op signature, cast to float at line 24) | — | `config.rms_norm_eps` |
| `out` | `[num_tokens, hidden_size]` | bf16 | allocated by `ir.ops.rms_norm` as `torch.empty_like(x)`; contiguous (asserted line 195) | Output buffer |

The kernel supports `NUM_DIMS ∈ {2, 3, 4}` via the `VLLM_STABLE_DISPATCH_RANK234` macro (line 218); V4's call is `NUM_DIMS=2`.

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `out` | `[num_tokens, hidden_size]` | bf16 | row-major contiguous (`out.is_contiguous()` is asserted) | RMS-normalized hidden state ready for `lm_head` (`logits_processor` path). |

## Grid / Block

- `grid_dim = (num_tokens,)` — one CTA per token (`layernorm_kernels.cu:214`).
- `block_dim`:
  - Selected as `block_size = min(hidden_size / vec_size, max_block_size)` where `max_block_size = 1024` when `num_tokens < 256` else `256` (lines 213, 223-225).
  - `vec_size = gcd(16 / sizeof(scalar_t), hidden_size)`; for bf16 (2 bytes) and `hidden_size=4096`, `vec_size = gcd(8, 4096) = 8`.
  - V4 trunk norm (`hidden_size=4096`, bf16) → `vec_size=8`, candidate block `= 512`; clamped to `max_block_size`. For decode (`num_tokens` small, often 1) → `block_size = min(512, 1024) = 512`. For larger prefills (`num_tokens ≥ 256`) → `block_size = min(512, 256) = 256`.
- Autotune configs: **none** — vec/block sizes are computed at launch, not autotuned. The macro `VLLM_STABLE_DISPATCH_VEC_SIZE` selects from `{1, 2, 4, 8}` at JIT/template-instantiation time.
- Per-CTA work: a single CTA accumulates `Σ x_i²` over the `hidden_size` lanes with vectorized loads (`vectorize_read_with_alignment<VEC_SIZE>`, line 58-59), reduces with `cub::BlockReduce<float, 1024>` (line 61-63, sized for the max possible block), broadcasts `s_variance = rsqrt(Σx²/H + eps)` via a single-element shared scalar, then writes `out[i] = (scalar_t)(x[i] * s_variance) * weight[i]` vectorized (lines 70-84). All accumulation in fp32; only the final store is cast back to `scalar_t`.
- The cast pattern `(scalar_t)(x * s_variance) * weight[i]` (line 81) — note the cast happens BEFORE the weight multiply — is the exact convention the V4 fused-RMSNorm Triton kernels copy (see `fused_q_kv_rmsnorm` doc line 52 referencing this file).

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:191-196` (`class RMSNorm.forward`). The V4-Flash reference RMSNorm matches this kernel exactly: fp32 reduction, weight applied after the square-root scale, dtype-preserving final cast.

For the trunk norm at `nvidia/model.py:1103` corresponds to reference `model.py:788` (`self.norm = RMSNorm(args.dim, self.norm_eps)`) and `model.py:722` (`self.get_logits(norm(x))`).

```python
# PyTorch-operator equivalent of one CTA's work for token t:
#
# Inputs:
#   x:        [T, H]  bf16     (H = hidden_size)
#   weight:   [H]     bf16
#   eps:      float
#
x_in = x[t, :].to(torch.float32)            # [H]
variance = (x_in * x_in).sum() / H          # scalar fp32
s_variance = torch.rsqrt(variance + eps)    # scalar fp32

# CUDA kernel does (scalar_t)(x * s_variance) * weight[i]: cast to bf16
# happens BEFORE multiplying with the bf16 weight (layernorm_kernels.cu:81).
out[t, :] = (x_in * s_variance).to(torch.bfloat16) * weight
```

Notes on fusion / quant:
- This kernel does NOT quantize. `rms_norm_dynamic_per_token_quant` (`_custom_ops.py:412-428`) and `rms_norm_per_block_quant` (lines 432+) are separate fused-quant variants — not used by V4 (V4's quant is folded into other fused kernels like `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert`).
- The `fused_add_rms_norm` variant (kernel at `layernorm_kernels.cu:91-187`) fuses residual addition into the same pass and writes back to `input` in place. **Not used by V4**: V4's residual path is folded into `mhc_fused_post_pre_tilelang` (TileLang) instead.
- `variance_size_override` (RMSNorm.__init__ line 59-61) reduces variance over only a leading slice of the last dim; **not used by V4** (all V4 RMSNorm instances use full-hidden variance).

## Config-dependent dispatch

- Activation condition: always on. Reachable on every V4-Flash NVIDIA forward at `nvidia/model.py:1103` (PP-last rank only — non-last ranks return `IntermediateTensors` at line 1088-1089 without calling `self.norm`).
- Variants: none on the production path. Three CustomOp-level dispatches exist but are inactive for V4-Flash NVIDIA:
  - `forward_cuda` (CUDA) vs `forward_native` (PyTorch fallback) — `RMSNorm.forward_cuda` (layernorm.py:104-116) actually just delegates to `forward_native` for the V4 case (no batch-invariant, no residual). The CUDA kernel is reached via `ir.ops.rms_norm`'s priority list (`priority.rms_norm[0]` defaults to `cuda` for SM100; layernorm.py:75-77).
  - `VLLM_BATCH_INVARIANT=1` → `rms_norm_batch_invariant` (line 114). Off by default; not used in production.
  - `forward_xpu` → `forward_cuda`. Not applicable on B200.
- Downstream consumer constraints:
  - The single live output feeds `lm_head` (ParallelLMHead, `nvidia/model.py:1282`) via `LogitsProcessor` (`nvidia/model.py:1301` → `logits_processor.py:96`). Layout requirement: `[T, hidden_size]` bf16 row-major contiguous — exactly what the kernel produces (`out.is_contiguous()` asserted at line 195).
- Hard preconditions (asserted at `layernorm_kernels.cu:195-200`):
  - `out.is_contiguous()`
  - `input.stride(-1) == 1` (otherwise the kernel calls `torch::stable::contiguous(input)` at line 197 to make it so).
  - `weight.is_contiguous()`
  - All three tensors must be the same floating-point dtype (`VLLM_STABLE_DISPATCH_FLOATING_TYPES` at line 219 dispatches on `input.scalar_type()`; `RMSNorm.weight` is allocated in the default dtype, layernorm.py:62-64).
