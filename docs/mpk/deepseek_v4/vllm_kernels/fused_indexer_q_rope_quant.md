# fused_indexer_q_rope_quant

## Identity
- Source file: `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:69-172` (Triton JIT body `_fused_indexer_q_rope_quant_kernel`); helper `_get_cos_sin` at lines 13-25.
- Launch wrapper: `fused_indexer_q_rope_quant` Python function at `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:284-438` (FP8 branch is the `not use_fp4` path, lines 400-437).
- Language/DSL: **Triton** (`@triton.jit`).
- Third-party dep: none (pure Triton + PyTorch).
- Registered as opaque custom op: **no** — called directly from `DeepseekV4Indexer.forward`.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/attention.py:841` (via `fused_indexer_q_rope_quant(..., use_fp4=self.use_fp4_kv)` selecting Triton FP8 branch when CuteDSL is absent and `use_fp4_kv==False`) | `DeepseekV4Indexer.forward` → `wq_b_and_q_quant` closure | `q: [num_tokens, n_heads=64, head_dim=128] bf16` (raw output of `wq_b` reshape at attention.py:840), `positions: [num_tokens] int64`, `cos_sin_cache: [max_pos, rope_head_dim=64] fp32`, `index_weights: [num_tokens, n_heads=64] bf16` (raw output of `weights_proj`) | inputs bf16, outputs fp8e4m3 Q + fp32 folded weights | `use_fp4_kv == False` AND `has_cutedsl() == False`; V4-Flash NVIDIA path uses CuteDSL when available |

Per-token-per-head scalar Q scale is **folded into output weights** (no separate Q-scale tensor emitted; see "Weight-fold contract" in source docstring lines 297-319 and the kernel body line 153-167 comment).

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `pos_ptr` | `[num_tokens]` | int64 | contiguous | Global positions for RoPE. |
| `index_q_ptr` | `[num_tokens, n_heads, INDEX_Q_HEAD_DIM]` | bf16 | row-major; stride0/stride1 passed as args | Pre-quant Q (output of `wq_b` reshaped at attention.py:840). |
| `index_q_cos_sin_ptr` | `[max_pos, rope_head_dim = 64]` | fp32 | first half cos, second half sin | RoPE cache **shared with the compressor** — built from `compress_rope_theta=160000` (passed as `rotary_emb.cos_sin_cache` at attention.py:844). NOT the main attention RoPE cache. |
| `INDEX_Q_HALF_ROT_DIM` | constexpr int = 32 | — | — | `cos_sin_cache.shape[-1] // 2`. |
| `index_q_fp8_ptr` | `[num_tokens, n_heads, INDEX_Q_HEAD_DIM]` | float8_e4m3fn | row-major | Output FP8 Q (allocated by wrapper at line 400). |
| `INDEX_Q_HEAD_DIM` | constexpr int = 128 | — | — | Indexer head dim. |
| `index_weights_ptr` | `[num_tokens, n_heads]` | bf16 (loaded as fp32) | row-major | Per-(token, head) raw weights from `weights_proj` (attention.py:846 passes `indexer_weights`). |
| `index_weights_stride` | scalar | int | — | `index_weights.stride(0)`. |
| `index_weights_softmax_scale` | scalar | fp32 | — | `head_dim ** -0.5 = 128**-0.5 ≈ 0.0884`. Passed as `self.softmax_scale` from attention.py:846. |
| `index_weights_head_scale` | scalar | fp32 | — | `n_heads ** -0.5 = 64**-0.5 = 0.125`. Passed at attention.py:847. |
| `index_weights_out_ptr` | `[num_tokens, n_heads]` | fp32 | row-major | Output folded weights (allocated as `torch.empty_like(index_weights, dtype=torch.float32)` at line 329). |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `index_q_fp8` (in-place via `index_q_fp8_ptr`) | `[num_tokens, n_heads, 128]` | `float8_e4m3fn` | row-major | FP8-quantized Q. Per-(token, head) scalar scale `q_scale` is NOT stored — it is folded into `index_weights_out` instead. |
| `index_weights_out` (in-place) | `[num_tokens, n_heads]` | fp32 | row-major | Folded weights: `index_weights × q_scale × softmax_scale × head_scale`. Consumed by `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits` as the `weights` argument. |

Wrapper return (line 438): `(index_q_fp8, index_weights_out)`. The downstream `SparseAttnIndexer.forward_*` passes `(q_fp8, None)` as the `q` tuple to DeepGEMM (Q-scale slot is `None` for FP8 — `sparse_attn_indexer.py:206-209`).

## Grid / Block

- `grid_dim = (num_tokens, n_heads) = (T, 64)` — one CTA per (token, head). Launch at wrapper line 418.
- `block_dim`: `num_warps=1` (wrapper line 436, with `# TODO: Tune this`), so 32 threads/CTA.
- Autotune configs: **none** — `num_warps=1` hard-coded.
- Per-CTA work:
  1. Load one full `[INDEX_Q_HEAD_DIM = 128]` head's Q and split into nope (`[0, NOPE_DIM = HEAD_DIM - 2*HALF_ROT_DIM = 64)`) and rope (`[NOPE_DIM, HEAD_DIM)`).
  2. Apply forward GPT-J RoPE to the rope half using `cos_sin_cache[position]`; bf16 roundtrip on `(r_even, r_odd)`.
  3. Compute `amax` over `(r_even, r_odd, x_nope)`; derive `q_scale = 2^ceil(log2(max(amax, 1e-4) / 448.0))` (UE8M0-discrete scalar).
  4. Quant-divide-and-cast → write 128 fp8e4m3 bytes (`tl.div_rn(x, q_scale).to(tl.float8e4nv)`).
  5. Fold `q_scale × softmax_scale × head_scale` into the per-(token, head) weight and store.
- Stride args are dynamic Python ints; head_dim / HALF_ROT_DIM constexpr.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:411-417` (`Indexer.forward`):
```python
q = self.wq_b(qr)
q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
apply_rotary_emb(q[..., -rd:], freqs_cis)        # rope on tail
q = rotate_activation(q)                          # Hadamard
fp4_act_quant(q, fp4_block_size, True)           # ← reference is FP4; vLLM Triton FP8 variant skips Hadamard upstream
```
The vLLM FP8 path does NOT do the Hadamard rotation in this kernel (it relies on the FP8 cache pair which doesn't need it). The reference's `fp4_act_quant` is replaced by FP8-block quant with the per-token-per-head scale folded into weights.

```python
# Per-CTA work for token t, head h (PyTorch-operator equivalent):
pos = positions[t]                                                 # int64
cos = cos_sin_cache[pos, : 32].to(torch.float32)                   # [32]
sin = cos_sin_cache[pos, 32 :].to(torch.float32)                   # [32]

q_full = index_q[t, h, :].to(torch.float32)                        # [128]
nope   = q_full[: 64]                                              # NOPE_DIM = HEAD_DIM - 2*HALF_ROT_DIM = 128 - 64 = 64
rope   = q_full[64 :]                                              # [64]

# GPT-J interleaved RoPE on the rope tail (even/odd pairs):
x_even = rope[0::2]                                                # [32]
x_odd  = rope[1::2]                                                # [32]
r_even = x_even * cos - x_odd * sin
r_odd  = x_odd  * cos + x_even * sin
# bf16 roundtrip for parity with reference / K-side compressor numerics.
r_even = r_even.to(torch.bfloat16).to(torch.float32)
r_odd  = r_odd.to(torch.bfloat16).to(torch.float32)

# Scalar per-(token, head) amax over (rope_even, rope_odd, nope).
amax = max(r_even.abs().amax(), r_odd.abs().amax(), nope.abs().amax())
q_scale = torch.exp2(torch.ceil(torch.log2(max(amax, 1e-4) / 448.0)))   # UE8M0-discrete

# FP8 quant (divide-and-cast; no clamp needed because q_scale = amax/448 rounded up).
index_q_fp8[t, h, :64]      = (nope   / q_scale).to(torch.float8_e4m3fn)
index_q_fp8[t, h, 64::2]    = (r_even / q_scale).to(torch.float8_e4m3fn)
index_q_fp8[t, h, 65::2]    = (r_odd  / q_scale).to(torch.float8_e4m3fn)

# Weight-fold: per-token-per-head scalar q_scale ABSORBED into weights.
# Rationale (line 153-160 comment): FP8 Q is stored WITHOUT a companion scale
# tensor; downstream fp8_fp4_mqa_logits applies `weights` to the logits as an
# in-line per-token Q dequant. So q_scale lives on the weights side.
w = index_weights[t, h].to(torch.float32)
index_weights_out[t, h] = w * q_scale * softmax_scale * head_scale
```

Notes on fusion / numerics:
- **`tl.div_rn` then cast**: kernel uses `tl.div_rn(x, scale).to(tl.float8e4nv)`. The division is round-to-nearest fp32, the cast is to e4m3 with saturation. No explicit clamp needed because `scale = 2^ceil(log2(amax/448)) ≥ amax/448`, so `x/scale ∈ [-1, 1] · 448 ⊆ e4m3_range`.
- **Same RoPE cache as the K-side compressor**: `rotary_emb.cos_sin_cache` passed at `attention.py:844` is the indexer compressor's RoPE cache (built with `compress_rope_theta=160000`). Q and K must use the same RoPE base or scoring is garbage.
- **No Hadamard `rotate_activation`**: the reference applies a Hadamard transform pre-quant; the vLLM FP8 Triton path does not — the assumption is that downstream FP8 MQA logits has enough precision without it. (The MXFP4 sibling DOES need a different pre-quant routine but also skips the Hadamard.)
- **Weight-fold semantics**: critical contract — the per-token-per-head scalar `q_scale` is the SOLE scale for FP8 Q. If MPK reimplements this kernel, the weight tensor must arrive at DeepGEMM `fp8_fp4_mqa_logits` already containing this factor, or the logits will be scaled wrong by a factor of `q_scale`.
- **bf16 roundtrip parity** (line 122-124 comment): matches the K-side compressor's `_fused_kv_compress_norm_rope_insert_indexer_attn` so Q and K have aligned absmax statistics.

## Config-dependent dispatch

- Activation condition: `use_fp4_kv == False` AND `has_cutedsl() == False`.
  - On NVIDIA when `has_cutedsl()` returns True (B200 fast path), vLLM dispatches to `fused_indexer_q_rope_quant_fp8_cutedsl` (`IndexerQFp8Kernel`) instead — see wrapper lines 401-416.
- Variants (**Class B** per D3 — FP8 vs MXFP4; D2 — Triton vs CuteDSL):
  - **MXFP4 Q-side sibling (this kernel's pair under `use_fp4_cache`)**: `_fused_indexer_q_rope_mxfp4_kernel` at `fused_indexer_q.py:174-282` (spec `fused_indexer_q_rope_mxfp4.md`).
  - **`use_fp4_cache` coupling (D3)**: this FP8 Q-side variant is selected when `use_fp4_kv = self.vllm_config.attention_config.use_fp4_indexer_cache = False`. The flag is shared with K-side — selecting FP8 Q here means the K-side **must** also be FP8 (`_fused_kv_compress_norm_rope_insert_indexer_attn`, spec `fused_kv_compress_norm_rope_insert_indexer_attn.md`). Cannot mix: DeepGEMM `fp8_fp4_mqa_logits` discriminates dtypes via the `(q_values, q_scale)` tuple — `q_scale=None` selects FP8, `q_scale: int32` selects MXFP4 — and the K dtype must match.
  - Coupled K-side spec: `fused_kv_compress_norm_rope_insert_indexer_attn.md`.
  - **CuteDSL sibling (locked active path on NVIDIA, D2)**: `IndexerQFp8Kernel` in `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py:428-610` — locked-alternative pointer; this Triton kernel is the universal fallback. The CuteDSL variant has identical I/O contract and identical weight-fold semantics; only the launch + register-allocation strategy differs.
- Class: **B** — part of a 4-variant coupled cluster (`use_fp4_cache`) × also Triton vs CuteDSL (this is the Triton path).
- Downstream consumer: `SparseAttnIndexer.forward_*` passes the returned `(index_q_fp8, index_weights_out)` (with `q_scale=None`) into `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits`. The folded weights are loaded as `tensor_map_weights` and applied per-token to the logits inside the DeepGEMM kernel.
