# mhc_pre_big_fuse_with_norm_tilelang

## Identity

- Source file: `vllm/model_executor/kernels/mhc/tilelang_kernels.py:197-354` (`mhc_pre_big_fuse_with_norm_tilelang`, TileLang `@tilelang.jit`).
- Launch wrapper: `vllm/model_executor/kernels/mhc/tilelang.py:232-251` inside `mhc_pre_tilelang` (norm_weight branch); also `vllm/model_executor/kernels/mhc/tilelang.py:528-547` inside `mhc_fused_post_pre_tilelang` when `num_tokens > 16 and norm_weight is not None`.
- Language/DSL: **TileLang** (`@tilelang.jit(pass_configs={TL_DISABLE_WARP_SPECIALIZED: True, TL_DISABLE_TMA_LOWER: True, TL_PTXAS_REGISTER_USAGE_LEVEL: 10})`).
- Third-party dep: `tilelang`.
- Registered as opaque custom op: parent wrapper `mhc_pre_tilelang` registered as `torch.ops.vllm.mhc_pre_tilelang` (`tilelang.py:658`).

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/model_executor/kernels/mhc/tilelang.py:232` | `mhc_pre_tilelang` (V4 first-layer entry; called once before first decoder layer) | `residual: [T, 4, 4096]`, `gemm_out_mul: [n_splits, T, 24]`, `norm_weight: [4096]` | bf16 residual + norm_weight, fp32 gemm-outs | `norm_weight is not None` |
| `vllm/model_executor/kernels/mhc/tilelang.py:528` | `mhc_fused_post_pre_tilelang` (large-batch branch) | same | same | `num_tokens > 16 and norm_weight is not None` |

On V4-Flash, both attention-pre and ffn-pre call sites in `vllm/models/deepseek_v4/nvidia/model.py:874, 888, 912` pass `norm_weight=attn_norm.weight / ffn_norm.weight` — so **this variant is the active path on V4-Flash for every decoder layer's hc_pre**.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `gemm_out_mul` | `[n_splits, num_tokens, gemm_last_dim]` (`gemm_last_dim` defaults to `hc_mult3=24`) | fp32 | row-major | Split-K partials of `residual_flat @ hc_fn.T`. |
| `gemm_out_sqrsum` | `[n_splits, num_tokens]` | fp32 | row-major | Split-K partials of `residual_flat.square().sum(dim=-1)`. |
| `hc_scale` | `[3]` | fp32 | flat | Per-block (pre/post/comb) scale factors. |
| `hc_base` | `[hc_mult3=24]` | fp32 | flat | Per-mix bias terms. |
| `residual` | `[num_tokens, hc_mult, hidden_size]` | bf16 | row-major contiguous | hc-stream residual (V4-Flash: `[T, 4, 4096]`). |
| `norm_weight` | `[hidden_size]` | bf16 | flat | RMSNorm γ (attn_norm or ffn_norm weight from `vllm/models/deepseek_v4/nvidia/model.py:809-810`). Must be bf16 + contiguous — wrapper coerces (`tilelang.py:155-158`). |
| `hidden_size` | scalar | int | — | 4096 on V4-Flash. |
| `rms_eps` | scalar | fp32 | — | RMSNorm ε for the hc-flatten denominator (V4-Flash: `config.rms_norm_eps ≈ 1e-6`). |
| `hc_pre_eps` | scalar | fp32 | — | Post-sigmoid floor on pre-mix (V4-Flash: `hc_eps=1e-6`). |
| `hc_sinkhorn_eps` | scalar | fp32 | — | Sinkhorn ε (V4-Flash: `hc_eps=1e-6`). |
| `hc_post_mult_value` | scalar | fp32 | — | Post-mix multiplier (V4-Flash: `hc_post_alpha=2.0`). |
| `sinkhorn_repeat` | scalar | int | — | Sinkhorn rounds (V4-Flash: `hc_sinkhorn_iters=20`). |
| `norm_eps` | scalar | fp32 | — | Independent ε used in the fused RMSNorm second pass (`attn_norm.variance_epsilon` / `ffn_norm.variance_epsilon` — typically same value as `rms_eps`). |
| `n_splits` | constexpr (kwarg) | int | — | Split-K factor. Default 16; in `mhc_fused_post_pre_tilelang` (line 525) `n_splits=1` is hard-coded for the large-batch path. |
| `hc_mult` | constexpr (kwarg) | int | — | 4 on V4-Flash. |
| `gemm_last_dim` | constexpr (kwarg) | int | — | Override for `hc_mult3` in case the producer packs extra columns; default `-1` → resolved to `hc_mult3`. |

**Constexpr/derived**: `hc_mult3 = hc_mult * (2 + hc_mult)` (24 for V4); `hidden_block = gcd(1024, hidden_size)` (1024 for hidden=4096 — *note: larger than the no-norm variant's 512*).

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `post_mix` | `[num_tokens, hc_mult]` | fp32 | row-major | Sigmoid-gated post coefficients (`× hc_post_mult_value`) for the next post-mapping. |
| `comb_mix` | `[num_tokens, hc_mult * hc_mult]` | fp32 | row-major (flattened from `[T, hc, hc]`) | Sinkhorn doubly-stochastic mix matrix. |
| `layer_input` | `[num_tokens, hidden_size]` | bf16 | row-major | **RMSNorm'd** pre-weighted-sum of the hc residual: γ-scaled and rsqrt-normalized using `norm_weight` and `norm_eps`. This is the input to the next attn/ffn block (the RMSNorm γ has been pre-applied; downstream calls `self.attn(positions, x, ...)` directly without re-norming, see `model.py:907-908`). |

## Grid / Block

- `grid_dim = (num_tokens,)` — one CTA per token.
- `block_dim`: **96 threads** (3 warps), same warp split as sibling: warp 0 handles post/comb/Sinkhorn (`tilelang_kernels.py:254-294`), warps 1-2 handle pre-mix sigmoid + 2-pass fused RMSNorm (`tilelang_kernels.py:295-350`).
- Autotune configs: none. `TL_DISABLE_WARP_SPECIALIZED=True`, `TL_DISABLE_TMA_LOWER=True`, `TL_PTXAS_REGISTER_USAGE_LEVEL=10`.
- PDL: `T.pdl_sync()` at line 241 (after the rsqrt prelude is set up but before reading shared `mixes_shared`), `T.pdl_trigger()` at line 353. PDL hands off the next stage (the attn input projection that consumes `layer_input`).
- Shared memory: `mixes_shared [24] fp32` (96B), `pre_mix_shared [4] fp32` (16B), `output_shared [hidden_size=4096] bf16` (8 KiB — stashes pass-1 unnormalized output), `xs [hc_mult, hidden_block=1024] bf16` per stage × 3 (24 KiB), `w_shared [hidden_block] bf16` (2 KiB) in pass 2. Total ~35-40 KiB.
- Pipelining: pass 1 (line 312) uses `T.Pipelined(hidden_size // hidden_block, num_stages=3)` — 4 stages on H=4096 with 3-deep pipeline. Pass 2 (line 336) uses `num_stages=2`.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:674-682` (`Block.hc_pre`) + `model.py:692, 698` (`attn_norm`/`ffn_norm` applied after `hc_pre`). The vLLM kernel fuses the two — reference computes `y = sum(pre * residual)` then a separate `attn_norm(y)`; this kernel computes both in one pass over the hidden axis.

```python
# Inputs: same as #1, plus norm_weight: [H] bf16, norm_eps: float.
# V4-Flash: H=4096, hc_mult=4, hc_mult3=24, hc_post_mult_value=2.0,
#           hc_pre_eps = hc_sinkhorn_eps = 1e-6, sinkhorn_repeat=20.

# 1. Reduce sqrsum and mixes, apply RMS rsqrt to mixes (same as #1).
sqrsum = gemm_out_sqrsum[:, t].sum()
rsqrt  = torch.rsqrt(sqrsum / (hc_mult * H) + rms_eps)
mixes  = gemm_out_mul[:, t, :].sum(dim=0) * rsqrt            # [24]

# 2. post_mix and comb_mix — identical to #1.
post_mix[t] = torch.sigmoid(mixes[4:8] * hc_scale[1] + hc_base[4:8]) * hc_post_mult_value

cm = mixes[8:24].view(4, 4) * hc_scale[2] + hc_base[8:24].view(4, 4)
cm = torch.softmax(cm, dim=-1) + hc_sinkhorn_eps
cm = cm / (cm.sum(dim=0, keepdim=True) + hc_sinkhorn_eps)
for _ in range(sinkhorn_repeat - 1):
    cm = cm / (cm.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
    cm = cm / (cm.sum(dim= 0, keepdim=True) + hc_sinkhorn_eps)
comb_mix[t] = cm.view(16)

# 3. pre_mix and fused weighted-sum + RMSNorm (the new bit vs #1):
pre_mix = torch.sigmoid(mixes[0:4] * hc_scale[0] + hc_base[0:4]) + hc_pre_eps   # [4]

# Pass 1: weighted sum of residual streams, write bf16 to shared output_shared
#         while accumulating squared sum for the SECOND RMSNorm (over y, not residual).
y = (pre_mix.view(4, 1) * residual[t].float()).sum(dim=0)    # [H], fp32
y_bf16 = y.to(torch.bfloat16)                                # bf16 round (output_shared stash)
sumsq_y = (y_bf16.float() ** 2).sum()                        # second-pass denominator

# Pass 2: apply post-norm with norm_weight γ.
rsqrt_norm = torch.rsqrt(sumsq_y / H + norm_eps)
layer_input[t] = (y_bf16.float() * rsqrt_norm * norm_weight.float()).to(torch.bfloat16)
```

Notes on fusion:

- **Two RMSNorm denominators** are present: (a) the hc-pre `rsqrt` over `sqrsum / (hc_mult * H)` (used inside `mixes`), (b) the post-weighted-sum `rsqrt_norm` over `sumsq_y / H + norm_eps` (used to apply γ). (a) is computed from upstream `gemm_out_sqrsum` partials; (b) is computed in-kernel during pass 1.
- The kernel does *not* use the upstream `gemm_out_sqrsum` for pass-2 denominator — it must recompute because `y = sum_hc(pre_mix * residual)` has different norm than `residual` flattened. The `y_bf16` round-trip (line 328) deliberately matches the bf16 precision that the unfused reference's `attn_norm(y.to(bf16))` would see, preserving numerical equivalence to the reference graph.
- Stashing `output_shared` (8 KiB at H=4096, bf16) frees registers for the larger `hidden_block=1024` vs the no-norm variant's 512. The 3-stage pipeline + larger block balances the higher SMEM footprint.
- **Pipelines into**: `self.attn(positions, layer_input, ...)` directly — `attn_norm`/`ffn_norm` modules in the V4 model are present but their forward is no-op'd since the norm is fused here (see `model.py:907` comment). PDL trigger at exit overlaps with the next attn input-projection (e.g. `fused_wqa_wkv`).
- Inputs come from: `hc_prenorm_gemm_*` (#3 / #4) for `gemm_out_mul` / `gemm_out_sqrsum`, with PDL handshake; `residual` directly from prior `mhc_post_tilelang` (#6) or `mhc_fused_tilelang` (#5).

## Config-dependent dispatch

- Activation condition: V4-Flash takes this path for **every** `hc_pre` call (3 per decoder layer: attn-pre on layer 0, attn-pre on layers >0 via `mhc_fused_post_pre_tilelang`, ffn-pre on every layer).
- Sibling locked-alternative: `mhc_pre_big_fuse_tilelang` (#1) for `norm_weight=None`. Class A locked since V4 always passes `norm_weight`.
- Producer GEMM selection (`tilelang.py:194-210`): `tf32_hc_prenorm_gemm` (DeepGEMM SM90/100) preferred over TileLang `_tilelang_hc_prenorm_gemm` when `is_deep_gemm_supported()`. SM100/B200 V4-Flash → DeepGEMM path.
- `n_splits` dynamic dispatch:
  - `mhc_pre_tilelang`: `n_splits = compute_num_split(block_k=64, k=hc_hidden_size, grid_size=cdiv(num_tokens, 64))` (`tilelang.py:172`).
  - `mhc_fused_post_pre_tilelang` large-batch branch: `n_splits=1` hard-coded (`tilelang.py:525`) — `compute_num_split` was deemed not helpful at that fusion's small-T regime.
- `gemm_last_dim`: defaulted to `hc_mult3` — both call sites in the wrapper pass nothing for this, but the kernel allows for a wider GEMM output (e.g. if the GEMM is padded for alignment).

Notes on fusion / pipeline:

- This is the canonical V4-Flash `hc_pre`-with-attn_norm fused kernel — the active path for every layer transition. The output `layer_input` feeds the next attention or FFN block directly (no separate norm call).
- Pairs with downstream `mhc_post_tilelang` (#6) / `mhc_fused_tilelang` (#5) which consumes `post_mix` + `comb_mix` to fold the post-layer activation back into the hc residual.
