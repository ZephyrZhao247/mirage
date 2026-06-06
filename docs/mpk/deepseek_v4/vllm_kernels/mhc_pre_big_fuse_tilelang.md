# mhc_pre_big_fuse_tilelang

## Identity

- Source file: `vllm/model_executor/kernels/mhc/tilelang_kernels.py:55-189` (`mhc_pre_big_fuse_tilelang`, TileLang `@tilelang.jit`).
- Launch wrapper: `vllm/model_executor/kernels/mhc/tilelang.py:213-230` inside `mhc_pre_tilelang` (the no-`norm_weight` branch); also re-dispatched from `vllm/model_executor/kernels/mhc/tilelang.py:509-526` inside `mhc_fused_post_pre_tilelang` when `num_tokens > 16` and `norm_weight is None`.
- Language/DSL: **TileLang** (`@tilelang.jit(pass_configs={TL_DISABLE_WARP_SPECIALIZED: True, TL_DISABLE_TMA_LOWER: True, TL_PTXAS_REGISTER_USAGE_LEVEL: 10})`).
- Third-party dep: `tilelang` (installed via `pip install tilelang`). TileLang lowers this Python DSL to CUDA/PTX; the generated kernel is not directly visible in the repo.
- Registered as opaque custom op: no, but its caller `mhc_pre_tilelang` is registered as `torch.ops.vllm.mhc_pre_tilelang` (`vllm/model_executor/kernels/mhc/tilelang.py:658-663`).

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/model_executor/kernels/mhc/tilelang.py:213` | `mhc_pre_tilelang` (Class A wrapper) | `residual: [T, hc_mult=4, hidden_size=4096]`; `gemm_out_mul: [n_splits, T, hc_mult3=24]`; `gemm_out_sqrsum: [n_splits, T]` | bf16 residual, fp32 gemm-outs | `norm_weight is None` |
| `vllm/model_executor/kernels/mhc/tilelang.py:509` | `mhc_fused_post_pre_tilelang` (large-batch branch) | same | same | `num_tokens > 16 and norm_weight is None` |

In V4-Flash on B200, both V4 model call sites in `vllm/models/deepseek_v4/nvidia/model.py:874, 888, 912` pass `norm_weight=attn_norm_weight/ffn_norm_weight` (non-None), so this no-norm variant is **NOT** taken on V4-Flash itself — the with-norm sibling `mhc_pre_big_fuse_with_norm_tilelang` is the active path. This kernel is in scope because it is the reference fusion structure for the with-norm variant and remains a Class A locked-alternative reachable when a config sets `attn_norm` / `ffn_norm` weight to None (e.g., for non-V4 mHC models on the same kernel family).

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `gemm_out_mul` | `[n_splits, num_tokens, hc_mult3]` where `hc_mult3 = hc_mult * (2 + hc_mult) = 24` for `hc_mult=4` | fp32 | row-major contiguous | Split-K partial sums of `residual_flat @ hc_fn.T` from `hc_prenorm_gemm_*_tilelang` (or DeepGEMM `tf32_hc_prenorm_gemm`). First `hc_mult` slots are pre-mix logits, next `hc_mult` are post-mix logits, last `hc_mult*hc_mult` are comb-mix logits. |
| `gemm_out_sqrsum` | `[n_splits, num_tokens]` | fp32 | row-major | Split-K partial sums of `residual_flat.square().sum(dim=-1)` from the same producer kernel. |
| `hc_scale` | `[3]` | fp32 | flat | Per-block (pre/post/comb) scale factors (`hc_attn_scale` or `hc_ffn_scale` from `vllm/models/deepseek_v4/nvidia/model.py:845-858`). |
| `hc_base` | `[hc_mult3]` | fp32 | flat | Per-mix bias terms (`hc_attn_base` or `hc_ffn_base`). |
| `residual` | `[num_tokens, hc_mult, hidden_size]` | bf16 | row-major contiguous on hidden_size | The hc_mult-stream residual state (V4 maintains `hc_mult=4` parallel residual copies). |
| `hidden_size` | scalar | int | — | Token feature dim. V4-Flash: 4096. |
| `rms_eps` | scalar | fp32 | — | RMSNorm epsilon (V4-Flash: `config.rms_norm_eps`, typically 1e-6). |
| `hc_pre_eps` | scalar | fp32 | — | Numerical floor added after sigmoid on pre-mix (V4-Flash: `hc_eps=1e-6`). |
| `hc_sinkhorn_eps` | scalar | fp32 | — | Sinkhorn iteration epsilon — same `hc_eps=1e-6` value reused. |
| `hc_post_mult_value` | scalar | fp32 | — | Post-mix multiplier (V4-Flash: `hc_post_alpha=2.0`, `vllm/models/deepseek_v4/nvidia/model.py:814`). |
| `sinkhorn_repeat` | scalar | int | — | Number of full Sinkhorn iteration rounds (V4-Flash: `hc_sinkhorn_iters=20`). |
| `n_splits` | constexpr (kwarg) | int | — | Split-K factor from the producer GEMM. Default 16; effective value is set by `compute_num_split(...)` at wrapper line 172 (`mhc_pre_tilelang`) — typically `n_SMs // grid_size` capped by `num_block_k // 4`. |
| `hc_mult` | constexpr (kwarg) | int | — | Number of HC residual streams. V4-Flash: 4. |

**Constexpr/derived**: `hc_mult3 = hc_mult * (2 + hc_mult)` (24 for V4); `hidden_block = gcd(512, hidden_size)` (512 for hidden=4096).

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `post_mix` | `[num_tokens, hc_mult]` | fp32 | row-major | Sigmoid-gated post coefficients used later by `mhc_post_tilelang` / `mhc_fused_tilelang` to expand a single attn/ffn output into hc_mult streams. |
| `comb_mix` | `[num_tokens, hc_mult * hc_mult]` (flattened `[T, hc, hc]`) | fp32 | row-major | Sinkhorn-doubly-stochastic combination matrix per token. Used by the next-stage post mapping to mix hc-stream residuals. |
| `layer_input` | `[num_tokens, hidden_size]` | bf16 | row-major | Pre-norm weighted sum of `residual[t, :, :]` with sigmoid-gated `pre_mix` weights — i.e. the collapsed single-stream input ready for the next attn/ffn block. Unweighted (no RMSNorm gamma in this variant). |

## Grid / Block

- `grid_dim = (num_tokens,)` — one CTA per token (`with T.Kernel(num_tokens, threads=96) as i`).
- `block_dim` (threads/CTA): **96 threads** (= 3 warps). The kernel splits work between warp 0 (`tid < 32`) and warps 1-2 (`tid >= 32`):
  - Warp 0 (32 threads): post_mix + comb_mix computation + Sinkhorn iterations (`tilelang_kernels.py:108-157`).
  - Warps 1-2 (64 threads): pre_mix sigmoid + pipelined hidden-axis weighted-sum write to `layer_input` (`tilelang_kernels.py:158-185`).
- Autotune configs: none — `threads=96`, `pass_configs` fixed. `TL_DISABLE_WARP_SPECIALIZED=True` and `TL_DISABLE_TMA_LOWER=True` mean no warp-specialization or TMA on these shapes; `TL_PTXAS_REGISTER_USAGE_LEVEL=10` raises register budget for the tight HC code.
- PDL: `T.pdl_sync()` at entry, `T.pdl_trigger()` at exit — Programmatic Dependent Launch is enabled iff `ENABLE_PDL` is True (`tilelang_kernels.py:27`, i.e. SM≥90 + CUDA). On SM100/B200 this kernel issues a PDL handshake with the upstream `hc_prenorm_gemm_*` GEMM so the GEMM tail can overlap with this kernel's prologue.
- Shared memory: `mixes_shared [hc_mult3] fp32` (96B), `pre_mix_shared [hc_mult] fp32` (16B), `xs [hc_mult, hidden_block=512] fp32` ×2 staged pipeline (16 KiB ×2 = 32 KiB).
- Pipelining: hidden axis loop runs `T.Pipelined(hidden_size // hidden_block, num_stages=2)` (line 171) on warps 1-2.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:674-682` (`Block.hc_pre`). Sinkhorn details in `kernel.py:hc_split_sinkhorn_kernel` and the per-token `mixes = F.linear(residual.flatten(2).float(), hc_fn) * rsqrt` decomposition. RMSNorm denominator uses `hc_mult * hidden_size` because the residual is flattened across hc streams before squared-sum.

This kernel fuses the *tail* of `hc_pre`: it consumes the already-computed (split-K-partial) `mixes` and `sqrsum` and produces `(pre, post, comb)` plus the weighted-sum `layer_input`. The GEMM and squared-sum themselves live in #3/#4 (`hc_prenorm_gemm_*`).

```python
# Inputs (one token t):
#   gemm_out_mul:    [S, T, 24] fp32   (S = n_splits; 24 = hc_mult3 = 4*(2+4))
#   gemm_out_sqrsum: [S, T]     fp32
#   residual:        [T, 4, H]  bf16   (H = hidden_size = 4096)
#   hc_scale:        [3]        fp32
#   hc_base:         [24]       fp32
#
# Constants (V4-Flash): hc_mult=4, hc_mult3=24, hc_post_mult_value=2.0,
#   hc_pre_eps = hc_sinkhorn_eps = 1e-6, sinkhorn_repeat=20.

# 1. Split-K reduce sqrsum and mixes, then apply rsqrt to mixes (RMSNorm
#    denominator over hc_mult*hidden features).
sqrsum = gemm_out_sqrsum[:, t].sum()                          # scalar
rsqrt  = torch.rsqrt(sqrsum / (hc_mult * H) + rms_eps)        # scalar
mixes  = gemm_out_mul[:, t, :].sum(dim=0) * rsqrt             # [24]

# 2. Partition the 24 mixes into pre / post / comb logits:
pre_logits  = mixes[0 : 4]                                    # [hc_mult]
post_logits = mixes[4 : 8]                                    # [hc_mult]
comb_logits = mixes[8 : 24].view(4, 4)                        # [hc_mult, hc_mult]

# 3. post_mix = sigmoid(post_logits * scale[1] + base[4:8]) * 2.0
post_mix[t] = torch.sigmoid(post_logits * hc_scale[1] + hc_base[4:8]) * hc_post_mult_value

# 4. comb_mix = Sinkhorn(comb_logits * scale[2] + base[8:24]) — sinkhorn_repeat
#    full iterations of (row-normalize, col-normalize) starting from one
#    row-softmax priming step (so total normalizations are 2*sinkhorn_repeat).
cm = comb_logits * hc_scale[2] + hc_base[8:24].view(4, 4)
# priming: row-softmax + eps, then col-normalize
cm = torch.softmax(cm, dim=-1) + hc_sinkhorn_eps              # [4, 4]
cm = cm / (cm.sum(dim=0, keepdim=True) + hc_sinkhorn_eps)
for _ in range(sinkhorn_repeat - 1):
    cm = cm / (cm.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
    cm = cm / (cm.sum(dim= 0, keepdim=True) + hc_sinkhorn_eps)
comb_mix[t] = cm.view(16)

# 5. pre_mix (sigmoid + eps) and pre-norm weighted sum into layer_input.
#    NOTE: the RMSNorm scale (rsqrt) is NOT applied here — only the sigmoid-
#    gated weights mix the hc streams. RMSNorm γ is absent (no norm_weight).
pre_mix = torch.sigmoid(pre_logits * hc_scale[0] + hc_base[0:4]) + hc_pre_eps   # [4]
layer_input[t] = (pre_mix.view(4, 1) * residual[t].float()).sum(dim=0).to(torch.bfloat16)
```

Notes on fusion:

- The producer `hc_prenorm_gemm_*` already computed the unreduced split-K partials. This kernel completes the reduction *and* applies rsqrt-RMSNorm in one pass over the 24-wide mixes vector — avoiding a separate `sum/rsqrt/*` traversal of `mixes`.
- Sinkhorn primer step (lines 130-142): first iteration is `row_softmax(cm) + eps`, then col-normalize. The remaining `sinkhorn_repeat - 1` iterations are plain `(row-normalize, col-normalize)`. This matches `model.py:680` via `hc_split_sinkhorn`.
- Warp 0 handles the small (hc_mult × hc_mult) Sinkhorn matrix locally in registers (`cm` fragment); warps 1-2 stream the hidden axis in tiles of `hidden_block=512` (4096/512 = 8 pipelined stages × 2 stages double-buffer).
- The pre_mix sigmoid uses `mixes[0:4]` after the global rsqrt — that's why `mixes_shared` is staged in shared memory: both warp groups read it.
- **Pipelines into**: `attn_norm(layer_input)` → `attn`/`ffn` → `mhc_post_tilelang` / `mhc_fused_post_pre_tilelang` which consumes `post_mix` + `comb_mix` to recombine the hc-stream residual. PDL handshake at exit lets the next stage (attention input projection) start prologue overlap.

## Config-dependent dispatch

- Activation condition on V4-Flash: never directly on V4 (V4 always passes `norm_weight`). Active when `mhc_pre_tilelang(norm_weight=None)` is called by a different mHC model — Class A locked-alternative documented here for completeness.
- Producer dispatch (`vllm/model_executor/kernels/mhc/tilelang.py:194-210`): the upstream GEMM is `tf32_hc_prenorm_gemm` (DeepGEMM) when `is_deep_gemm_supported()` is True (SM90/100), else `_tilelang_hc_prenorm_gemm` (TileLang #3/#4). `n_splits` is dynamically set by `compute_num_split(block_k=64, k=hc_hidden_size, grid_size=cdiv(num_tokens, 64))` for DeepGEMM, else fixed to 1.
- No SM90 vs SM100 branch inside this kernel — TileLang generates one variant per `(hidden_size, n_splits, hc_mult)` tuple via JIT cache. SM100/B200 path differs only via the PDL macro and the DeepGEMM-vs-TileLang upstream dispatch.
- No CuteDSL alternative.

Notes on fusion / pipeline:

- Output `layer_input` is the input to **`attn_norm` / `ffn_norm`** in non-V4 callers; since this variant has no fused norm, downstream `RMSNorm.forward` consumes `layer_input` directly (vs. with-norm sibling that fuses both).
- Output `post_mix` and `comb_mix` feed **`mhc_post_tilelang`** (#6) or the fused **`mhc_fused_tilelang`** (#5) at the next layer boundary, which produces a new hc_mult-stream residual.
- Output `gemm_out_*` from the previous `hc_prenorm_gemm_tilelang` (#3) or `hc_prenorm_gemm_block_m_tilelang` (#4) is the input — those kernels' PDL trigger overlaps with this kernel's prologue.
