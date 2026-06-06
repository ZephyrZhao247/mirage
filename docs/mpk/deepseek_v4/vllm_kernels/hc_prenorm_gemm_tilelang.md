# hc_prenorm_gemm_tilelang

## Identity

- Source file: `vllm/model_executor/kernels/mhc/tilelang_kernels.py:537-618` (`hc_prenorm_gemm_tilelang`, TileLang `@tilelang.jit`).
- Launch wrapper: `vllm/model_executor/kernels/mhc/tilelang.py:21-87` (`_tilelang_hc_prenorm_gemm`); this kernel is invoked at lines 63 (the small-batch override config) and 76 (the default config). It is the **TileLang fallback** of `tf32_hc_prenorm_gemm` from DeepGEMM — used when `is_deep_gemm_supported()` is False (`tilelang.py:202, 488` upstream).
- Language/DSL: **TileLang** (`@tilelang.jit(pass_configs={TL_DISABLE_WARP_SPECIALIZED: True, TL_DISABLE_TMA_LOWER: True, TL_PTXAS_REGISTER_USAGE_LEVEL: 10})`).
- Third-party dep: `tilelang`.
- Registered as opaque custom op: no — invoked directly via the JITKernel returned by `tilelang.jit`.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/model_executor/kernels/mhc/tilelang.py:63` | `_tilelang_hc_prenorm_gemm` small-batch override (uses `n_thr=1024`, `tile_n=4`) | `x: [T<128, hc_mult*H = 16384]`, `fn: [24, 16384]` | bf16 x, fp32 fn | `n_splits==1 and use_default_config and T<128 and (hc_mult*H)%1024==0` |
| `vllm/model_executor/kernels/mhc/tilelang.py:76` | `_tilelang_hc_prenorm_gemm` default fallback | same | same | otherwise (covers most non-DeepGEMM SM-capability paths and split-K) |

Upstream of `_tilelang_hc_prenorm_gemm`:
- `mhc_pre_tilelang` at `tilelang.py:203`: only when `not is_deep_gemm_supported()`.
- `mhc_fused_post_pre_tilelang` large-batch branch at `tilelang.py:499`: same gate.

On V4-Flash on B200 (SM100) the canonical path is `tf32_hc_prenorm_gemm` (DeepGEMM) — this TileLang kernel is the **portable fallback** when DeepGEMM is unavailable (e.g. SM<90 or DeepGEMM not built). It is still in scope as Class A because it shares the exact output contract with DeepGEMM (`gemm_out_mul: [n_splits, T, n_out]`, `gemm_out_sqrsum: [n_splits, T]`) and downstream kernels do not see the difference.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `x` | `[num_tokens, hc_hidden_size]` where `hc_hidden_size = hc_mult * hidden_size` = 16384 (V4-Flash) | bf16 | row-major contiguous (= `residual.view(T, hc_mult*H)` from `tilelang.py:193`) | Flattened hc residual fed into the linear/sqrsum. |
| `fn` | `[n_out, hc_hidden_size]` where `n_out = hc_mult3 = hc_mult*(2+hc_mult)` = 24 | fp32 | row-major (the `hc_attn_fn` / `hc_ffn_fn` parameter from `model.py:817-830`) | hc projection weights (mix-axis × flattened hc residual). |
| `hidden_size` | constexpr | int | — | 4096 on V4-Flash. |
| `hc_mult` | constexpr (kwarg) | int | — | 4 on V4-Flash. |
| `n_out` | constexpr (kwarg) | int | — | Output mix dim; default 24, set to `fn.shape[0]` by the wrapper. |
| `n_thr` | constexpr (kwarg) | int | — | Threads per CTA. Wrapper passes 512 (default) or 1024 (small-batch override at `tilelang.py:71`). |
| `tile_n` | constexpr (kwarg) | int | — | Output mix tile per CTA. Wrapper passes 12 (default) or 4 (small-batch override at `tilelang.py:72`). |
| `n_splits` | constexpr (kwarg) | int | — | Split-K factor along the `hc_hidden_size` reduction axis. Wrapper passes from outer; in the no-DeepGEMM path it is always 1 (`tilelang.py:174`). |

**Constexpr/derived**: `hc_hidden_size = hc_mult * hidden_size` (16384); `k_per_split = hc_hidden_size / n_splits`; `k_iters = k_per_split / n_thr` (must divide evenly — wrapper asserts at line 41); `n_tiles = ceildiv(n_out, tile_n)` (= 2 with `n_out=24, tile_n=12`).

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `out` (`gemm_out_mul`) | `[n_splits, num_tokens, n_out=24]` | fp32 | row-major contiguous | Split-K partial sums of `x @ fn.T`. Reduced over splits inside the consumer (`mhc_pre_big_fuse_*`). |
| `sqrsum` (`gemm_out_sqrsum`) | `[n_splits, num_tokens]` | fp32 | row-major | Split-K partial sums of `x.float().square().sum(dim=-1)` — but **only the `i_t == 0` tile writes** (`tilelang_kernels.py:581-587`), so per-(split, token) it is written exactly once. |

## Grid / Block

- `grid_dim = (num_tokens, n_tiles, n_splits)` — one CTA per (token, output-mix tile, K-split).
- `block_dim`: `n_thr` (default 512, small-batch override 1024). `num_warps = n_thr / 32` = 16 (or 32 for the small-batch override).
- Autotune configs: implicit table at `tilelang.py:42-75`:
  - `n_splits==1 and T>=1024 and use_default_config` → diverted to `hc_prenorm_gemm_block_m_tilelang` (#4) instead of this kernel.
  - `n_splits==1 and T<128 and (hc_mult*H)%1024==0 and use_default_config` → `(n_thr=1024, tile_n=4)`.
  - Otherwise: `(n_thr=512, tile_n=12)` (the documented defaults).
- PDL: `T.pdl_sync()` at line 572, `T.pdl_trigger()` at line 617 — handshake with upstream (residual producer) and downstream (`mhc_pre_big_fuse_*`).
- Shared memory: `warp_acc [num_warps=16, tile_n=12] fp32` (768B), `warp_sqr [16] fp32` (64B) — cross-warp reduction scratch. Effectively no SMEM tile staging for `x` / `fn` because the kernel uses a thread-stride-1 K-axis stream where each thread directly loads `x[t, i_k]` and `fn[out_idx, i_k]` into registers.
- Per-CTA work: each thread loads `k_iters = k_per_split / n_thr` K-axis elements of x and fn (e.g. 16384 / 1 / 512 = 32 elements/thread). Across the warp, the `warp_reduce_sum` reduces the dot-product `acc` and `sqr` along K. Warp 0 then writes via shared-memory cross-warp reduction.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:677-679` — the `mixes = F.linear(x.flatten(2).float(), hc_fn) * rsqrt` and `rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)` lines. This kernel emits the *un-reduced split-K partials* of both `F.linear(x.float(), hc_fn)` (`out`) and `x.square().sum(-1)` (`sqrsum`); the rsqrt and multiply happen in the downstream `mhc_pre_big_fuse_*` consumer.

```python
# Inputs:
#   x:   [T, K] bf16     (K = hc_mult * H = 16384)
#   fn:  [n_out=24, K] fp32
#
# Output split-K partials (s = 0 .. n_splits-1):
#   out[s, t, j]    = sum over K_s of  x[t, k].float() * fn[j, k]
#   sqrsum[s, t]    = sum over K_s of  x[t, k].float() ** 2     (only when i_t == 0)
#
# where K_s = [s*k_per_split, (s+1)*k_per_split). The consumer then reduces
# over s to get the full sums.

# Pseudo per-CTA (i_n=token, i_t=output tile, i_s=K split, threads=n_thr=512):
for thread tid in range(n_thr):
    acc = torch.zeros(tile_n=12)            # 12 outputs per CTA
    sqr = 0.0
    for it in range(k_iters):
        i_k = i_s * k_per_split + it * n_thr + tid
        x_val = x[i_n, i_k]                  # bf16, implicitly upcast to fp32
        for i_o in range(tile_n):
            out_idx = i_t * tile_n + i_o
            if out_idx < n_out:
                acc[i_o] += x_val * fn[out_idx, i_k]
        if i_t == 0:
            sqr += x_val * x_val

    # Warp-reduce: warp_reduce_sum over each lane in acc[i_o] and sqr.
    # Cross-warp via shared memory + warp 0 emits final.
out[i_s, i_n, i_t*tile_n + lane] = sum_w warp_acc[w, lane]            # lanes 0..tile_n-1
if i_t == 0:
    sqrsum[i_s, i_n]            = sum_w warp_sqr[w]                    # lane 0 only
```

Notes on fusion:

- `sqrsum` is only written for the `i_t == 0` output tile. With `n_out=24, tile_n=12` there are 2 tiles per token; the second tile (`i_t==1`) skips the sqrsum reduction (line 581, 586, 598, 610). This avoids redundant work — each (split, token) pair sees one sqrsum write.
- The K-axis split (`i_s` grid dim) is the "split-K" optimization. With `n_splits` SMs sharing the K dim, the consumer kernel (`mhc_pre_big_fuse_*`) reduces over splits in its `for i_split in T.serial(n_splits)` loop. `n_splits=1` is the common case for this TileLang variant (DeepGEMM is the higher-`n_splits` path).
- All-thread strided-K access: thread `tid` of CTA `(i_n, i_t, i_s)` reads `x[i_n, i_s*k_per_split + it*n_thr + tid]` for each iteration `it` — a coalesced stride-1 access pattern along K (`tilelang_kernels.py:575`). `fn` is read row-major along K too (`fn[out_idx, i_k]`), coalesced within a warp.
- bf16→fp32: `x_val = x[i_n, i_k]` is bf16 in memory; the multiply with fp32 `fn` promotes to fp32 implicitly via TileLang type rules. Acc is fp32 throughout — no precision loss vs the reference's `.float()` cast.
- **Pipelines into**: `mhc_pre_big_fuse_tilelang` (#1) or `mhc_pre_big_fuse_with_norm_tilelang` (#2) which reduces over the `n_splits` axis and applies RMSNorm rsqrt. PDL trigger at exit overlaps with the consumer's prologue.

## Config-dependent dispatch

- Activation condition on V4-Flash B200: never on the happy path — `is_deep_gemm_supported()` returns True on SM100, so `tf32_hc_prenorm_gemm` (DeepGEMM, `vllm/utils/deep_gemm.py`) is taken instead at `tilelang.py:195, 491`. This TileLang kernel is the fallback when DeepGEMM is absent.
- Class A locked-alternatives selected by `_tilelang_hc_prenorm_gemm` heuristic (`tilelang.py:42-75`):
  - **#4 `hc_prenorm_gemm_block_m_tilelang`** when `n_splits==1, T≥1024, use_default_config`. Document there.
  - This kernel with `(n_thr=1024, tile_n=4)` when `n_splits==1, T<128, K%1024==0, use_default_config`.
  - This kernel with `(n_thr=512, tile_n=12)` otherwise (the documented default).
- No SM90/SM100 branch inside the kernel — TileLang generates one variant per `(n_thr, tile_n, n_splits, hc_mult, hidden_size)`. The outer DeepGEMM-vs-TileLang dispatch is the only SM-flavored decision.
- No CuteDSL alternative.

Notes on fusion / pipeline:

- Output `gemm_out_mul` (`[n_splits, T, 24]`) and `gemm_out_sqrsum` (`[n_splits, T]`) feed directly into `mhc_pre_big_fuse_with_norm_tilelang` (#2) on V4-Flash — the consumer kernel reduces the `n_splits` dim and applies `rsqrt(sumsq / (hc_mult*H) + rms_eps)`.
- The producer side (this kernel's `x` input) is `residual.view(T, hc_mult*H)` — the same hc-stream residual that downstream kernels read with the `[T, hc_mult, H]` view.
- PDL handshake on both ends — this kernel is part of a strict 3-stage chain: `[upstream residual writer | this GEMM | mhc_pre_big_fuse_*]`.
