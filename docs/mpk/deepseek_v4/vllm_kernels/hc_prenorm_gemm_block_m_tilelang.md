# hc_prenorm_gemm_block_m_tilelang

## Identity

- Source file: `vllm/model_executor/kernels/mhc/tilelang_kernels.py:623-713` (`hc_prenorm_gemm_block_m_tilelang`, TileLang `@tilelang.jit`).
- Launch wrapper: `vllm/model_executor/kernels/mhc/tilelang.py:21-87` (`_tilelang_hc_prenorm_gemm`); invoked at line 44 in the **large-batch override branch**.
- Language/DSL: **TileLang** (same pass_configs as siblings).
- Third-party dep: `tilelang`.
- Registered as opaque custom op: no.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/model_executor/kernels/mhc/tilelang.py:44` | `_tilelang_hc_prenorm_gemm` large-batch override (only path that selects this kernel) | `x: [T≥1024, hc_mult*H = 16384]`, `fn: [24, 16384]` | bf16 x, fp32 fn | `n_splits==1 and use_default_config (tile_n==12, n_thr==512) and T >= 1024` |

This is a **Class A locked-alternative** of `hc_prenorm_gemm_tilelang` (#3): dispatched purely on batch size. On V4-Flash B200, the canonical path is still DeepGEMM `tf32_hc_prenorm_gemm` — this kernel is the TileLang fallback variant taken only when DeepGEMM is unavailable AND `T ≥ 1024`.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `x` | `[num_tokens, hc_hidden_size = hc_mult * hidden_size = 16384]` | bf16 | row-major contiguous | Flattened hc residual (`residual.view(T, hc_mult*H)`). |
| `fn` | `[n_out = hc_mult3 = 24, hc_hidden_size]` | fp32 | row-major | hc projection weights. |
| `hidden_size` | constexpr | int | — | 4096 on V4-Flash. |
| `hc_mult` | constexpr (kwarg) | int | — | 4 on V4-Flash. |
| `n_out` | constexpr (kwarg) | int | — | 24 (passed by wrapper as `fn.shape[0]`). |
| `n_thr` | constexpr (kwarg) | int | — | 512 (wrapper hard-codes at line 52). |
| `tile_n` | constexpr (kwarg) | int | — | 12 (wrapper hard-codes at line 53). |
| `block_m` | constexpr (kwarg) | int | — | 2 (wrapper hard-codes at line 54). **The M-tiling factor that distinguishes this variant from #3.** |

**Constexpr/derived**: `hc_hidden_size = 16384`; `k_iters = hc_hidden_size / n_thr` = 32; `n_tiles = ceildiv(n_out, tile_n)` = 2; `m_tiles = ceildiv(num_tokens, block_m)` = T/2 when T even.

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `out` (`gemm_out_mul`) | `[1, num_tokens, n_out=24]` | fp32 | row-major | Final (no split-K dim) `x @ fn.T`. Leading `1` is the degenerate `n_splits=1`. |
| `sqrsum` (`gemm_out_sqrsum`) | `[1, num_tokens]` | fp32 | row-major | `x.float().square().sum(dim=-1)`. |

**Output shape difference vs #3**: this variant always emits `n_splits=1` (the wrapper requires `n_splits==1` to take this branch). Otherwise the downstream consumer sees the same `[1, T, 24]` contract as the `n_splits=1` case of `hc_prenorm_gemm_tilelang`.

## Grid / Block

- `grid_dim = (m_tiles, n_tiles)` — `(ceildiv(T, block_m=2), ceildiv(24, 12) = 2)`. Note: **2D grid** (no split-K dim), vs #3's 3D grid `(T, n_tiles, n_splits)`.
- `block_dim`: `n_thr = 512` (16 warps). Each CTA processes `block_m = 2` tokens × `tile_n = 12` output mixes.
- Autotune configs: none; chosen by outer heuristic only.
- PDL: `T.pdl_sync()` at line 654, `T.pdl_trigger()` at line 711.
- Shared memory: `warp_acc [num_warps=16, block_m=2, tile_n=12] fp32` (1536B), `warp_sqr [16, 2] fp32` (128B).
- Per-CTA work: each thread loads `k_iters = 32` K-axis elements of `fn` into a local fragment `fn_val [tile_n=12]`, then iterates over the `block_m = 2` tokens reusing that `fn_val` — **this is the key optimization vs #3**. With `T ≥ 1024`, M-blocking amortizes the `fn` load over 2 tokens, halving TMA traffic on `fn` (the wrapper comment "reduces TMA communication").
- The kernel uses register-allocated `fn_val` rather than shared memory for `fn` (`tilelang_kernels.py:658-664`), trading SMEM for register pressure to keep the 2-deep M loop tight.

## Math

Reference: identical to #3 — `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:677-679`. Both kernels emit the un-reduced `F.linear` and `x.square().sum(-1)` partials; the difference is in the partitioning strategy.

```python
# Inputs:
#   x:  [T, K=16384] bf16
#   fn: [n_out=24, K] fp32
#
# Outputs (n_splits=1 implicit):
#   out[0, t, j]  = sum over K of  x[t, k].float() * fn[j, k]
#   sqrsum[0, t]  = sum over K of  x[t, k].float() ** 2

# Pseudo per-CTA (i_mt=M tile, i_t=output tile, threads=n_thr=512):
acc = torch.zeros(block_m=2, tile_n=12)
sqr = torch.zeros(block_m=2)
for it in range(k_iters):                          # k_iters = K / n_thr = 32
    i_k = it * n_thr + tid                         # tid 0..511
    # Load tile_n=12 fn rows for this K element (per-thread).
    fn_val = [fn[i_t*tile_n + i_o, i_k] if (i_t*tile_n+i_o)<n_out else 0.0
              for i_o in range(tile_n)]
    # Reuse fn_val across block_m=2 tokens.
    for i_m in range(block_m):
        t = i_mt * block_m + i_m
        if t < num_tokens:
            x_val = x[t, i_k]
            for i_o in range(tile_n):
                acc[i_m, i_o] += x_val * fn_val[i_o]
            if i_t == 0:
                sqr[i_m] += x_val * x_val

# Warp reduce + cross-warp shared reduce per (i_m, i_o), write final.
for i_m in range(block_m):
    t = i_mt * block_m + i_m
    if t < num_tokens:
        out[0, t, i_t*tile_n + lane] = sum_w warp_acc[w, i_m, lane]    # lane 0..11
        if i_t == 0:
            sqrsum[0, t]            = sum_w warp_sqr[w, i_m]
```

Notes on fusion:

- **M-blocking is the only material difference from #3**. By loading `fn_val[tile_n]` once per K iteration and reusing across `block_m=2` tokens, the kernel halves the `fn` memory traffic relative to #3, which loads `fn` independently for every token (since #3's grid is `(T, n_tiles, n_splits)` — one CTA per token).
- The trade-off is that `fn_val` lives in registers (`T.alloc_local`), increasing register pressure. The 16-warp CTA absorbs this thanks to the `TL_PTXAS_REGISTER_USAGE_LEVEL=10` config.
- Sqrsum bookkeeping: only `i_t == 0` writes `sqrsum`, same as #3.
- Token-edge handling: `if token_idx < num_tokens` (lines 667, 697) guards the tail M-tile when `T` is odd.
- **Pipelines into**: same consumers as #3 — `mhc_pre_big_fuse_*` reduces over the (degenerate `n_splits=1`) axis. PDL handshake at exit.

## Config-dependent dispatch

- Activation condition on V4-Flash: only when DeepGEMM is unavailable AND `T ≥ 1024` AND `n_splits == 1` AND wrapper config uses `tile_n=12, n_thr=512` defaults. The combined gate at `tilelang.py:43`:
  ```python
  if n_splits == 1 and use_default_config and x.shape[0] >= 1024:
      hc_prenorm_gemm_block_m_tilelang(...)
  ```
- Class A locked-alternative of #3, selected by batch-size threshold `T ≥ 1024`. **Heuristic rationale**: at large batch, the `fn` matrix (24 × 16384 × 4B = 1.5 MiB) becomes the bottleneck across many independent token CTAs; M-blocking re-uses each `fn` load across 2 tokens, cutting effective traffic in half. At small batch (`T < 1024`), there are too few CTAs to make this worthwhile, and #3's per-token CTA wins via better SM occupancy.
- **DECISION REQUIRED** (Class B-like nature of the heuristic): on V4-Flash, B200 prefers DeepGEMM `tf32_hc_prenorm_gemm` so this path is rarely taken. If the MPK implementation needs to mirror the TileLang fallback, it should implement both #3 and #4 with the same `T >= 1024` heuristic, *or* implement only one (#3 alone covers all batch sizes correctly, just less efficiently at large T).
- No SM90/SM100 branch inside this kernel — same as #3, the SM-flavored dispatch is at the DeepGEMM-vs-TileLang boundary one layer up.
- No CuteDSL alternative.

Notes on fusion / pipeline:

- Output contract is identical to #3 at `n_splits=1`, so downstream `mhc_pre_big_fuse_*` does not branch — `gemm_out_mul[0]` and `gemm_out_sqrsum[0]` are consumed transparently. The leading `1` in the output shape preserves the `[n_splits, T, n_out]` shape convention; the downstream reduce loop `for i_split in T.serial(n_splits)` simply runs once.
- M-blocking is a **memory-traffic optimization on the `fn` weights**. It does not change numerical results — accumulation is in fp32 with the same order of operations modulo CTA assignment.
- This kernel feeds the same downstream chain as #3: `mhc_pre_big_fuse_with_norm_tilelang` (#2) on V4-Flash.
