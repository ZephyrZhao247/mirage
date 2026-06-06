# mhc_fused_tilelang

## Identity

- Source file: `vllm/model_executor/kernels/mhc/tilelang_kernels.py:359-477` (`mhc_fused_tilelang`, TileLang `@tilelang.jit`).
- Launch wrapper: `vllm/model_executor/kernels/mhc/tilelang.py:461-475` inside `mhc_fused_post_pre_tilelang`, the small-token branch (`use_small_fma = num_tokens <= 16`).
- Language/DSL: **TileLang** (`@tilelang.jit(pass_configs={TL_DISABLE_WARP_SPECIALIZED: True, TL_DISABLE_TMA_LOWER: True, TL_PTXAS_REGISTER_USAGE_LEVEL: 10})`).
- Third-party dep: `tilelang`.
- Registered as opaque custom op: parent wrapper `mhc_fused_post_pre_tilelang` registered as `torch.ops.vllm.mhc_fused_post_pre_tilelang` (`tilelang.py:671-676`).

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/model_executor/kernels/mhc/tilelang.py:461` | `mhc_fused_post_pre_tilelang` (small-token fused path) | `x: [T≤16, 4096]`, `residual_in: [T, 4, 4096]`, `post_mix: [T, 4]`, `comb_mix: [T, 4, 4]`, `weight_t: [24, 4, 4096]` | bf16 x/residual, fp32 mixes/weight | `num_tokens <= 16` (V4-Flash decode regime) |

Upstream call chain on V4-Flash: `vllm/models/deepseek_v4/nvidia/model.py:888, 912` → `mhc_fused_post_pre_tilelang` → this kernel when `T ≤ 16`. **This is the V4-Flash decode-regime hot path** since typical decode batches have `T = batch_size * 1 token ≤ 16`.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `comb_mix` | `[num_tokens, hc, hc]` | fp32 | row-major (= `comb_res_mix.view(T, hc, hc)`) | Sinkhorn doubly-stochastic mix matrix from the previous `mhc_pre_big_fuse_with_norm_tilelang` (#2). |
| `residual_in` | `[num_tokens, hc, hidden]` | bf16 | row-major | Prior hc-stream residual (output of last layer's `mhc_post`). |
| `post_mix` | `[num_tokens, hc]` | fp32 | row-major (= `post_layer_mix_flat`) | Sigmoid-gated post coefficients from previous `mhc_pre`. |
| `x_in` | `[num_tokens, hidden]` | bf16 | row-major | The current layer's pre-`hc_post` activation (`attn(...)` or `ffn(...)` output). |
| `weight_t` | `[n_out, hc, hidden]` | fp32 | row-major (= `fn.view(hc_mult3, hc_mult, hidden_size)`, see `tilelang.py:466`) | hc projection weights, reshaped for direct (hc, hidden) striding. `n_out = hc_mult3 = 24` on V4-Flash. |
| `hc` | constexpr (kwarg) | int | — | `hc_mult` = 4 on V4-Flash. |
| `hidden` | constexpr (kwarg) | int | — | 4096 on V4-Flash. |
| `n_out` | constexpr (kwarg) | int | — | `hc_mult3 = 24` (passed by wrapper at line 472). |
| `n_thr` | constexpr (kwarg) | int | — | 256 (kernel default; not overridden by wrapper). |
| `h_blk` | constexpr (kwarg) | int | — | 256 (default; unused — `h_blk = gcd(hidden, 256)` = 256, but actual per-thread work uses `h_iters`). |
| `tile_n` | constexpr (kwarg) | int | — | Output mix tile size. Set by wrapper (`tilelang.py:414`) to `2` if `T < 8` else `3`. |
| `split_k` | constexpr (kwarg) | int | — | K-axis split. Set by wrapper (`tilelang.py:415`) to `8` if `T < 8 and hidden ≤ 4096`, else `4`. |

**Constexpr/derived**: `h_per_split = hidden / split_k`; `n_tiles = n_out / tile_n` (24/2=12 or 24/3=8); `h_iters = h_per_split / n_thr` (e.g. 4096/8/256 = 2 when split_k=8); `num_warps = n_thr / 32 = 8`.

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `yp_out` (`gemm_out_mul`) | `[split_k, num_tokens, n_out=24]` | fp32 | row-major | Split-K partial sums of `new_residual @ fn.T` — directly consumes the kernel's freshly-computed new residual without writing it to HBM first. |
| `rp_out` (`gemm_out_sqrsum`) | `[split_k, num_tokens]` | fp32 | row-major | Split-K partial sums of `new_residual.square().sum(dim=-1)`. Only `i_nt == 0` tile writes. |
| `residual_out` (`residual_cur`) | `[num_tokens, hc, hidden]` | bf16 | row-major | The newly-computed hc-stream residual: `new_r[hc] = post_mix[hc] * x_in + comb_mix[:, hc] @ residual_in[:, :]`. Only `i_nt == 0` writes (see Math). |

## Grid / Block

- `grid_dim = (num_tokens, n_tiles, split_k)` — one CTA per (token, output-mix tile, K-split).
- `block_dim`: `n_thr = 256` (8 warps).
- Autotune configs: indirect via wrapper heuristics (`tile_n ∈ {2, 3}`, `split_k ∈ {4, 8}`) — these flow as constexpr template args, so TileLang JIT-caches per shape.
- PDL: `T.pdl_sync()` at line 416, `T.pdl_trigger()` at line 476.
- Shared memory: `s_warp [num_warps=8, tile_n+1] fp32` (8×3+8=32 entries = 128B), `s_post [hc=4] fp32` (16B), `s_comb [hc, hc=4×4] fp32` (64B).
- Per-CTA work: each thread owns `h_iters` elements of the K-split's `h_per_split` slice. For each owned `h_idx`:
  1. Compute `new_r[hc=4]` = `post_mix * x_in[h_idx]` + `comb_mix.T @ residual_in[:, h_idx]` (line 432-435).
  2. If `i_nt == 0`, write `residual_out[h_idx]` and accumulate `sqr` (line 438-441).
  3. FMA into `acc[tile_n]` against `weight_t[i_nt*tile_n + n, :, h_idx]` (line 444-446).
- Warp-reduce `acc` and `sqr` (line 448-451), then cross-warp shared-memory reduce in warp 0 (line 462-473).

## Math

Reference: combines two ops from `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`:
1. `hc_post` body at lines 684-687 (`y = post.unsqueeze(-1) * x.unsqueeze(-2) + sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)`).
2. The first part of `hc_pre` (lines 674-679) — specifically the `mixes = F.linear(y, hc_fn)` and `sqrsum` computation that #3 / #4 would normally do — fused into the same kernel by consuming `new_r` directly without round-tripping through HBM.

```python
# Inputs:
#   comb_mix:     [T, hc=4, hc=4] fp32     (Sinkhorn comb from previous mhc_pre)
#   residual_in:  [T, hc, H=4096]   bf16
#   post_mix:     [T, hc]           fp32
#   x_in:         [T, H]            bf16   (current attn/ffn output)
#   weight_t:     [n_out=24, hc, H] fp32   (= fn reshaped to [24,4,H])

# For each token t, each K-split i_ks, each output-mix tile i_nt:
for h_idx in [i_ks*h_per_split, (i_ks+1)*h_per_split):
    # 1. new residual stream values at hidden index h_idx:
    new_r = post_mix[t].float() * x_in[t, h_idx].float()                # [hc]
    new_r += comb_mix[t].T @ residual_in[t, :, h_idx].float()           # [hc]

    # 2. Write residual_out and accumulate sqrsum — only on tile 0 to dedup:
    if i_nt == 0:
        residual_out[t, :, h_idx] = new_r.to(torch.bfloat16)
        sqr += (new_r ** 2).sum()                                       # scalar per thread

    # 3. FMA into the K-axis accumulator for tile i_nt:
    for n in range(tile_n):
        out_idx = i_nt * tile_n + n
        # acc[n] += sum over hc of  weight_t[out_idx, :, h_idx] * new_r
        acc[n] += (weight_t[out_idx, :, h_idx] * new_r).sum()

# Warp-reduce + cross-warp reduce + warp 0 writes:
yp_out[i_ks, t, i_nt*tile_n + lane] = sum_w s_warp[w, lane]              # lane < tile_n
if i_nt == 0:
    rp_out[i_ks, t]               = sum_w s_warp[w, tile_n]
```

Notes on fusion:

- **Two-op fusion**: this kernel computes the new hc residual (`hc_post`) AND the linear projection + sqrsum needed by the NEXT layer's `hc_pre` (`mixes`) in a single pass over the hidden axis. The `new_r` value lives in registers across the FMA — never round-trips through HBM.
- **Why small-T only**: At small `T ≤ 16`, occupancy per CTA matters less than minimizing kernel launches. By bundling `hc_post + hc_prenorm_gemm`, the small-batch path saves a full kernel launch + HBM round-trip for `residual_cur`. At large T, splitting into separate `mhc_post_tilelang` (#6) + `tf32_hc_prenorm_gemm` (DeepGEMM) wins via better SM utilization.
- **Per-tile dedup of residual writes**: the same `new_r` is computed by every `i_nt` CTA (the grid is `(T, n_tiles, split_k)`), but only `i_nt == 0` writes to `residual_out` and accumulates `sqr` (lines 438-441). All `i_nt` CTAs do the FMA into their `acc[tile_n]` slice. This avoids races on `residual_out` and double-counting on `rp_out`.
- The FMA pattern `acc[n] += weight_t[..., h_idx] * new_r[j]` (line 444-446) is `n_thr` threads each accumulating along K (hidden) via a stride-`n_thr` access; cross-warp reduce sums the per-thread partials.
- **Pipelines into**: `mhc_pre_big_fuse_with_norm_tilelang` (#2) which reduces `yp_out` over the `split_k` dim and applies rsqrt + RMSNorm γ. PDL trigger on this kernel's exit overlaps with #2's prologue. Inputs come from prior layer's `mhc_pre_big_fuse_with_norm_tilelang` (`post_mix`, `comb_mix`, `residual_in` as `residual_cur` output) and from `self.attn(...) / self.ffn(...)` for `x_in`.

## Config-dependent dispatch

- Activation condition: `mhc_fused_post_pre_tilelang` selects this when `use_small_fma = num_tokens <= 16` (`tilelang.py:411-415`). On V4-Flash this is the **decode regime** — single-token batches with small concurrent request count.
- Sibling alternatives at `tilelang.py:476-506` (large-batch path):
  - `mhc_post_tilelang` (#6) for the `hc_post` step.
  - `tf32_hc_prenorm_gemm` (DeepGEMM SM100) or `_tilelang_hc_prenorm_gemm` (#3 / #4 TileLang) for the GEMM+sqrsum step.
  - Decision: this is a fusion *strategy* choice — small-T fuses both, large-T splits. **Class A** since the dispatch is locked by batch size, not user config.
- `tile_n` and `split_k` sub-heuristic (`tilelang.py:414-415`):
  - `T < 8 and hidden ≤ 4096` → `(tile_n=2, split_k=8)`.
  - `T < 8 and hidden > 4096` → `(tile_n=2, split_k=4)`.
  - `8 ≤ T ≤ 16` → `(tile_n=3, split_k=4)`.
  Wrapper comment notes `TODO(gnovack): investigate autotuning these heuristics`.
- No SM90/SM100 branch inside the kernel.

Notes on fusion / pipeline:

- This kernel sits at the **layer boundary** in V4-Flash decode: it consumes the previous layer's `post_mix`/`comb_mix` and the current layer's `attn`/`ffn` output, emits the new residual AND the GEMM partials needed by the next `mhc_pre` to apply rsqrt-RMSNorm.
- The 3-kernel layer chain in decode is: `[mhc_pre_big_fuse_with_norm | self.attn/ffn | mhc_fused_tilelang]` → next layer. In prefill (large T), it expands to: `[mhc_pre_big_fuse_with_norm | self.attn/ffn | mhc_post_tilelang | tf32_hc_prenorm_gemm | mhc_pre_big_fuse_with_norm]`.
