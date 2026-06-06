# mhc_post_tilelang

## Identity

- Source file: `vllm/model_executor/kernels/mhc/tilelang_kernels.py:482-532` (`mhc_post_tilelang`, TileLang `@tilelang.jit`).
- Launch wrapper:
  - `vllm/model_executor/kernels/mhc/tilelang.py:303-323` (`mhc_post_tilelang` Python wrapper, also re-exported as `torch.ops.vllm.mhc_post_tilelang`).
  - Invoked at `vllm/model_executor/kernels/mhc/tilelang.py:477-485` inside `mhc_fused_post_pre_tilelang` (the large-batch branch, `num_tokens > 16`).
- Language/DSL: **TileLang** (`@tilelang.jit(pass_configs={TL_DISABLE_WARP_SPECIALIZED: True, TL_DISABLE_TMA_LOWER: True, TL_PTXAS_REGISTER_USAGE_LEVEL: 10})`).
- Third-party dep: `tilelang`.
- Registered as opaque custom op: parent wrapper registered as `torch.ops.vllm.mhc_post_tilelang` (`tilelang.py:664-669`).

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/model_executor/kernels/mhc/tilelang.py:477` | `mhc_fused_post_pre_tilelang` (large-batch branch) | `b: [T>16, 4, 4096] bf16`, `a: [T, 4, 4]` fp32, `c: [T, 4]` fp32, `d: [T, 4096]` bf16 | mixed | `num_tokens > 16` (V4 prefill regime) |
| `vllm/models/deepseek_v4/nvidia/model.py:1084` | `DeepseekV4Model.forward` final post-mapping after last decoder layer (before `hc_head_fused_kernel_tilelang`) | same | mixed | always — runs once per forward pass |

The final-layer call site (`model.py:1084`) is `hidden_states = mhc_post_tilelang(hidden_states, residual, post_mix, res_mix)` — the model's `__init__.py:21` import alias points to the **wrapper** function `mhc_post_tilelang` from `tilelang.py:303`, which in turn invokes this kernel.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `a` (`comb_mix`) | `[num_tokens, hc, hc]` | fp32 | row-major | Sinkhorn doubly-stochastic mix matrix from previous `mhc_pre_big_fuse_with_norm`. |
| `b` (`residual_in`) | `[num_tokens, hc, hidden]` | bf16 | row-major | Prior hc-stream residual. |
| `c` (`post_mix`) | `[num_tokens, hc]` | fp32 | row-major (wrapper passes `post_layer_mix.squeeze(-1)` so this is the post-mix vector per token) | Sigmoid-gated post coefficients from previous `mhc_pre`. |
| `d` (`x`) | `[num_tokens, hidden]` | bf16 | row-major | Current layer's attn/ffn output. |
| `hc` | constexpr (kwarg) | int | — | 4 on V4-Flash. |
| `hidden` | constexpr (kwarg) | int | — | 4096 on V4-Flash. |
| `n_thr` | constexpr (kwarg) | int | — | 128 (kernel default; wrapper passes nothing — uses default). |
| `h_blk` | constexpr (kwarg) | int | — | 1024 (kernel default). Actual block = `gcd(hidden, 1024)` = 1024 for V4-Flash. |

**Constexpr/derived**: `h_blk = gcd(hidden, 1024) = 1024`; loop iterates `ceildiv(hidden, h_blk) = 4` hidden chunks per token.

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `x` (`residual_out`, return value) | `[num_tokens, hc, hidden]` | bf16 | row-major | New hc-stream residual: `x[h] = c[h] * d + a[:, h].T @ b[:, :]` for each hc stream `h`. |

## Grid / Block

- `grid_dim = (num_tokens,)` — one CTA per token (`with T.Kernel(n, threads=n_thr) as i_n`).
- `block_dim`: `n_thr = 128` (4 warps).
- Autotune configs: none.
- PDL: `T.pdl_sync()` at line 514, `T.pdl_trigger()` at line 531.
- Shared memory: `b_shared [hc=4, h_blk=1024] bf16` (8 KiB), `d_shared [h_blk] bf16` (2 KiB). Local fragments: `x_local [hc, h_blk] fp32` (16 KiB across registers), `b_local [hc, h_blk] fp32` (also fp32 promoted), `a_local [hc, hc] fp32` (64B), `c_local [hc] fp32` (16B).
- Pipelining: serial outer loop over `ceildiv(hidden, h_blk)` (line 518) — **not** `T.Pipelined` (no double-buffer). The 4 hidden chunks are processed sequentially.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:684-687` (`Block.hc_post`):
```python
y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
```

```python
# Inputs:
#   a:  [T, hc=4, hc=4] fp32         (comb_mix)
#   b:  [T, hc, H=4096] bf16         (residual_in)
#   c:  [T, hc]         fp32         (post_mix)
#   d:  [T, H]          bf16         (current layer activation x)
#
# Output:
#   x:  [T, hc, H]      bf16
#
# Per token t, per hc-output stream i_hco, per hidden index i1_h:
#   x[t, i_hco, i1_h] = c[t, i_hco] * d[t, i1_h]
#                       + sum over i_hci of  a[t, i_hci, i_hco] * b[t, i_hci, i1_h]

# Pseudo per-CTA (i_n = token):
a_local = a[i_n, :, :].clone()                                  # [hc, hc] fp32
c_local = c[i_n, :].clone()                                     # [hc] fp32

for i0_h in range(ceildiv(H, h_blk)):
    b_local = b[i_n, :, i0_h*h_blk : (i0_h+1)*h_blk].float()    # [hc, h_blk] fp32
    d_local = d[i_n,    i0_h*h_blk : (i0_h+1)*h_blk].float()    # [h_blk]    fp32

    # T.Parallel(hc, h_blk) — across the 128 threads:
    for i_hco in range(hc):
        for i1_h in range(h_blk):
            x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
            for i_hci in range(hc):       # T.vectorized(hc) — 4-element FMA chain
                x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]

    # Write back as bf16:
    x[i_n, :, i0_h*h_blk : (i0_h+1)*h_blk] = x_local.to(torch.bfloat16)
```

Notes on fusion:

- This is the **standalone `hc_post` kernel** — no fusion with the next-layer GEMM (unlike #5 `mhc_fused_tilelang` which fuses both). Selected by `mhc_fused_post_pre_tilelang` when `T > 16` (prefill regime), and by the model's final post-mapping (`model.py:1084`) where there is no "next layer" anyway (the next op is `hc_head_fused_kernel_tilelang`).
- Element-wise + matrix term in one pass: `c * d` is a per-(hc, hidden) outer-product-like element-wise term, and `a.T @ b` is a small `[hc, hc] × [hc, h_blk]` matmul reduced over `hc`. The `T.vectorized(hc)` on line 526 unrolls the inner reduction across `hc=4` lanes — very tight on registers since hc=4 is tiny.
- The `[hc, h_blk]` parallel axis (line 524) spreads across the 128 threads of the CTA: `4 × 1024 = 4096` work items per chunk, 32 per thread, 4 chunks per token — total 4096 hidden elements per stream per token, balanced.
- bf16 round at write (line 529 via `T.copy(x_local, x[...])`) — TileLang implicit cast on the copy from fp32 fragment to bf16 global memory.
- **Pipelines into**:
  - In `mhc_fused_post_pre_tilelang` large-batch path: output `residual_cur` is then handed to `tf32_hc_prenorm_gemm` (DeepGEMM) or `_tilelang_hc_prenorm_gemm` (#3 / #4), then to `mhc_pre_big_fuse_with_norm_tilelang` (#2). HBM round-trip on `residual_cur` between this kernel and the GEMM — this is the cost that #5 avoids for small T.
  - In the final post-mapping (`model.py:1084`): output feeds `hc_head_fused_kernel_tilelang` (#7) which does the final hc → 1 collapse + RMSNorm.
- Inputs come from prior `mhc_pre_big_fuse_with_norm_tilelang` (#2): `post_mix`, `comb_mix`; and from `self.attn(...) / self.ffn(...)` for `x`.

## Config-dependent dispatch

- Activation condition: V4-Flash takes this kernel for `mhc_fused_post_pre_tilelang` whenever `num_tokens > 16` (prefill regime). It is also unconditionally the final-layer post-mapping at `model.py:1084` regardless of batch size — at that point there is no subsequent `hc_pre` to fuse with.
- Sibling: `mhc_fused_tilelang` (#5) takes over when `num_tokens ≤ 16` (decode regime). **Class A locked-alternative** by batch-size heuristic.
- No SM90/SM100 branch.
- No CuteDSL alternative.

Notes on fusion / pipeline:

- This is the simplest of the mHC kernels — pure element-wise + small-matmul + bf16-cast. It is the **post-only** reference path; the V4-Flash prefill chain at a layer boundary is `[mhc_pre_big_fuse_with_norm | attn/ffn | mhc_post_tilelang | hc_prenorm_gemm (DeepGEMM) | next mhc_pre_big_fuse_with_norm]` — 5 kernel launches vs the decode regime's 3.
- The kernel does not consume or produce split-K partials (`n_splits` is 1 by definition since there is no K-axis split here). Its output `residual_cur` is the full bf16 hc-stream residual ready for the next GEMM.
- The final-layer call at `model.py:1084` has no downstream `hc_pre` — the result feeds `hc_head_fused_kernel_tilelang` (#7) directly.
