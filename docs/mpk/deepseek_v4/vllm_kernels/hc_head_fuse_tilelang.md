# hc_head_fuse_tilelang

## Identity

- Source file: `vllm/model_executor/kernels/mhc/tilelang_kernels.py:718-812` (`hc_head_fuse_tilelang`, TileLang `@tilelang.jit`).
- Launch wrapper:
  - `vllm/model_executor/kernels/mhc/tilelang.py:613-641` (`hc_head_fused_kernel_tilelang`), registered as `torch.ops.vllm.hc_head_fused_kernel_tilelang` (`tilelang.py:678-683`).
  - Invoked at `vllm/model_executor/kernels/mhc/tilelang.py:630`.
- Language/DSL: **TileLang** (`@tilelang.jit(pass_configs={TL_DISABLE_WARP_SPECIALIZED: True, TL_DISABLE_TMA_LOWER: True, TL_PTXAS_REGISTER_USAGE_LEVEL: 10})`).
- Third-party dep: `tilelang`.
- Registered as opaque custom op: yes — `torch.ops.vllm.hc_head_fused_kernel_tilelang`.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/model_executor/kernels/mhc/tilelang.py:630` | `hc_head_fused_kernel_tilelang` wrapper | `residual: [T, 4, 4096]`, `fn: [4, 16384]`, `hc_scale: [1]`, `hc_base: [4]` | bf16 residual, fp32 fn/scale/base | always (when T > 0; see line 626 short-circuit) |
| `vllm/models/deepseek_v4/nvidia/model.py:1095` | `DeepseekV4Model.forward` final logits prelude (after `mhc_post_tilelang` and before `self.norm`) | same | same | runs once per forward pass (final post-network reduction) |

V4-Flash applies this once at the very end of the network — the **final hc_mult → 1 collapse** that produces the bf16 hidden state fed into `self.norm` (one more RMSNorm) and then `lm_head` for logits.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `residual` (`hs_flat`) | `[num_tokens, hc_mult, hidden_size]` | bf16 | row-major contiguous | Final hc-stream residual (output of last `mhc_post_tilelang` at `model.py:1084`). |
| `fn` | `[hc_mult, hc_dim = hc_mult * hidden_size]` | fp32 | row-major (= `self.hc_head_fn`, a `[hc_mult, hc_dim]` fp32 parameter from `model.py`) | hc_head projection weights. **Note: only `hc_mult` output rows** (vs the `hc_mult3 = 24` rows in #1/#2/#3/#4); this kernel only needs the *pre*-mix portion since `hc_head` has no post/comb terms. |
| `hc_scale` | `[1]` | fp32 | scalar buffer | Single scalar scale for the pre-mix logits (vs the `[3]` scale used in `hc_pre`). |
| `hc_base` | `[hc_mult]` | fp32 | flat | Per-mix bias (4 values). |
| `hidden_size` | constexpr | int | — | 4096 on V4-Flash. |
| `rms_eps` | scalar (kwarg) | fp32 | — | RMSNorm ε (V4-Flash: `config.rms_norm_eps`). |
| `hc_eps` | scalar (kwarg) | fp32 | — | Post-sigmoid floor (V4-Flash: `hc_eps=1e-6`). |
| `hc_mult` | constexpr (kwarg) | int | — | 4 on V4-Flash. |
| `n_thr` | constexpr (kwarg) | int | — | 128 (kernel default). |
| `h_blk` | constexpr (kwarg) | int | — | 1024 default; `h_block = gcd(1024, hidden_size)` = 1024 for V4-Flash. |

**Constexpr/derived**: `hc_dim = hc_mult * hidden_size = 16384`; `h_block = gcd(1024, 4096) = 1024`; `n_h = hidden_size / h_block = 4`.

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `out` | `[num_tokens, hidden_size]` | bf16 | row-major | Pre-norm weighted sum: `out[t, :] = sum_{m_c} pre_mix[m_c] * residual[t, m_c, :]`. **RMSNorm denominator is computed in-kernel** but γ is **not** applied here (the caller runs `self.norm(hidden_states)` separately at `model.py:1103`). |

## Grid / Block

- `grid_dim = (num_tokens,)` — one CTA per token.
- `block_dim`: `n_thr = 128` (4 warps).
- Autotune configs: none.
- PDL: `T.pdl_sync()` at line 752, `T.pdl_trigger()` at line 811.
- Shared memory: `pre_mix_shared [hc_mult=4] fp32` (16B), `xs [hc_mult, h_block=1024] bf16` (8 KiB, pass 2 only), `xl` fragments (`[h_block]` fp32 ≈ 4 KiB across registers per warp).
- Reducers: `sqrsum_r [1] fp32` with `replication="all"` (cross-thread reducer — implemented as warp-/block-level reduction by TileLang lowering), `mixes_r [hc_mult] fp32` same. Both finalized at line 778-779.
- Pipelining: pass 2 (line 795) uses `T.Pipelined(n_h=4, num_stages=2)`. Pass 1 (line 764) is `T.serial(hc_mult)` × `T.serial(n_h)` — sequential.
- `disable_tma=True` on copies at lines 798, 808 — forces non-TMA loads on the residual reads in pass 2 (likely because TMA conflicts with the reducer-based pass-1 outputs which TileLang keeps as register-resident).

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:729-736` (`ParallelHead.hc_head`):
```python
x = x.flatten(2).float()
rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
mixes = F.linear(x, hc_fn) * rsqrt
pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
return y.to(dtype)
```

This kernel is the **fused version** of the entire `hc_head` body. Two passes share the residual tile in a single CTA:

```python
# Inputs:
#   residual: [T, hc_mult=4, H=4096] bf16
#   fn:       [hc_mult, hc_dim = hc_mult*H = 16384] fp32
#   hc_scale: [1] fp32
#   hc_base:  [hc_mult] fp32
#
# Per token t:
# Pass 1 — accumulate sqrsum (over all hc_mult * H elements) and hc_mult dot-products:
sqrsum = 0.0                                          # scalar
mixes  = torch.zeros(hc_mult)                         # [hc_mult]
for m_c in range(hc_mult):                            # hc residual channel
    x_chan = residual[t, m_c, :].float()              # [H]
    sqrsum += (x_chan ** 2).sum()
    # Project x_chan against fn rows. The fn matrix is [hc_mult, hc_mult*H]
    # where the H-axis is concatenated across all m_c channels: fn[m_m, m_c*H + h]
    # is the weight from x_chan[h] to mix m_m. So mixes[m_m] = sum_{m_c, h} x[m_c, h] * fn[m_m, m_c*H + h].
    for m_m in range(hc_mult):
        fn_row = fn[m_m, m_c*H : (m_c+1)*H]           # [H]
        mixes[m_m] += (x_chan * fn_row).sum()

# rsqrt over the hc-flattened residual (denominator hc_dim = hc_mult*H):
rsqrt = torch.rsqrt(sqrsum / hc_dim + rms_eps)

# Pre-mix sigmoid (single scale[0] in hc_head; vs hc_pre's scale[0..2]):
pre_mix = torch.sigmoid(mixes * rsqrt * hc_scale[0] + hc_base) + hc_eps    # [hc_mult]

# Pass 2 — pipelined weighted sum across hc streams into out:
for i0_h in range(n_h):                               # n_h = 4 chunks of h_block=1024
    xs = residual[t, :, i0_h*h_block : (i0_h+1)*h_block].float()    # [hc_mult, h_block]
    ol = (pre_mix.view(hc_mult, 1) * xs).sum(dim=0)                # [h_block]
    out[t, i0_h*h_block : (i0_h+1)*h_block] = ol.to(torch.bfloat16)
```

Notes on fusion:

- **Single-pass-then-two-passes structure**: pass 1 reads the residual twice (once per `m_c` iteration for sqrsum + per-`m_m` projections), but writes nothing to HBM. The reducers `sqrsum_r` and `mixes_r` finalize via cross-thread reduction (TileLang's `T.alloc_reducer(..., replication="all")` semantics — all threads end up holding the reduced value).
- Pass 2 is a clean pipelined hidden-axis sweep — the same pattern as the no-norm-variant tail of #1, but with no separate norm pass since RMSNorm γ is applied by the *next* op (`self.norm` at `model.py:1103`), not by this kernel.
- **Two reads of `residual` per token**: pass 1 streams it for the reducers, pass 2 streams it again for the weighted sum. This is intentional — caching the full `[hc_mult, H]` = 32 KiB residual in shared memory would exceed the budget for a 128-thread CTA at H=4096. The HBM bandwidth on B200 absorbs the duplicate reads cheaply.
- **fn layout subtlety**: `fn` is `[hc_mult, hc_dim]` where `hc_dim = hc_mult * H`, and the projection is `mixes[m_m] = sum_{m_c, h} x[m_c, h] * fn[m_m, m_c*H + h]`. The kernel accesses `fn[m_m, m_c * hidden_size + i_h * h_block]` (line 774), confirming this row-major hc-concatenated layout.
- **No comb / no post**: `hc_head` has no Sinkhorn iteration and no post-mix — it is strictly the `hc → 1` collapse via sigmoid-gated weighted sum of the hc streams. That is why `fn` here is `[hc_mult, ...]` not `[hc_mult3, ...]`.
- **Pipelines into**: `self.norm(hidden_states)` (vLLM native `RMSNorm`, `model.py:1103`) → `lm_head` for logits. PDL trigger on this kernel's exit overlaps with `self.norm`'s prologue.
- Input comes from `mhc_post_tilelang` (#6) called at `model.py:1084` — the final hc-stream residual.

## Config-dependent dispatch

- Activation condition: V4-Flash unconditionally invokes this kernel once per forward pass on the last PP rank (`model.py:1088, 1095`). Single call site.
- Wrapper short-circuit: when `num_tokens == 0` (empty batch — guarded at `tilelang.py:626-627`), the wrapper returns an empty `out` tensor without launching the kernel.
- No sibling alternatives. No SM90/SM100 branch. No CuteDSL alternative.
- Note: the V4-Flash `ParallelHead` / `MTPHead` reference at `model.py:719-736` (`ParallelHead.forward` / `hc_head`) also includes the post-norm `self.norm(x)` step; the vLLM model code applies `self.norm` as a separate op rather than fusing it into this kernel (compare to #2 which fuses `attn_norm` into the pre-side). The norm γ scale is small (4096 bf16 fp32-promoted) so a separate launch is acceptable; the heavy compute here is the hc-flatten + sqrsum + projection.

Notes on fusion / pipeline:

- This is the **terminal mHC kernel** — collapses `[T, hc_mult, H]` → `[T, H]` for downstream `lm_head` logits. It is the mirror of the `hidden_states.unsqueeze(-2).repeat(1, hc_mult, 1)` expansion at `model.py:1065` that initializes the hc-stream state at network entry.
- Fuses RMSNorm denominator (sqrsum + rsqrt) + the `mixes = F.linear * rsqrt` projection + sigmoid+eps gating + the hc-weighted sum — five reference ops in one kernel, no HBM round-trips between them.
- Downstream `self.norm` re-uses the bf16 output and applies a *second* RMSNorm — note that this second norm is over the collapsed `[T, H]` shape, not the hc-flatten denominator used inside this kernel. The two are distinct RMSNorms (hc-flatten vs single-stream), so this kernel cannot fuse the second one without changing semantics.
