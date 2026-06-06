# deepseek_v4_fp8_einsum

## Identity
- Source file (vLLM wrapper): `vllm/utils/deep_gemm.py:298-302` (`fp8_einsum` thin dispatcher) — late-binds to `_fp8_einsum_impl`, resolved at `_lazy_init` lines 178-244 via `_fp8_einsum_impl = getattr(_dg, "fp8_einsum", None)` (line 222) where `_dg` is the deep_gemm module from either site-packages or the vendored copy (`_import_deep_gemm` lines 143-175).
- Vendored DeepGEMM entry: `vllm/third_party/deep_gemm/__init__.py:60-61` re-exports `einsum, fp8_einsum` from the compiled `_C` extension. The kernel body lives in DeepGEMM's CUDA sources (out of repo for compiled wheels); device implementation is **`sm_100a` DeepGEMM einsum** (per dispatch recipe — see below) or **SM90 cuBLAS** fallback.
- Language/DSL: **CUDA** (DeepGEMM compiled kernel, accessed as a Python C-extension binding).
- Third-party dep: **DeepGEMM** (`deep_gemm` package or vendored `vllm.third_party.deep_gemm`).
- Registered as: Python-level callable `vllm.utils.deep_gemm.fp8_einsum` (NOT a `torch.ops` op). Falls back to `_missing()` raising `RuntimeError("DeepGEMM backend is not available or outdated.")` when DeepGEMM cannot be imported (`vllm/utils/deep_gemm.py:120-125, 298-302`).

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/attention.py:338-344` | `DeepseekV4MultiHeadLatentAttentionWrapper.forward` (NVIDIA `_o_proj_path`, post-attention) | `(o_fp8 [T, n_groups, d], o_scale [T, n_groups, scale_inner])`, `(wo_a_fp8, wo_a_scale)`, `z [T, n_groups, o_lora_rank]` | fp8 ×  fp8 → bf16 | NVIDIA only (ROCm branch at attention.py:307 short-circuits before this call); selected over cuBLAS path via `recipe = (1, 1, 128)` on `cap.major >= 10` (attention.py:198) |

Only one production call site in DeepseekV4 attention. (Other modules in vLLM also call `fp8_einsum`; this spec is scoped to the V4-Flash o-projection caller per the user's batch.)

Dispatch logic (attention.py:193-199):
```python
cap = current_platform.get_device_capability()
assert cap is not None, "DeepseekV4 attention requires a CUDA device"
self._einsum_recipe = (1, 128, 128) if cap.major <= 9 else (1, 1, 128)
self._tma_aligned_scales = cap.major >= 10
```
For V4-Flash on B200, `cap.major == 10` so `_einsum_recipe = (1, 1, 128)` and `_tma_aligned_scales = True` — the call at line 338 hits the DeepGEMM `sm_100a` path.

## Inputs

Signature (from caller `attention.py:338-344`): `fp8_einsum(equation, lhs_tuple, rhs_tuple, out, recipe)`.

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `equation` | scalar | str | — | Always `"bhr,hdr->bhd"` for this caller. `b = num_tokens (T)`, `h = n_local_groups (8/TP)`, `r = d = heads_per_group * head_dim` (V4-Flash: 8 × 512 = 4096), `d = o_lora_rank` (V4-Flash: 1024). |
| `lhs = (o_fp8, o_scale)` | `o_fp8: [T, n_groups, d]`; `o_scale: [n_groups, T, scale_inner]` (returned by `fused_inv_rope_fp8_quant` after `.transpose(0,1)`; on-device buffer is allocated as `[n_groups, T, ...]`) | `o_fp8: float8_e4m3fn`; `o_scale: int32` (UE8M0-packed, `sm_100a`) | Both group-major (stride convention from `fused_inv_rope_fp8_quant`'s `as_strided` view, attention.py:319-328). `o_fp8.stride()` per the source: `(d, T*d, 1)`. `o_scale` is TMA-aligned MN-major: `(scale_inner * tma_aligned_T, 1, tma_aligned_T)`. `_tma_aligned_scales=True` ⇒ scales already in DeepGEMM's expected layout. | LHS = quantized attention output and its block scales (per `head_dim/128 = 4` blocks per head). |
| `rhs = (wo_a_fp8, wo_a_scale)` | `wo_a_fp8: [n_groups, o_lora_rank, d]`; `wo_a_scale: shape per the checkpoint quant recipe` | `wo_a_fp8: float8_e4m3fn`; `wo_a_scale: fp32` or `int32` (UE8M0) per checkpoint | `wo_a.weight` and `wo_a.weight_scale_inv` from `attention.py:330-331`. Reference geometry from `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:462,538`: `wo_a` is a `ColumnParallelLinear` of `(n_heads * head_dim // n_groups, n_groups * o_lora_rank)` viewed as `(n_local_groups, o_lora_rank, -1)` — i.e. `[n_local_groups, o_lora_rank, d]`. | RHS = wo_a weight + scales (FP8 block-quantized in the checkpoint). |
| `out` (`z`) | `[T, n_local_groups, o_lora_rank]` | bf16 | row-major contiguous (allocated via `torch.empty` at attention.py:333-337) | Output buffer for `bhd` index — written in place; the function does not return it. |
| `recipe` | tuple | `(int, int, int)` | — | DeepGEMM scale-granularity tuple `(sfa_gran_m, sfa_gran_n, sfb_gran_mn)`. **`sm_100a`: `(1, 1, 128)`**. SM90 locked alternative: `(1, 128, 128)`. See dispatch section. |

Notes:
- Per attention.py:188-190, `self._wo_a_act_quant.use_deep_gemm_supported = False`. The comment ("Bypass packed-for-deepgemm path — we need FP32 scales … so fp8_einsum can handle layout transform internally") refers to a separate runtime activation-quant path; the active `o_fp8`/`o_scale` here are pre-transformed by `fused_inv_rope_fp8_quant` so DeepGEMM's `fp8_einsum` reads them without any further `transform_sf_into_required_layout` call.
- `o_scale` and `wo_a_scale` dtypes per dispatch: `sm_100a` uses int32 UE8M0-packed scales (4 exponent bytes per int32, packed by `fused_inv_rope_fp8_quant` lines 120-130); SM90 uses fp32 block scales.

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `out` (alias `z`) — in-place write | `[T, n_local_groups, o_lora_rank]` (V4-Flash: `[T, 8/TP, 1024]`) | bf16 | row-major contiguous | Result of `Σ_r o_fp8[t, h, r] * wo_a_fp8[h, d, r]` with block-scale correction. Consumed immediately by `self.wo_b(z.flatten(1))` at attention.py:346. |

`fp8_einsum` returns `None` from the vLLM wrapper's POV (the compiled DeepGEMM kernel writes through `out`). The wrapper at `vllm/utils/deep_gemm.py:298-302` does `return _fp8_einsum_impl(*args, **kwargs)`, but the call site discards the return.

## Grid / Block

Grid and block dimensions are owned by the DeepGEMM `sm_100a` einsum kernel — not authored in this repo. Configuration knobs observable from vLLM:
- `recipe = (1, 1, 128)` (sm_100a) selects DeepGEMM's per-row scales on LHS (`sfa_gran_m=1, sfa_gran_n=1`) and 128-wide column blocks on RHS (`sfb_gran_mn=128`).
- `set_num_sms(...)` / `get_num_sms()` (`vllm/utils/deep_gemm.py:246-259`) bound the SM count made available to DeepGEMM; vLLM does not override it in the V4-Flash attention path.
- `set_pdl(...)` (re-exported at `vllm/third_party/deep_gemm/__init__.py:25`) controls programmatic launch dependency; default-on for `sm_100a`.
- TMA alignment: `_tma_aligned_scales=True` on `sm_100a` (`cap.major >= 10`) means `o_scale` (and the checkpoint's `wo_a_scale` when prepared accordingly) already satisfy DeepGEMM's TMA-aligned MN-major layout; the kernel skips `transform_sf_into_required_layout` (`vllm/utils/deep_gemm.py:332-338`).

The vLLM wrapper does no autotuning of its own; DeepGEMM internally JIT-tunes (cache directory `$VLLM_CACHE_ROOT/deep_gemm` per lines 209-214).

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:537-542`:
```
o = o.view(bsz, seqlen, self.n_local_groups, -1)                      # line 537
wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)  # line 538
# NOTE: wo_a is FP8 in checkpoint; could do FP8 einsum here for better perf,
# but using BF16 for simplicity.                                       # 539-540
o = torch.einsum("bsgd,grd->bsgr", o, wo_a)                            # line 541
```
The reference uses `"bsgd,grd->bsgr"` (4 indices on LHS, with `b*s` flattened to `b` in the deployed code so the equation becomes `"bhr,hdr->bhd"`). `g` (groups) → `h`, `d` (per-group heads × head_dim) → `r`, and the output's `r` (= `o_lora_rank`) is renamed `d` in the deployed equation. Semantics are identical to the reference; the kernel adds FP8 block-scale arithmetic that the reference comment (model.py:539-540) acknowledges as the production recipe.

```python
# PyTorch-operator equivalent — equation "bhr,hdr->bhd":
#
# Inputs:
#   o_fp8:      [T, G, R]    float8_e4m3fn   (R = heads_per_group * head_dim)
#   o_scale:    [G, T, R/128] int32 (UE8M0 sm_100a packed) or fp32 (SM90)
#   wo_a_fp8:   [G, D, R]    float8_e4m3fn   (D = o_lora_rank)
#   wo_a_scale: [G, D, R/128] (UE8M0 or fp32 per checkpoint)
#   recipe = (sfa_gran_m=1, sfa_gran_n=1, sfb_gran_mn=128) on sm_100a
#
# Per-element (t ∈ [T], h ∈ [G], d ∈ [D]):
acc = 0.0
for r_blk in range(R // 128):
    r_slice = slice(r_blk*128, (r_blk+1)*128)
    # Dequantize LHS block: scalar scale per (t, h, r_blk).
    a = o_fp8[t, h, r_slice].to(torch.float32)
    if sm_100a:
        # UE8M0: extract 8-bit exponent for this block from the int32 packed lane.
        ue8m0_byte = (o_scale[h, t, r_blk // 4] >> ((r_blk % 4) * 8)) & 0xFF
        sa = torch.exp2(torch.tensor(ue8m0_byte - 127, dtype=torch.float32))
    else:
        sa = o_scale[h, t, r_blk].to(torch.float32)         # fp32 scale per block
    # Dequantize RHS block: scalar scale per (h, d, r_blk) — sfb_gran_mn=128.
    b = wo_a_fp8[h, d, r_slice].to(torch.float32)
    sb = wo_a_scale[h, d // 128 if sm90 else d, r_blk].to(torch.float32)
    # Accumulate in fp32; DeepGEMM uses tensor cores with block-scale dequant.
    acc += (a * sa) @ (b * sb)
out[t, h, d] = acc.to(torch.bfloat16)
```

Notes on fusion / quant:
- DeepGEMM `sm_100a` `fp8_einsum` fuses the block-scale dequantization into the GEMM main loop — fp32 accumulator is biased per block by `(sa * sb)`. UE8M0 makes the per-block scale a single byte (8-bit exponent), so per-tensor-core-block scale lookup is one byte-load + bit-cast + `exp2`.
- The `(1, 1, 128)` recipe means LHS scales are per-element along the `b` (token) and `h` (group) axes — `o_scale` has one int32 per `(group, token, r_blk_quad)` and packs 4 r-blocks per int32 — and RHS scales are per `128`-wide block along the contracted `r` axis. This matches the layout `fused_inv_rope_fp8_quant` produces with `tma_aligned_scales=True`.
- The `wo_a_scale` layout/dtype is set at checkpoint-load time by the weight-loader for `mla_modules.wo_a` (a `ColumnParallelLinear` with FP8 block quant). The recipe `(1, 1, 128)` is consistent only when both LHS and RHS scales use UE8M0 with the same `sfb_gran_mn=128` blocking on the contraction axis.
- The reference comment at model.py:539-540 says "using BF16 for simplicity" — production runs FP8 here; the einsum bit-equivalence to the reference holds modulo the block-quant noise.

## Config-dependent dispatch

- Activation condition: NVIDIA only (ROCm short-circuits at attention.py:306-316). Profile run is gated upstream — `o` is non-degenerate by the time this fires.
- SM dispatch (attention.py:193-199):
  - **`sm_100a` (V4-Flash B200, `cap.major == 10`)** — active path. `recipe = (1, 1, 128)`, `tma_aligned_scales = True`. DeepGEMM `fp8_einsum` runs the `sm_100a` UE8M0-aware kernel; scales pre-packed by `fused_inv_rope_fp8_quant` (the matching producer spec); `transform_sf_into_required_layout` is skipped.
  - **SM90 (Hopper, `cap.major <= 9`)** — **locked alternative**, one-line pointer: same `fp8_einsum` entry, but `recipe = (1, 128, 128)` and `tma_aligned_scales = False` (fp32 scales). Per attention.py comment lines 188-190: a separate cuBLAS-flavored path applies because the SM90 DeepGEMM einsum expects fp32 block scales — `o_scale` is produced as fp32 by `fused_inv_rope_fp8_quant` on SM90 via the same kernel's `TMA_ALIGNED_SCALES=False` branch. No separate spec; routing differs by recipe alone.
  - **ROCm** — out of scope. The ROCm caller (`rocm_inv_rope_einsum` at attention.py:307) keeps wo_a in BF16 and does not invoke `fp8_einsum`.
- DeepGEMM availability gate (`vllm/utils/deep_gemm.py:298-302`): if DeepGEMM cannot be imported (`_fp8_einsum_impl is None`), `_missing()` raises. There is no Triton/native fallback for this einsum — DeepGEMM is required on NVIDIA V4-Flash.
- Per user decision (D5 / batch scope): focus on `sm_100a`. SM90 noted as the locked alternative via recipe `(1, 128, 128)`; no CuteDSL / TileLang variants in scope.
- Downstream consumer constraint: `out` (`z`) feeds `self.wo_b(z.flatten(1))` at attention.py:346 — a `RowParallelLinear` taking `[T, n_local_groups * o_lora_rank]` bf16. The in-place write must preserve `z`'s `[T, n_local_groups, o_lora_rank]` shape and bf16 dtype exactly.
