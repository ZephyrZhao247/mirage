# Kernel spec template — canonical worked example

This file is the canonical reference for vLLM-canonical kernel specs under
`docs/mpk/deepseek_v4/vllm_kernels/`. Every `<kernel>.md` in this directory
must follow the section structure below. The body is a fully filled-out
example for the Triton kernel `fused_inv_rope_fp8_quant` at
`deps/vllm/vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`.

## Mandatory sections (in order, headings verbatim)

- `## Identity`
- `## Call sites`
- `## Inputs`
- `## Outputs`
- `## Grid / Block`
- `## Math`
- `## Config-dependent dispatch`

Use "none" rather than eliding a section.

## Scope reminders for spec authors

- **Platform**: NVIDIA only. A kernel is in scope iff reachable from `deps/vllm/vllm/models/deepseek_v4/nvidia/model.py`. `common/ops/*` is in scope when called from NVIDIA. `amd/` subtree is OUT.
- **Math citation**: every `## Math` section must cite `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:<lines>` as the semantic source of truth. Express logic with PyTorch operators that mirror the kernel's **fusion structure** (not the unfused decomposition).
- **Variant coverage**: when a kernel is selected by config:
  - **Class A** (locked by V4-Flash config: `expert_dtype=fp8`, `score_func=sqrtsoftplus`, NVIDIA platform, SM100 codepath, FlashMLA over Aiter) → document only the active path; note alternatives by source pointer.
  - **Class B** (`use_fp4_cache`, CuteDSL vs Triton when both NVIDIA-reachable, `moe_backend=mega_moe` vs `FusedMoE`, per-layer `compress_ratio ∈ {0,4,128}`) → both variants get their own spec file; mark **DECISION REQUIRED** in `## Config-dependent dispatch`.
- **Web search**: agents may use WebSearch/WebFetch ONLY when documenting DeepGEMM, FlashMLA, or TileLang kernels. For all other kernels, vLLM source only.
- **Forbidden reads**: no MPK source, no `docs/mpk/deepseek_v4/kernels/` (existing MPK-side specs), no `amd/` subtree.

## Routine `nn.Linear` instances

These collapse into a single `linear_cublas.md` with a table of (instance name, in_dim, out_dim, dtype, parallelism, caller). Do NOT create one spec per Linear instance.

---

# fused_inv_rope_fp8_quant

## Identity
- Source file: `deps/vllm/vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py:17-136` (Triton JIT body `_fused_inv_rope_fp8_quant_per_head`); user-facing entry `fused_inv_rope_fp8_quant` at lines 138-211.
- Launch wrapper: `_fused_inv_rope_fp8_quant_kernel_impl` at lines 214-277.
- Language/DSL: **Triton** (`@triton.jit`).
- Third-party dep: none (pure Triton + PyTorch).
- Registered as opaque custom op: `torch.ops.vllm.fused_inv_rope_fp8_quant_kernel`.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `deps/vllm/vllm/models/deepseek_v4/attention.py:319` | `DeepseekV4MultiHeadLatentAttentionWrapper._o_proj_path` | `o: [T, n_heads=64, head_dim=512]`, `positions: [T] int64`, `cos_sin_cache: [max_pos, rope_dim=64] fp32` | bf16 attention output | always on NVIDIA path (ROCm uses `rocm_inv_rope_einsum` at attention.py:307 instead) |

Single call site on NVIDIA. Runs once per attention-layer forward, immediately after `flash_mla_with_kvcache` / `flash_mla_sparse_fwd` returns.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `o` | `[num_tokens, num_heads = n_groups * heads_per_group, head_dim = nope + rope]` | bf16 | row-major, contiguous on `head_dim` | Attention output, RoPE still applied on last `rope_dim` dims |
| `positions` | `[num_tokens]` | int64 | contiguous | Absolute token positions for inverse-RoPE de-rotation |
| `cos_sin_cache` | `[max_pos, rope_dim=64]` | fp32 | first `rope_dim/2` lanes = cos, second half = sin | Precomputed RoPE cache shared with forward RoPE |
| `n_groups` | scalar | int | — | Number of o-projection groups (V4-Flash: `o_groups=8`) |
| `heads_per_group` | scalar | int | — | `num_heads // n_groups` (V4-Flash: 8) |
| `nope_dim` | scalar | int | — | Non-RoPE dims per head (V4-Flash: 448) |
| `rope_dim` | scalar | int | — | RoPE dims per head (V4-Flash: 64); must be even |
| `quant_group_size` | scalar | int | — | FP8 block size (default 128); requires `head_dim % 128 == 0` and `nope_dim % 128 == 128 - rope_dim` |
| `tma_aligned_scales` | scalar | bool | — | **Class A / locked by SM** — True on SM100 (UE8M0-packed INT32 scales), False on SM90 (fp32 scales). See dispatch section. |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `o_fp8` | `[n_groups, num_tokens, d = heads_per_group * head_dim]` (after `.transpose(0,1)`: caller-visible shape is `[num_tokens, n_groups, d]`) | `float8_e4m3fn` | group-major, contiguous along last dim | FP8-quantized attention output ready for `fp8_einsum("bhr,hdr->bhd", ...)` against `wo_a` |
| `o_scale` | `[n_groups, num_tokens, scale_inner]`; `scale_inner = (num_scale_blocks + 3) // 4` (SM100, packed UE8M0) or `num_scale_blocks` (SM90) | int32 (SM100) / fp32 (SM90) | TMA-aligned MN-major: `as_strided` to `(scale_inner * tma_aligned_T, 1, tma_aligned_T)`; padding rows zero-filled | Pre-transformed scales — `fp8_einsum` skips `transform_sf_into_required_layout` |

`tma_aligned_T = get_tma_aligned_size(num_tokens, 4)`. Padding CTAs (`pid_token ∈ [num_tokens, tma_aligned_T)`) zero-fill scales and skip FP8 writes.

On SM100 the kernel packs 4 consecutive blocks' UE8M0 exponent bytes into one int32: `packed = Σ_k ((scale_bits[k] >> 23) & 0xFF) << (k*8)`.

## Grid / Block

- `grid_dim = (tma_aligned_T, n_groups * heads_per_group)` — one CTA per (token-slot, global-head).
- `block_dim` (threads/CTA): driven by `num_warps=1` → 32 threads/CTA. Single warp per CTA.
- Autotune configs: **none** — `num_stages=1`, `num_warps=1` hard-coded at launch (file lines 273-275). `launch_pdl=False` on CUDA, no PDL on ROCm/XPU.
- Per-CTA work: a single CTA loads its `[HEAD_DIM = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE]` slice of `o` for one (token, head), inverse-rotates the trailing `rope_dim` lanes in-register, computes per-`quant_group_size` block absmax + UE8M0-rounded scale, writes FP8 and scale.
- Int64 index trick: `pid_token` and `pid_gh` are cast to int64 because stride multiplication overflows int32 once `num_tokens ≥ 32768` (lines 41-43 comment).

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:534` (`apply_rotary_emb(o[..., -rd:], freqs_cis, True)` — the `inverse=True` flag at `apply_rotary_emb` lines 232-242). The FP8 block-quant step is NOT in the reference (reference uses bf16 throughout for simplicity; see model.py:539 comment); the vLLM fused kernel adds it to feed the FP8 `wo_a` einsum.

```python
# PyTorch-operator equivalent of one CTA's work for token t, global head h:
#
# Inputs:
#   o:           [T, H, D] bf16     (D = nope_dim + rope_dim, e.g. 448+64=512)
#   positions:   [T] int64
#   cos_sin:     [max_pos, rope_dim] fp32  (first half cos, second half sin)
#
x_full = o[t, h, :].to(torch.float32)              # [D]
pos    = positions[t]
cos    = cos_sin[pos, : rope_dim // 2]              # [rope_dim/2]
sin    = cos_sin[pos, rope_dim // 2 :]              # [rope_dim/2]

# Inverse RoPE on the trailing rope_dim lanes (last QUANT_GROUP_SIZE chunk).
# Pairing is even/odd interleaved — kernel fetches partner via offset ^ 1.
x_rope = x_full[-rope_dim:].view(rope_dim // 2, 2)
even = x_rope[:, 0] *  cos + x_rope[:, 1] * sin
odd  = x_rope[:, 0] * -sin + x_rope[:, 1] * cos
x_full[-rope_dim:] = torch.stack([even, odd], dim=-1).view(-1)

# Block-FP8 quant with UE8M0-rounded scale (ceil log2 → exp2):
chunks   = x_full.view(CHUNKS_PER_HEAD, QUANT_GROUP_SIZE)   # e.g. 4 × 128
absmax   = chunks.abs().amax(dim=-1).clamp_min(1e-10)       # [CHUNKS_PER_HEAD]
scale    = torch.exp2(torch.ceil(torch.log2(absmax / fp8_max)))  # UE8M0-representable
x_q      = (chunks / scale.unsqueeze(-1)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)

# Write FP8 to o_fp8[g, t, qb_start*GROUP_SIZE : (qb_start+CHUNKS_PER_HEAD)*GROUP_SIZE]
o_fp8[g, t, qb_start*GROUP_SIZE : (qb_start+CHUNKS_PER_HEAD)*GROUP_SIZE] = x_q.view(-1)

# Scale write — two formats (Class A dispatch on SM):
if tma_aligned_scales:    # SM100: pack 4 UE8M0 exponent bytes into one int32
    ue8m0 = ((scale.to(torch.int32, bitcast=True) >> 23) & 0xFF)
    packed = (ue8m0 << (torch.arange(CHUNKS_PER_HEAD) * 8)).sum()
    o_scale[g, t, head_in_group] = packed
else:                     # SM90: write fp32 scale per block
    o_scale[g, t, qb_start : qb_start + CHUNKS_PER_HEAD] = scale
```

Notes on fusion / quant:
- The `inverse=True` RoPE in `model.py:236-240` is `freqs_cis.conj()` then complex-multiply; the kernel materializes it as the `x*cos + x_partner*sin` / `x*cos - x_partner*sin` pair selected by `is_even`.
- UE8M0: since `scale = 2^k` exactly, only the 8-bit IEEE-754 exponent matters. The kernel pulls the exponent field directly (`>> 23 & 0xFF`) — no rounding needed because `exp2(ceil(log2(x)))` is already a power of two.
- Stride trick: `o_scale` is allocated as a flat int32 buffer with `as_strided` to a 3D view where the middle dim has stride 1. After the wrapper's final `.transpose(0,1)`, the returned tensor is in the exact layout `fp8_einsum` expects with `transform_sf_into_required_layout` skipped.

## Config-dependent dispatch

- Activation condition: always on NVIDIA `_o_proj_path` (attention.py:319). ROCm uses `rocm_inv_rope_einsum` (attention.py:307) — out of scope (NVIDIA-only this wave).
- Variants on NVIDIA:
  - `tma_aligned_scales=True` (SM100, UE8M0-packed INT32 scales) vs `tma_aligned_scales=False` (SM90, FP32 scales) — **Class A / locked** for V4-Flash on B200 (SM100, so `True`). The flag is set in `DeepseekV4MultiHeadLatentAttentionWrapper.__init__` based on `current_platform.has_device_capability((10, 0))`. SM90 alternative branch lives in the SAME kernel via `TMA_ALIGNED_SCALES: tl.constexpr` (no separate spec needed).
  - No CuteDSL alternative for this kernel — Triton is the only NVIDIA path.
- Downstream consumer constraint: outputs feed `fp8_einsum("bhr,hdr->bhd", (o_fp8, o_scale), (wo_a_fp8, wo_a_scale), z, recipe=self._einsum_recipe)` at attention.py:338. Any spec change here must preserve the `o_fp8` shape `[n_groups, T, d]` and the scale's TMA-aligned MN-major layout exactly, or the skipped `transform_sf_into_required_layout` reads garbage.
