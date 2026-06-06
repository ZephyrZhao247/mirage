# fused_inv_rope_fp8_quant

## Identity
- Source file: `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py:17-136` (Triton JIT body `_fused_inv_rope_fp8_quant_per_head`); user-facing entry `fused_inv_rope_fp8_quant` at lines 138-211.
- Launch wrapper: `_fused_inv_rope_fp8_quant_kernel_impl` at lines 214-277.
- Language/DSL: **Triton** (`@triton.jit`).
- Third-party dep: none (pure Triton + PyTorch).
- Registered as opaque custom op: `torch.ops.vllm.fused_inv_rope_fp8_quant_kernel` (registered via `direct_register_custom_op` at lines 314-318).

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/attention.py:319` | `DeepseekV4MultiHeadLatentAttentionWrapper.forward` (NVIDIA `_o_proj_path` branch) | `o: [T, n_local_heads=64, head_dim=512]`, `positions: [T] int64`, `cos_sin_cache: [max_pos, rope_dim=64] fp32` | bf16 attention output | always on NVIDIA path (ROCm uses `rocm_inv_rope_einsum` at attention.py:307 instead) |
| `tests/kernels/test_fused_inv_rope_fp8_quant.py:293,348,392,424,460,475,527,559,595,629,791,897` | unit-test cases | matches above | bf16 | test-only |

Single production call site. Runs once per attention-layer forward, immediately after `flash_mla_with_kvcache` / `flash_mla_sparse_fwd` returns `o_padded` and the wrapper slices to `o = o_padded[:, : n_local_heads, :]` (attention.py:303).

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
| `tma_aligned_scales` | scalar | bool | — | **Class A / locked by SM** — True on `sm_100a` (UE8M0-packed INT32 scales), False on SM90 (fp32 scales). See dispatch section. |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `o_fp8` | `[n_groups, num_tokens, d = heads_per_group * head_dim]` (after `.transpose(0,1)`: caller-visible shape is `[num_tokens, n_groups, d]`) | `float8_e4m3fn` | group-major, contiguous along last dim | FP8-quantized attention output ready for `fp8_einsum("bhr,hdr->bhd", ...)` against `wo_a` |
| `o_scale` | `[n_groups, num_tokens, scale_inner]`; `scale_inner = (num_scale_blocks + 3) // 4` (`sm_100a`, packed UE8M0) or `num_scale_blocks` (SM90) | int32 (`sm_100a`) / fp32 (SM90) | TMA-aligned MN-major: `as_strided` to `(scale_inner * tma_aligned_T, 1, tma_aligned_T)`; padding rows zero-filled | Pre-transformed scales — `fp8_einsum` skips `transform_sf_into_required_layout` |

`tma_aligned_T = get_tma_aligned_size(num_tokens, 4)`. Padding CTAs (`pid_token ∈ [num_tokens, tma_aligned_T)`) zero-fill scales and skip FP8 writes.

On `sm_100a` the kernel packs 4 consecutive blocks' UE8M0 exponent bytes into one int32: `packed = Σ_k ((scale_bits[k] >> 23) & 0xFF) << (k*8)`.

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
if tma_aligned_scales:    # sm_100a: pack 4 UE8M0 exponent bytes into one int32
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
  - `tma_aligned_scales=True` (`sm_100a`, UE8M0-packed INT32 scales) — the active V4-Flash B200 path. Flag set in `DeepseekV4MultiHeadLatentAttentionWrapper.__init__` via `current_platform.has_device_capability((10, 0))` (attention.py:199: `self._tma_aligned_scales = cap.major >= 10`).
  - `tma_aligned_scales=False` (SM90, FP32 scales) — **locked alternative** dispatched inside the SAME kernel via `TMA_ALIGNED_SCALES: tl.constexpr` (lines 50-69, 120-135). One-line pointer only; no separate spec.
  - No CuteDSL alternative for this kernel — Triton is the only NVIDIA path.
- Downstream consumer constraint: outputs feed `fp8_einsum("bhr,hdr->bhd", (o_fp8, o_scale), (wo_a_fp8, wo_a_scale), z, recipe=self._einsum_recipe)` at attention.py:338. Any spec change here must preserve the `o_fp8` shape `[n_groups, T, d]` and the scale's TMA-aligned MN-major layout exactly, or the skipped `transform_sf_into_required_layout` reads garbage.
