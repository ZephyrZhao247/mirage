# apply_rotary_emb

## Identity
- Source file (Triton kernel body): `vllm/vllm_flash_attn/ops/triton/rotary.py:12-131` (`@triton.jit rotary_kernel`).
- Source file (Python launch wrapper): `vllm/vllm_flash_attn/ops/triton/rotary.py:134-229` (`def apply_rotary(...)`).
- Source file (autograd / inplace wrapper): `vllm/vllm_flash_attn/layers/rotary.py:38-126` (`class ApplyRotaryEmb(torch.autograd.Function)` + `def apply_rotary_emb(...)`).
- Source file (vLLM CustomOp wrapper): `vllm/model_executor/layers/rotary_embedding/common.py:122-290` (`class ApplyRotaryEmb(CustomOp)`); `forward_cuda` at lines 227-248 imports and dispatches to the flash-attn wrapper.
- Language/DSL: **Triton** (`@triton.jit`), copied from Tri Dao's flash-attn (file header comment line 1).
- Third-party dep: `einops` (used in the `apply_rotary_emb_torch` reference but not in the Triton path); `triton`.
- Registered as: Python `CustomOp` (`@CustomOp.register("apply_rotary_emb")`, common.py:123). NOT a `torch.ops` op.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| (none reachable from V4-Flash NVIDIA in production) | — | — | — | — |

**Reachability analysis for V4-Flash NVIDIA**:

- V4 instantiates `DeepseekV4ScalingRotaryEmbedding` via `get_rope(...)` at `vllm/models/deepseek_v4/nvidia/model.py:715`. Its constructor (super: `RotaryEmbeddingBase.__init__` at `vllm/model_executor/layers/rotary_embedding/base.py:76`) creates an `ApplyRotaryEmb` instance, **but** `DeepseekV4ScalingRotaryEmbedding.forward_cuda` (`vllm/model_executor/layers/rotary_embedding/deepseek_scaling_rope.py:318-345`) overrides the parent and dispatches to `torch.ops._C.rotary_embedding` (CUDA op), NOT the Triton kernel.
- The standard RoPE on Q/KV in the MLA forward path is folded into `torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` (`vllm/models/deepseek_v4/attention.py:531-541`); RoPE on the inverse-O path is the Triton `fused_inv_rope_fp8_quant` (different kernel — see `fused_inv_rope_fp8_quant.md`); RoPE in the compressor and indexer is folded into `compress_norm_rope_store_*` (CuteDSL/Triton) and `fused_indexer_q_rope_quant_*` (Triton/CuteDSL).
- **Conclusion**: on V4-Flash NVIDIA the flash-attn Triton `apply_rotary_emb` is *not* invoked in production. The Triton kernel and `ApplyRotaryEmb` CustomOp wrapper remain in the call graph as the *generic fallback* — any RoPE that flows through `RotaryEmbedding.forward_static` (`base.py:160-201`, e.g. `forward_native`) routes through `ApplyRotaryEmb.forward_static` (lines 180, 194), which is pure PyTorch. The Triton kernel is only reached via `ApplyRotaryEmb.forward_cuda` (common.py:227-248) when the parent `RotaryEmbedding` class's own `forward_cuda` is bypassed. The V4 wrapper short-circuits that.

This spec documents the kernel completeness for any future MPK path that wants generic (non-V4-specific) bf16 RoPE before quantization.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `x` | `[batch, seqlen, nheads, headdim]` (dense) or `[total_seqlen, nheads, headdim]` (varlen, when `cu_seqlens != None`) | bf16 / fp16 / fp32 (must match `cos`/`sin`; asserted at `rotary.py:176-178`) | row-major; arbitrary batch/seqlen/heads strides — passed as runtime ints to the kernel | Input tensor; first `rotary_dim` lanes of `headdim` are rotated, trailing `headdim - rotary_dim` lanes are copied through unchanged (rotary.py:189-190 copies the tail to `output` when not in-place). |
| `cos` | `[seqlen_ro, rotary_dim // 2]` | matches `x.dtype` (asserted line 173-175) | contiguous (made contiguous on line 180) | Precomputed cosines (one value per rotary pair, length `rotary_dim/2`). |
| `sin` | `[seqlen_ro, rotary_dim // 2]` | same as `cos` | contiguous | Precomputed sines. |
| `seqlen_offsets` | scalar int OR `[batch]` int32/int64 | int | scalar literal (compile-time path) OR contiguous tensor (per-batch dynamic offset) | Position offset added to row index when indexing into `cos`/`sin`. Used in KV-cache decode to shift logical positions. Compile-time vs runtime selected by `IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr` (kernel line 35, 58-61). |
| `cu_seqlens` | `[batch + 1]` or `None` | int32/int64 | contiguous | Cumulative sequence lengths for variable-length packed inputs. When `None`, the dense path uses `batch * stride_x_batch` indexing; when set, the varlen path uses `start_idx = cu_seqlens[pid_batch]` and per-batch `seqlen = cu_seqlens[pid_batch+1] - start_idx` (kernel lines 46-53). Selected by `IS_VARLEN: tl.constexpr`. |
| `max_seqlen` | int or `None` | — | — | Required when `cu_seqlens` is set; sets the grid_x extent (rotary.py:165). |
| `interleaved` | bool | — | compile-time | `False` → split-half (Neox / GPT-NeoX style: first half pairs with second half); `True` → interleaved (GPT-J style: even index pairs with odd, kernel lines 96-131). |
| `inplace` | bool | — | — | If True, kernel writes back into `x` (`output = x`, line 188); else allocates `torch.empty_like(x)`. |
| `conjugate` | bool | — | compile-time | If True, kernel computes the inverse rotation by negating `sin` (line 85 / 125: `sin = -sin`). Used by the autograd backward and by callers needing de-rotation. |

Constexprs baked at JIT:
- `BLOCK_K: tl.constexpr` — power-of-two ceiling of `rotary_dim`, picked from `{32, 64, 128, 256}` at rotary.py:192-196.
- `BLOCK_M: tl.constexpr` — `4` if interleaved else `8` (for `rotary_dim ≤ 128`) else `4` (line 198).
- `INTERLEAVED`, `CONJUGATE`, `IS_VARLEN`, `IS_SEQLEN_OFFSETS_TENSOR` — all `tl.constexpr` bools.

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `output` | matches `x` exactly | same as `x.dtype` | `torch.empty_like(x)` (or alias `x` if `inplace=True`) | Rotary-rotated input. Trailing `headdim - rotary_dim` lanes are bit-copied from `x` by the host wrapper (rotary.py:189-190) *before* the kernel launches; the kernel itself writes only the leading `rotary_dim` lanes. |

`headdim` must satisfy `rotary_dim ≤ headdim ≤ 256` (rotary.py:169-170).

## Grid / Block

- `grid_dim = (cdiv(seqlen, BLOCK_M), nheads, batch)` (rotary.py:197). Grid-x is "M tiles of `BLOCK_M` tokens", grid-y is per-head (no sharing across heads — each head's cos/sin pointer is the same but offsets differ), grid-z is batch.
- `block_dim` (threads/CTA): driven by `num_warps = 2 if rotary_dim <= 64 else 4` (line 227). With Triton's 32-thread warps that's 64 or 128 threads/CTA.
- `num_stages`: default Triton (no explicit override).
- Autotune configs: **none** — `BLOCK_K`, `BLOCK_M`, and `num_warps` are picked deterministically from `rotary_dim` in Python (lines 192-198, 227).
- Per-CTA work: a single CTA loads a `[BLOCK_M, BLOCK_K]` tile (`BLOCK_M` tokens × `BLOCK_K` head-dim lanes). Two paths:
  - **Non-interleaved (Neox)** (kernel lines 65-95): load `x0 = x[:, :rotary_dim/2]`, `x1 = x[:, rotary_dim/2:rotary_dim]`, compute `o0 = x0*cos - x1*sin`, `o1 = x0*sin + x1*cos`, write to the two halves. Cos/sin tile is `[BLOCK_M, BLOCK_K/2]`.
  - **Interleaved (GPT-J)** (lines 96-131): load `x0 = x[:, 0,1,2,...]` (fast contiguous), `x1 = x[:, 1,0,3,2,...]` (strided via `rk_swap`), with `cos = COS[..., 0,0,1,1,...]` and `sin = SIN[..., 0,0,1,1,...]` (broadcast via `rk_repeat`). Computes `out = where(even, x0*cos - x1*sin, x0*cos + x1*sin)`, writes one combined tile.
- Mask handling: separate masks on input rows (`rm < seqlen`) and cos/sin rows (`rm_cs < seqlen_ro`). Out-of-bound rows load `0.0` for `x`/`sin` and `1.0` for `cos` (identity rotation) — see lines 70-83.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:232-242` (`def apply_rotary_emb`). Reference uses complex-number multiplication on `view_as_complex(x.float().unflatten(-1, (-1, 2)))` with `freqs_cis = exp(i*θ)`; `inverse=True` conjugates `freqs_cis`. The Triton kernel materializes the same operation as real-valued cos/sin pair-multiply.

For Neox-style (V4-Flash uses `is_neox_style=False`, i.e. **interleaved/GPT-J style**; see `vllm/models/deepseek_v4/nvidia/model.py:719`), the kernel's interleaved branch is the active one.

```python
# PyTorch-operator equivalent of one CTA's work for token row m, head h, batch b:
#
# Inputs:
#   x:           [B, T, H, D]  bf16     (D = headdim; first rotary_dim lanes rotated)
#   cos, sin:    [seqlen_ro, rotary_dim/2]  bf16
#   seqlen_offsets: int  (or [B] tensor)
#
m  = pid_m * BLOCK_M + arange(BLOCK_M)            # token rows in this tile
cs_row = m + seqlen_offsets                       # cos/sin row (scalar add or per-batch)

if not interleaved:                               # Neox (split-half)
    x0 = x[b, m, h, : rotary_dim // 2].to(float32)
    x1 = x[b, m, h, rotary_dim // 2 : rotary_dim].to(float32)
    c  = cos[cs_row, : rotary_dim // 2].to(float32)
    s  = sin[cs_row, : rotary_dim // 2].to(float32)
    if conjugate: s = -s
    o0 = x0 * c - x1 * s
    o1 = x0 * s + x1 * c
    out[b, m, h, : rotary_dim // 2]            = o0
    out[b, m, h, rotary_dim // 2 : rotary_dim] = o1
else:                                             # GPT-J (interleaved, V4-Flash style)
    # x_pairs has shape [BLOCK_M, rotary_dim/2, 2]; even = pairs[..., 0], odd = pairs[..., 1]
    x_pairs = x[b, m, h, :rotary_dim].view(BLOCK_M, rotary_dim // 2, 2).to(float32)
    even, odd = x_pairs[..., 0], x_pairs[..., 1]
    c = cos[cs_row].to(float32)            # [BLOCK_M, rotary_dim/2]
    s = sin[cs_row].to(float32)
    if conjugate: s = -s
    out_even = even * c - odd * s
    out_odd  = even * s + odd * c
    out[b, m, h, :rotary_dim] = torch.stack([out_even, out_odd], dim=-1).flatten(-2)

# Trailing lanes (rotary_dim:headdim) are pre-copied by the wrapper (rotary.py:189-190)
# when not inplace; the kernel does not touch them.
```

Notes on fusion / quant:
- **No quantization**: this is a bf16-in, bf16-out RoPE kernel. The V4 NVIDIA path needs FP8 quant fused with RoPE, which is why V4 uses `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` and `fused_inv_rope_fp8_quant` instead — those fuse RoPE into the quant path with no intermediate bf16 spill.
- **No norm**: this is a standalone RoPE kernel. V4's compressor / indexer path fuses RoPE with the upstream RMSNorm.
- **Conjugate trick**: `conjugate=True` is mathematically equivalent to `apply_rotary_emb(..., inverse=True)` in the reference (`model.py:236-237`) — both negate `sin` so the rotation matrix is transposed. Used for the backward pass and for inverse-RoPE on attention output (though V4 prefers `fused_inv_rope_fp8_quant` over a separate `apply_rotary` + quant).

## Config-dependent dispatch

- Activation condition: NOT in V4-Flash NVIDIA's production call graph (see Call sites above). All RoPE sites are fused. The kernel remains the generic vLLM fallback used by `RotaryEmbedding.forward_cuda` → `ApplyRotaryEmb.forward_cuda` (common.py:233-248) for other models.
- Variants on NVIDIA:
  - `interleaved=True` (GPT-J / V4-Flash style) vs `interleaved=False` (Neox / standard Llama style) — selected by `INTERLEAVED: tl.constexpr`; both branches live in the same kernel.
  - `conjugate=True/False` — `CONJUGATE: tl.constexpr`, both in same kernel.
  - `IS_VARLEN`, `IS_SEQLEN_OFFSETS_TENSOR` — `tl.constexpr` toggles.
  - All four constexpr bools combine into 16 JIT specializations on first use; not autotuned.
- ROCm path: `ApplyRotaryEmb.forward_hip` (common.py:250-276) calls `flash_attn.ops.triton.rotary.apply_rotary` from the external `flash_attn` package (NOT the vllm in-tree copy). The kernel body is the same Tri-Dao reference; differences are pure plumbing.
- No SM90/SM100 split — the kernel is generic Triton.
- Hard preconditions (asserts in `apply_rotary`):
  - `cos.shape == sin.shape` (line 167).
  - `2 * cos.shape[-1] == rotary_dim ≤ headdim` (lines 168-169).
  - `headdim ≤ 256` (line 170).
  - `seqlen_ro ≥ seqlen` (line 171).
  - `x.dtype == cos.dtype == sin.dtype` (lines 173-178).
  - `seqlen_offsets.dtype ∈ {int32, int64}` if it's a tensor (line 183).
