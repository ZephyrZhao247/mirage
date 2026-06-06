# fused_indexer_q_rope_mxfp4

## Identity
- Source file: `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:174-282` (Triton JIT body `_fused_indexer_q_rope_mxfp4_kernel`); MXFP4 helpers `_fp32x2_to_fp4x2` (lines 28-44) and `_quantize_mxfp4_pair` (lines 47-66).
- Launch wrapper: `fused_indexer_q_rope_quant` Python function at `vllm/models/deepseek_v4/common/ops/fused_indexer_q.py:284-438` (MXFP4 branch is the `use_fp4 == True` path, lines 331-398).
- Language/DSL: **Triton** (`@triton.jit`).
- Third-party dep: none (pure Triton + PyTorch). Uses inline-asm `cvt.rn.satfinite.e2m1x2.f32` for the E2M1×2 byte pack.
- Registered as opaque custom op: **no**.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/attention.py:841` (via `fused_indexer_q_rope_quant(..., use_fp4=True)` selecting Triton MXFP4 branch when CuteDSL is absent and `use_fp4_kv==True`) | `DeepseekV4Indexer.forward` → `wq_b_and_q_quant` closure | `q: [num_tokens, n_heads=64, head_dim=128] bf16`, `positions: [num_tokens] int64`, `cos_sin_cache: [max_pos, 64] fp32`, `index_weights: [num_tokens, 64] bf16` | inputs bf16, outputs MXFP4-packed Q + UE8M0 block scales + fp32 weights | `use_fp4_kv == True` AND `has_cutedsl() == False` |

Per-block (32-element) UE8M0 scales are kept ALONGSIDE the Q values (NOT folded into weights — contrast with the FP8 sibling).

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `pos_ptr` | `[num_tokens]` | int64 | contiguous | Global positions. |
| `index_q_ptr` | `[num_tokens, n_heads, INDEX_Q_HEAD_DIM]` | bf16 | row-major | Pre-quant Q. |
| `index_q_cos_sin_ptr` | `[max_pos, 64]` | fp32 | cos/sin halves | Compressor RoPE cache (`compress_rope_theta=160000`). |
| `INDEX_Q_HALF_ROT_DIM` | constexpr int = 32 | — | — | `cos_sin_cache.shape[-1] // 2`. |
| `index_q_mxfp4_ptr` | `[num_tokens, n_heads, INDEX_Q_HEAD_DIM // 2 = 64]` | uint8 | row-major | Output MXFP4 packed values (2 E2M1 nibbles/byte). |
| `index_q_scale_ptr` | `[num_tokens, n_heads, INDEX_Q_HEAD_DIM // MXFP4_BLOCK = 4]` | uint8 | row-major | Output UE8M0 block scales (1 byte per 32-elem block). |
| `INDEX_Q_HEAD_DIM` | constexpr int = 128 | — | — | Indexer head dim. |
| `MXFP4_BLOCK` | constexpr int = 32 | — | — | `MXFP4_BLOCK_SIZE` from line 10. |
| `index_weights_ptr` | `[num_tokens, n_heads]` | bf16 (loaded as fp32) | row-major | Raw weights. |
| `index_weights_softmax_scale` | scalar | fp32 | — | `128**-0.5`. |
| `index_weights_head_scale` | scalar | fp32 | — | `64**-0.5 = 0.125`. |
| `index_weights_out_ptr` | `[num_tokens, n_heads]` | fp32 | row-major | Output folded weights. |

Static assertions (lines 207-210):
```python
tl.static_assert(INDEX_Q_NOPE_DIM >= 0)            # NOPE_DIM = HEAD_DIM - 2*HALF_ROT_DIM = 128 - 64 = 64
tl.static_assert(INDEX_Q_NOPE_DIM % MXFP4_BLOCK == 0)
tl.static_assert(INDEX_Q_ROT_DIM % MXFP4_BLOCK == 0)
tl.static_assert(MXFP4_BLOCK % 2 == 0)
```

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `index_q_packed` (in-place via `index_q_mxfp4_ptr`) | `[T, H, 64]` | uint8 | row-major | MXFP4 values, 2 E2M1 nibbles per byte. Low nibble = even-pair value, high nibble = odd-pair value. |
| `index_q_scale` (in-place via `index_q_scale_ptr`) | `[T, H, 4]` | uint8 | row-major | UE8M0 byte per 32-elem block. Wrapper reinterprets as `int32` (4 bytes → 1 int32) and squeezes the last dim before returning. |
| `index_weights_out` (in-place) | `[T, H]` | fp32 | row-major | Folded weights: `index_weights × softmax_scale × head_scale` — **q_scale NOT folded** (contrast with FP8). |

Wrapper return (line 395-398):
```python
return (
    index_q_packed,
    index_q_scale.view(torch.int32).squeeze(-1),    # (T, H, 1) int32 → (T, H)
), index_weights_out
```

Downstream `SparseAttnIndexer.forward_*` passes `(q_packed, q_scale_int32)` as the `q` tuple to DeepGEMM (`sparse_attn_indexer.py:211-216` casts the packed values to `int8` (kPackedFP4 tag), the scales stay int32).

## Grid / Block

- `grid_dim = (num_tokens, n_heads) = (T, 64)`.
- `block_dim`: `num_warps=1` (wrapper line 387), 32 threads/CTA.
- Autotune configs: **none**.
- Per-CTA work:
  1. Load position; compute per-token-per-head base pointers (q in, q_packed out, q_scale out).
  2. **NoPE blocks** (`NUM_NOPE_BLOCKS = NOPE_DIM // MXFP4_BLOCK = 64/32 = 2`): for each block, load `(x_lo = q[base + 2*i], x_hi = q[base + 2*i+1])` as fp32 (no RoPE), call `_quantize_mxfp4_pair` → 16 packed bytes + 1 UE8M0 byte. Iterated via `tl.static_range`.
  3. **RoPE blocks** (`NUM_ROPE_BLOCKS = ROT_DIM // MXFP4_BLOCK = 64/32 = 2`): for each block, fetch the 16 cos/sin lanes corresponding to that block's pair indices, apply GPT-J RoPE on even/odd halves, bf16 roundtrip, quantize via `_quantize_mxfp4_pair`.
  4. Fold weights: `index_weights_out = index_weights * softmax_scale * head_scale` (NO `q_scale`).
- All loops are `tl.static_range` — fully unrolled.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:411-416`:
```python
q = self.wq_b(qr).unflatten(-1, (n_local_heads, head_dim))
apply_rotary_emb(q[..., -rd:], freqs_cis)         # rope on tail
q = rotate_activation(q)                           # Hadamard (skipped in vLLM)
fp4_act_quant(q, fp4_block_size, True)            # ← the MXFP4 step this kernel implements
```
The Hadamard `rotate_activation` is skipped in the vLLM MXFP4 path (same omission as the FP8 sibling). MXFP4 details mirror DeepGEMM's HuggingFace V4-Pro `kernel.py#L163` reference: per-32-element block scale `2^ceil(log2(amax / 6.0))` (E2M1 max = 6.0), packed two E2M1 per byte. Compressor RoPE uses `compress_rope_theta=160000`.

```python
# Per-CTA work for token t, head h (PyTorch-operator equivalent):
pos = positions[t]                                                  # int64

q_full = index_q[t, h, :].to(torch.float32)                         # [128]
nope   = q_full[: 64]                                               # NOPE_DIM = 64
rope   = q_full[64 :]                                               # ROT_DIM = 64

# ---- NoPE blocks: 2 blocks × 32 elems (no RoPE, just MXFP4) ----
nope_blocks_packed = []
nope_blocks_scale  = []
for b in range(2):                                                  # NUM_NOPE_BLOCKS
    x_lo = nope[b*32 + 0::2]                                        # [16]
    x_hi = nope[b*32 + 1::2]                                        # [16]
    packed, ue8m0 = _quantize_mxfp4_pair(x_lo, x_hi)                # 16 bytes, scalar uint8
    nope_blocks_packed.append(packed)                               # store to index_q_packed[t, h, b*16 : (b+1)*16]
    nope_blocks_scale.append(ue8m0)                                 # store to index_q_scale[t, h, b]

# ---- RoPE blocks: 2 blocks × 32 elems with GPT-J RoPE per block ----
half = 16                                                           # HALF_BLOCK = MXFP4_BLOCK // 2
for b in range(2):                                                  # NUM_ROPE_BLOCKS
    pair_off = b*16 + torch.arange(16)                              # indices in [0, HALF_ROT_DIM = 32)
    cos = cos_sin_cache[pos, pair_off].to(torch.float32)            # [16]
    sin = cos_sin_cache[pos, 32 + pair_off].to(torch.float32)       # [16]
    x_even = rope[2*pair_off].to(torch.float32)                     # [16]
    x_odd  = rope[2*pair_off + 1].to(torch.float32)                 # [16]
    r_even = x_even * cos - x_odd * sin
    r_odd  = x_odd  * cos + x_even * sin
    # bf16 roundtrip for parity with FP8 sibling and K-side numerics.
    r_even = r_even.to(torch.bfloat16).to(torch.float32)
    r_odd  = r_odd.to(torch.bfloat16).to(torch.float32)
    packed, ue8m0 = _quantize_mxfp4_pair(r_even, r_odd)
    rope_byte_off = (64 + b*32) // 2                                # = 32 + b*16
    # store to index_q_packed[t, h, rope_byte_off : rope_byte_off + 16]
    # store to index_q_scale[t, h, 2 + b]

# ---- Per-block MXFP4 quant (helper at fused_indexer_q.py:47-66) ----
def _quantize_mxfp4_pair(x_lo, x_hi):
    amax = max(x_lo.abs().amax(), x_hi.abs().amax())
    amax = max(amax, 6.0 * 2**-126)                                 # MXFP4 subnormal floor
    log2_ratio = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
    scale = torch.exp2(log2_ratio)
    ue8m0 = (log2_ratio + 127).to(torch.uint8)                      # block scale byte
    inv_scale = 1.0 / scale
    packed = _fp32x2_to_fp4x2(x_lo * inv_scale, x_hi * inv_scale)   # [16] uint8 via inline-asm cvt.rn.satfinite.e2m1x2.f32
    return packed, ue8m0

# ---- Weights: per-token-per-head q_scale NOT folded (it's per-block now). ----
w = index_weights[t, h].to(torch.float32)
index_weights_out[t, h] = w * softmax_scale * head_scale            # NO q_scale multiplier
```

Notes on fusion / numerics:
- **Weight-fold contrast with FP8**: FP8 emits ONE scalar q_scale per (token, head) and folds it into weights; MXFP4 emits FOUR UE8M0 bytes per (token, head) (one per 32-elem block) and stores them with the values. Per-block scales can't be folded into a per-token weight scalar, so weights here only carry `softmax_scale × head_scale` (line 266-272 comment).
- **NoPE / RoPE block boundaries**: with `NOPE_DIM=64` and `ROT_DIM=64`, the `MXFP4_BLOCK=32` partition gives exactly 2 NoPE blocks (indices 0-31, 32-63) and 2 RoPE blocks (indices 64-95, 96-127). The block boundary is intentionally aligned to the NoPE/RoPE split — no MXFP4 block straddles them.
- **(even, odd) pair layout within a block**: the helper takes `x_lo, x_hi` = the values at even / odd indices within the block. For NoPE the kernel loads `q[base + 2*i]` and `q[base + 2*i+1]`. For RoPE, after rotation, `(r_even, r_odd)` ARE the (even-index, odd-index) values of the block — so the same pair convention naturally applies.
- **bf16 roundtrip** (line 258-260): matches K-side compressor MXFP4 numerics.
- **DeepGEMM tuple convention**: wrapper returns `(packed_uint8, scales_int32)`. Reinterpreting the 4 UE8M0 bytes as one int32 lets DeepGEMM's `tensor_map_sf_q` consume them as `int32` block scales (`smxx_fp8_fp4_paged_mqa_logits.cuh:233` comment cited in `v1/attention/backends/mla/indexer.py:274`).

## Config-dependent dispatch

- Activation condition: `use_fp4_kv == True` AND `has_cutedsl() == False`.
  - On NVIDIA when `has_cutedsl()` is True (B200 fast path), wrapper dispatches to `fused_indexer_q_rope_quant_mxfp4_cutedsl` (`IndexerQMxFp4Kernel`) instead — wrapper lines 347-363.
- Variants (**Class B** per D3 — FP8 vs MXFP4; D2 — Triton vs CuteDSL):
  - **FP8 Q-side sibling (this kernel's pair under `use_fp4_cache`)**: `_fused_indexer_q_rope_quant_kernel` at `fused_indexer_q.py:69-172` (spec `fused_indexer_q_rope_quant.md`).
  - **`use_fp4_cache` coupling (D3)**: this MXFP4 Q-side variant is selected when `use_fp4_kv = True`. The flag couples Q-side AND K-side — selecting MXFP4 Q here means the K-side **must** also be MXFP4 (`_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn`, spec `fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn.md`). Cannot mix: DeepGEMM `fp8_fp4_mqa_logits` discriminates via the `(q_values, q_scale)` tuple where `q_scale=int32` selects MXFP4 — and the K dtype must match.
  - Coupled K-side spec: `fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn.md`.
  - **CuteDSL sibling (locked active path on NVIDIA, D2)**: `IndexerQMxFp4Kernel` in `vllm/models/deepseek_v4/nvidia/ops/fused_indexer_q_cutedsl.py:253-425` — locked-alternative pointer; identical I/O contract (returns `(packed, scales_int32)` tuple + folded weights).
- Class: **B** — part of a 4-variant coupled cluster (`use_fp4_cache`) × Triton vs CuteDSL.
- Downstream consumer: `SparseAttnIndexer.forward_*` passes `((q_packed, q_scale_int32), index_weights_out)` into `fp8_fp4_mqa_logits` / `fp8_fp4_paged_mqa_logits`. The per-block scales are loaded as `tensor_map_sf_q` (UMMA block-scaled MMA descriptor); per-block dequant happens inside the DeepGEMM MMA — not in this kernel.
