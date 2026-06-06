# fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn

## Identity
- Source file: `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:479-667` (Triton JIT body `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn`).
- Launch dispatcher: `compress_norm_rope_store_triton` at `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:31-106` (selects this kernel by `head_dim != 512` AND `use_fp4_cache == True`).
- Language/DSL: **Triton** (`@triton.jit`).
- Third-party dep: none (pure Triton + PyTorch). Uses `_fp32x2_to_fp4x2` helper from `fused_indexer_q.py` for the E2M1×2 byte pack (inline-asm `cvt.rn.satfinite.e2m1x2.f32`).
- Registered as opaque custom op: **no**.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/compressor.py:357` (via dispatcher `compress_norm_rope_store_triton` at `fused_compress_quant_cache.py:64`) | `DeepseekCompressor.forward` for the **indexer's** compressor (`head_dim=128`, `use_fp4_cache=True`) | `state_cache: [num_blocks, block_size, 512] fp32`, `kv_cache: [num_blocks, kv_block_size, 1, TOKEN_STRIDE + SCALE_DIM] = [..., 64 + 4] uint8`, `block_table`, `positions`, `slot_mapping`, `cos_sin_cache` | inputs fp32, outputs MXFP4 packed bytes + UE8M0 byte scales | `head_dim == 128` AND `use_fp4_cache == True` (`attention_config.use_fp4_indexer_cache=True`) |

V4-Flash indexer always uses `compress_ratio=4`.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `state_cache_ptr` | `[num_blocks, block_size, 512]` | fp32 | paged 3-D | Indexer compressor state cache. |
| `token_to_req_indices_ptr` | `[num_tokens]` | int32 | contiguous | Request index per token. |
| `positions_ptr` | `[num_tokens]` | int64 | contiguous | Global positions. |
| `slot_mapping_ptr` | `[num_tokens]` | int64 | contiguous | State-cache slot; `-1` pad. |
| `block_table_ptr` | `[num_reqs, max_blocks]` | int32 | strided | Per-request logical→physical. |
| `block_size` | scalar | int | — | Tokens/block. |
| `rms_norm_weight_ptr` | `[128]` | bf16/fp32 | contiguous | RMSNorm gain. |
| `rms_norm_eps` | scalar | fp32 | — | Epsilon. |
| `cos_sin_cache_ptr` | `[max_pos, 64]` | fp32 | cos/sin halves | Compressor RoPE cache (`compress_rope_theta=160000`). |
| `k_cache_ptr` | `[num_kv_blocks, kv_cache_block_size, 1, TOKEN_STRIDE + SCALE_DIM]` per token | uint8 | per token: `[0, 64) MXFP4 packed bytes`, scale region `[block_size*64 + slot*4, +4)` | Indexer paged K cache (MXFP4 layout; **same physical allocation size as FP8** — see `attention.py:786-787` — but only `64 + 4 = 68` bytes/token are used). |
| `kv_slot_mapping_ptr` | `[num_tokens]` | int64 | contiguous | KV slot; `-1` pad. |
| `kv_cache_block_size` | scalar | int | — | KV tokens/block. |
| `HEAD_SIZE` | constexpr int = 128 | — | — | Indexer head dim. |
| `TRITON_BLOCK_SIZE` | constexpr int = 128 | — | — | `next_power_of_2(128) = 128`. |
| `STATE_WIDTH` | constexpr int = 256 | — | — | `coff * head_dim = 2*128`. |
| `COMPRESS_RATIO` | constexpr int = 4 | — | — | Always 4. |
| `OVERLAP` | constexpr bool = True | — | — | True for `ratio=4`. |
| `ROPE_HEAD_DIM` | constexpr int = 64 | — | — | RoPE tail. |
| `FP8_MAX` | constexpr fp32 = 448.0 | — | — | Unused for MXFP4 (signature parity only — see line 509 comment). |
| `QUANT_BLOCK` | constexpr int = 32 | — | — | **MXFP4 block size** (32 elems/block). |
| `TOKEN_STRIDE` | constexpr int = 64 | — | — | `HEAD_SIZE // 2 = 64` packed bytes/token (two E2M1 nibbles per byte). |
| `SCALE_DIM` | constexpr int = 4 | — | — | `HEAD_SIZE // QUANT_BLOCK = 4` UE8M0 bytes/token (one per 32-elem block). |
| `KV_BLOCK_STRIDE` | constexpr int | — | — | `kv_cache.stride(0)`. |

Static assertions (lines 639-642):
```python
tl.static_assert(TRITON_BLOCK_SIZE == HEAD_SIZE)
tl.static_assert(HEAD_SIZE % QUANT_BLOCK == 0)
tl.static_assert(TOKEN_STRIDE == HEAD_SIZE // 2)
tl.static_assert(SCALE_DIM == HEAD_SIZE // QUANT_BLOCK)
```

## Outputs

In-place writes:

| Region | Shape per token | Dtype | Meaning |
| --- | --- | --- | --- |
| MXFP4 packed values | `[0, TOKEN_STRIDE = 64)` (64 bytes packing 128 E2M1 nibbles) | uint8 | Two E2M1 4-bit values per byte. Low nibble = "even-position" value within block (even idx in `(even, odd)` pair), high nibble = "odd-position" value. Produced by inline-asm `cvt.rn.satfinite.e2m1x2.f32`. |
| UE8M0 block scales | `[block_size * TOKEN_STRIDE + slot*SCALE_DIM, +4)` | uint8 | One UE8M0 byte per 32-elem block (4 bytes/token). Scale = `2^(byte - 127)`; saturated to `byte ∈ [0, 254]` (because `log2_ratio ∈ [-127, 127]`). |

Non-boundary and padded tokens early-exit.

## Grid / Block

- `grid_dim = (num_actual,)` — one CTA per token.
- `block_dim`: `num_warps=1` (dispatcher line 68 — same as the FP8 sibling), 32 threads/CTA.
- Autotune configs: **none**.
- Per-CTA work:
  1. Early-exit on `slot_id < 0` or `(position + 1) % 4 != 0`.
  2. Gather 8-row state-cache window, softmax + weighted-sum compress (identical to the FP8 sibling).
  3. RMSNorm fp32.
  4. Forward GPT-J RoPE on the rope tail. **Keeps the (even, odd) split** instead of `tl.interleave` — MXFP4 pack naturally consumes the (even, odd) pair layout (line 609-611 comment).
  5. bf16 roundtrip on the rotated `(new_even, new_odd)` to match the FP8 K-side / reference numerics.
  6. **MXFP4 quant**: tile each of `new_even`/`new_odd` into `(N_QUANT_BLOCKS=4, HALF_BLOCK=16)`; per-block amax over both halves, `2^ceil(log2(amax/6.0))` (E2M1 max magnitude = 6.0); store packed bytes + 4 UE8M0 scale bytes.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:316-377` (`Compressor.forward`) + `fp4_act_quant` at model.py:370 (used for the indexer's compressor because `rotate=True` enables FP4-style quant per `Indexer.__init__` model.py:398). MXFP4 details mirror DeepGEMM's docs: per-32-element block scale, E2M1 (1 sign + 2 exp + 1 mant) packed two per byte, `max_repr = 6.0`. Compressor RoPE uses `compress_rope_theta=160000`.

```python
# Per-CTA work for token t.
slot_id  = slot_mapping[t]; position = positions[t]
if slot_id < 0 or (position + 1) % 4 != 0: return

# 1. Compress + RMSNorm — identical to the FP8 sibling.
W = 8
start, pos = position - W + 1, position - W + 1 + torch.arange(W)
mask = pos >= 0
blk_no = block_table[token_to_req_indices[t], pos // block_size]
blk_off = pos % block_size
head_off = (torch.arange(W) >= 4).to(int) * 128
sc = state_cache[blk_no, blk_off, head_off + 256 : head_off + 256 + 128]
sc = sc.masked_fill(~mask, -inf).softmax(dim=0)
kv = state_cache[blk_no, blk_off, head_off : head_off + 128]
compressed_kv = (kv * sc).sum(dim=0)                                      # [128] fp32

var = (compressed_kv**2).mean()
normed = compressed_kv * torch.rsqrt(var + rms_norm_eps) * rms_norm_weight # [128]

# 2. Forward GPT-J RoPE on the rope tail, but KEEP even/odd split (no interleave).
even, odd = normed.view(64, 2).unbind(-1)                                  # each [64]
rope_pair_local = torch.arange(64) - 32                                    # NOPE_PAIRS = 32
is_rope_pair = rope_pair_local >= 0
cs_idx = rope_pair_local.clamp_min(0)
compressed_pos = (position // 4) * 4
cos = torch.where(is_rope_pair, cos_sin_cache[compressed_pos, cs_idx], 1.0)
sin = torch.where(is_rope_pair, cos_sin_cache[compressed_pos, 32 + cs_idx], 0.0)
new_even = even * cos - odd * sin                                          # [64]
new_odd  = odd  * cos + even * sin                                         # [64]
new_even = new_even.to(torch.bfloat16).to(torch.float32)                   # bf16 roundtrip
new_odd  = new_odd.to(torch.bfloat16).to(torch.float32)

# 3. MXFP4 quant per 32-elem block (4 blocks for HEAD_SIZE=128).
# Reshape (even, odd) into (N_BLOCKS=4, HALF_BLOCK=16) — each row covers one block's
# (even, odd) pairs, i.e. 32 contiguous output elements.
even_2d = new_even.view(4, 16)                                             # [N_BLOCKS, HALF_BLOCK]
odd_2d  = new_odd.view(4, 16)
amax = torch.maximum(even_2d.abs().amax(-1), odd_2d.abs().amax(-1))        # [4]
amax = amax.clamp_min(6.0 * 2**-126)                                       # MXFP4 subnormal floor
log2_ratio = torch.ceil(torch.log2(amax / 6.0)).clamp(-127, 127)
inv_scale  = torch.exp2(-log2_ratio)
ue8m0      = (log2_ratio + 127).to(torch.uint8)                            # [4]

# Pack two E2M1 nibbles per byte via inline cvt.rn.satfinite.e2m1x2.f32
packed = _fp32x2_to_fp4x2(even_2d * inv_scale.unsqueeze(-1),
                          odd_2d  * inv_scale.unsqueeze(-1))               # [4, 16] uint8
k_cache.values_region[slot] = packed.view(-1)                              # 64 bytes
k_cache.scale_region[slot]  = ue8m0                                        # 4 bytes
```

Notes on fusion / numerics:
- **MXFP4 block scale ceiling**: matches DeepSeek's `kernel.py` exact formula `2^ceil(log2(amax / 6.0))` (model.py source: `fp4_block_size`, and DeepGEMM ref at HuggingFace V4-Pro `kernel.py#L163`). E2M1 max = 6.0.
- **No `tl.interleave`** after RoPE: the FP8 sibling interleaves `(new_even, new_odd)` back to a flat 128-vector before quant; here we keep the split because MXFP4's pack consumes them as separate `x_lo, x_hi` halves (low nibble / high nibble of each output byte).
- **bf16 roundtrip** (line 631-632): both `new_even` and `new_odd` are cast bf16→fp32 before MXFP4 quant. This mirrors the Q-side MXFP4 kernel and ensures K and Q absmax statistics use the same precision floor.
- **Scale layout per token**: 4 UE8M0 bytes laid out contiguously per token in a separate scale region (NOT interleaved with values). This is the same physical layout V3.2 used for FP8 scales (4 bytes), but reinterpreted: V3.2/this-FP8-sibling stores ONE fp32 scale (4 bytes); this MXFP4 path stores FOUR UE8M0 bytes (also 4 bytes total).
- **Same cache slot size as FP8**: `attention.py:786-787` allocates `k_cache_head_dim = 128 + 128/128 * 4 = 132 bytes/token` regardless of `use_fp4_cache`. MXFP4 uses only the first `64 + 4 = 68` bytes; the remaining 64 bytes are wasted. This is intentional so the same cache pool can serve either path without reallocation.

## Config-dependent dispatch

- Activation condition: `head_dim == 128` AND `use_fp4_cache == True`.
- Variants (**Class B** per D3):
  - **Sibling K-side kernel (this kernel's pair under `use_fp4_cache`)**: `_fused_kv_compress_norm_rope_insert_indexer_attn` at `fused_compress_quant_cache.py:302-474` (FP8 K-side cache, spec file `fused_kv_compress_norm_rope_insert_indexer_attn.md`).
  - **`use_fp4_cache` coupling (D3)**: this MXFP4 K-side variant is selected by `use_fp4_cache=True`. The flag couples Q-side AND K-side — selecting MXFP4 here means the Q-side **must** also be MXFP4 (`_fused_indexer_q_rope_mxfp4_kernel`, spec `fused_indexer_q_rope_mxfp4.md`). Cannot mix (DeepGEMM `fp8_fp4_mqa_logits` requires consistent Q/K dtypes — the `(q_values, q_scale)` tuple's `q_scale=None` vs. `int32` discriminates FP8 vs MXFP4 at the kernel level).
  - Coupled Q-side spec: `fused_indexer_q_rope_mxfp4.md`.
- Class: **B** — `use_fp4_cache` parameterizes a 4-variant coupled cluster (this kernel + sibling FP8 K-side + 2 Q-side variants in `fused_indexer_q.py`).
- Downstream consumer: `fp8_fp4_mqa_logits` (prefill, `sparse_attn_indexer.py:233`) and `fp8_fp4_paged_mqa_logits` (decode, `sparse_attn_indexer.py:324`) read this paged K cache via `kv_cache_as_quant_view(..., use_fp4_cache=True)`. At the DeepGEMM boundary the values are tagged `kPackedFP4` (int8 view) and the scales are reinterpreted as `int32` (4 UE8M0 bytes → 1 int32 lane per token).
