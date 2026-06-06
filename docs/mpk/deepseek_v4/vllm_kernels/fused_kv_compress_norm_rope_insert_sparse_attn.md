# fused_kv_compress_norm_rope_insert_sparse_attn

## Identity
- Source file: `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:112-297` (Triton JIT body `_fused_kv_compress_norm_rope_insert_sparse_attn`).
- Launch dispatcher: `compress_norm_rope_store_triton` at `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:31-106` (selects this kernel by `head_dim == 512`).
- Language/DSL: **Triton** (`@triton.jit`).
- Third-party dep: none (pure Triton + PyTorch).
- Registered as opaque custom op: **no** — invoked via the dispatcher.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/compressor.py:357` (via dispatcher `compress_norm_rope_store_triton` at `fused_compress_quant_cache.py:61` selecting this kernel because `head_dim==512`) | `DeepseekCompressor.forward` (attention compressor, `head_dim=512`) | `state_cache: [num_blocks, block_size, 2048] fp32`, `kv_cache: [num_blocks, kv_block_size, 1, 584] uint8`, `block_table: [num_reqs, max_blocks] int32`, `positions: [num_tokens] int64`, `slot_mapping`, `kv_slot_mapping`, `cos_sin_cache: [max_pos, 64] fp32` | inputs fp32 (cache), outputs fp8e4m3 + bf16 (paged KV cache) | `head_dim == 512` AND ROCm OR (CUDA path that didn't trigger CuteDSL fast path); see Class B siblings below |

V4-Flash on B200 CUDA: the *triton* attention-compressor path is the locked-alternative — CuteDSL is the active path on NVIDIA (`compressor.py:340-347`). This Triton kernel runs on ROCm and as a fallback. Specced here per D2 because both variants are in scope; CuteDSL siblings are listed under Config-dependent dispatch.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `state_cache_ptr` | `[num_blocks, block_size, 2 * coff * head_dim = 2048]` (for `coff=2, head_dim=512`) | fp32 | paged 3-D; last dim = `[kv_state \| score_state]` | Compressor state cache previously populated by `save_partial_states`. |
| `token_to_req_indices_ptr` | `[num_tokens]` | int32 | contiguous | Per-token request index used to look up its row in `block_table`. |
| `positions_ptr` | `[num_tokens]` | int64 | contiguous | Global token positions. |
| `slot_mapping_ptr` | `[num_tokens]` | int64 | contiguous | State-cache slot per token; `-1` = pad. |
| `block_table_ptr` | `[num_reqs, max_blocks]` | int32 | strided | Maps per-request logical block index → physical state-cache block. |
| `block_size` | scalar | int | — | State-cache block size (tokens/block). |
| `rms_norm_weight_ptr` | `[HEAD_SIZE = 512]` | bf16/fp32 | contiguous | RMSNorm gain. Loaded element-wise. |
| `rms_norm_eps` | scalar | fp32 | — | RMSNorm epsilon. |
| `cos_sin_cache_ptr` | `[max_pos, rope_head_dim = 64]` | fp32 | per row: first half cos, second half sin (each `rope_head_dim/2 = 32` lanes) | Precomputed RoPE cache **using `compress_rope_theta=160000`** (the Compressor's RoPE — DIFFERENT base from main RoPE's `rope_theta=10000`; see Math). Built by the Compressor's own RoPE wrapper. |
| `k_cache_ptr` | `[num_kv_blocks, kv_cache_block_size, 1, TOKEN_STRIDE + SCALE_DIM] = [..., 576 + 8]` per token in scale region for V4 | uint8 (FP8 + bf16 + UE8M0 scales overlaid) | paged 4-D; per token: `[0,448) FP8 nope`, `[448,576) bf16 rope (64 elems × 2 bytes)`, then per-block UE8M0 scale region at `[block_size * 576, ...)` | Paged KV cache (the Compressor's K cache, written here). |
| `kv_slot_mapping_ptr` | `[num_tokens]` | int64 | contiguous | KV-cache slot per token; `-1` = pad. |
| `kv_cache_block_size` | scalar | int | — | Tokens per KV-cache block. |
| `HEAD_SIZE` | constexpr int = 512 | — | — | Compressor head dim. |
| `TRITON_BLOCK_SIZE` | constexpr int = 512 | — | — | `next_power_of_2(512) = 512`. |
| `STATE_WIDTH` | constexpr int = 1024 | — | — | `coff * head_dim = 2*512`. Width of kv-half of state slot. |
| `COMPRESS_RATIO` | constexpr int | — | — | 4 (overlap) or 128 (non-overlap). |
| `OVERLAP` | constexpr bool | — | — | `compress_ratio == 4`. Sets gather window to `2*COMPRESS_RATIO` tokens. |
| `ROPE_HEAD_DIM` | constexpr int = 64 | — | — | RoPE dims at the tail of head. |
| `FP8_MAX` | constexpr fp32 = 448.0 | — | — | e4m3 max magnitude. |
| `QUANT_BLOCK` | constexpr int = 64 | — | — | UE8M0 block size for the NoPE region. |
| `TOKEN_STRIDE` | constexpr int = 576 | — | — | `nope_head_dim + 2 * rope_head_dim = 448 + 2*64`. Per-token byte stride in the paged cache (FP8 nope + bf16 rope). |
| `SCALE_DIM` | constexpr int = 8 | — | — | UE8M0 scale bytes per token (`nope_head_dim/QUANT_BLOCK + 1 pad = 7 + 1`). |
| `KV_BLOCK_STRIDE` | constexpr int | — | — | `kv_cache.stride(0)` byte stride between KV-cache blocks. |

## Outputs

All writes are in-place into `k_cache_ptr`:

| Region | Shape per token | Dtype | Meaning |
| --- | --- | --- | --- |
| FP8 NoPE | `[0, 448)` (= `NOPE_HEAD_DIM = HEAD_SIZE - ROPE_HEAD_DIM`) | `float8_e4m3fn` (stored as uint8) | UE8M0 block-FP8 quant of the post-RMSNorm compressed KV's NoPE portion. 7 quant blocks × 64 elems. |
| bf16 RoPE | `[448, 576)` (64 elements × 2 bytes) | bf16 | Forward-RoPE-rotated rope portion of the compressed KV. |
| UE8M0 scales | per-block `[block_size * TOKEN_STRIDE + slot_in_block * 8, +8)` | uint8 | 7 UE8M0 exponent bytes (one per NoPE quant block) + 1 zero-pad byte (lane 7). Scale encoding: `byte = clamp(exponent + 127, 0, 255)`. |

CTAs whose `(position + 1) % COMPRESS_RATIO != 0` (non-boundary tokens) early-exit without writing.

## Grid / Block

- `grid_dim = (num_actual,)` — one CTA per token.
- `block_dim`: `num_warps=4` (set at dispatcher line 62), so 128 threads/CTA.
- Autotune configs: **none** — `num_warps=4` hard-coded.
- Per-CTA work:
  1. Early-exit on `slot_id < 0` or `(position + 1) % COMPRESS_RATIO != 0`.
  2. Gather `(1 + OVERLAP) * COMPRESS_RATIO` rows from the state cache (the boundary window): each row indexes through `block_table` for the request, with `head_offset` toggled between `0` and `HEAD_SIZE` to pick the "overlap" half vs the "current" half of the slot (see Math).
  3. Softmax across the window on the score-half, weighted sum on the kv-half → `compressed_kv [HEAD_SIZE]` fp32.
  4. RMSNorm fp32 with `rms_norm_weight`.
  5. UE8M0 block-FP8 quant of NoPE portion (`N_NOPE_BLOCKS = 7` blocks of 64 elems) via `tl.reshape((8, 64))` → per-block absmax → `2^ceil(log2(amax / FP8_MAX))` → divide-and-clamp → cast to fp8e4m3, written as uint8.
  6. Forward GPT-J RoPE on the rope portion using `cos_sin_cache[(position // COMPRESS_RATIO) * COMPRESS_RATIO]` (rotated position = window-aligned), stored as bf16.
- All compute is in fp32; cast to bf16→fp32 before the UE8M0 absmax pass to match the reference numerics (`fused_compress_quant_cache.py:242`).
- Int64 index: `block_numbers` cast to i64 before stride multiplication to avoid overflow.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:316-377` (`Compressor.forward`; the `should_compress` boundary path lines 343-377). RoPE comes from line 367 `apply_rotary_emb(kv[..., -rd:], freqs_cis)` with `freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio]` (line 366) — note this is the *window-aligned* position, mirroring vLLM's `(position // COMPRESS_RATIO) * COMPRESS_RATIO`. The Compressor's RoPE cache is built from `compress_rope_theta=160000` (model.py:67 default; V4-Flash-Base config sets it to 160000) which differs from the main attention's `rope_theta=10000`. Block-FP8 quant uses `act_quant(..., 64, scale_fmt, ...)` (model.py:372).

```python
# Per-CTA work for token t (PyTorch-operator equivalent for the active branch):
slot_id  = slot_mapping[t]
position = positions[t]
if slot_id < 0 or (position + 1) % COMPRESS_RATIO != 0:
    return                              # non-boundary, nothing to compress

req_idx = token_to_req_indices[t]

# 1. Gather (1+OVERLAP)*COMPRESS_RATIO rows from the state cache spanning the
#    compression window.  When OVERLAP=True (ratio=4) the window is 2*ratio
#    long; tokens in the first half pick the "overlap" slot (head_offset = 0),
#    tokens in the second half pick the "current" slot (head_offset = HEAD_SIZE).
W = (1 + OVERLAP) * COMPRESS_RATIO
start = position - W + 1
pos   = start + torch.arange(W)
mask  = pos >= 0
blk_idx = pos // block_size; blk_off = pos % block_size
blk_no  = block_table[req_idx, blk_idx]                          # masked load
head_off = (torch.arange(W) >= COMPRESS_RATIO).to(int) * HEAD_SIZE

row_base = state_cache[blk_no, blk_off, head_off : head_off+HEAD_SIZE]
# score (kv) lives at row_base + STATE_WIDTH (resp. row_base) — STATE_WIDTH == 2*HEAD_SIZE here.

# 2. Softmax across the W rows, weighted sum.
score = state_cache[blk_no, blk_off, head_off + STATE_WIDTH : head_off + STATE_WIDTH + HEAD_SIZE]
score = score.masked_fill(~mask, -inf).softmax(dim=0)            # [W, HEAD_SIZE]
kv    = state_cache[blk_no, blk_off, head_off : head_off + HEAD_SIZE]
compressed_kv = (kv * score).sum(dim=0)                          # [HEAD_SIZE] fp32

# 3. RMSNorm (fp32).
var   = (compressed_kv ** 2).mean()
normed = compressed_kv * torch.rsqrt(var + rms_norm_eps) * rms_norm_weight    # [HEAD_SIZE]

# 4. UE8M0 block-FP8 quant on the NoPE half (head_dim=448, blocks of 64).
nope = normed[:NOPE_HEAD_DIM]                                    # 448 = HEAD_SIZE - ROPE_HEAD_DIM
# bf16 roundtrip to match reference
nope = nope.to(torch.bfloat16).to(torch.float32)
chunks = nope.view(N_NOPE_BLOCKS, QUANT_BLOCK)                   # 7 × 64
absmax = chunks.abs().amax(dim=-1).clamp_min(1e-4)
exponents = torch.ceil(torch.log2(absmax / FP8_MAX))             # UE8M0 exponent (fp32)
inv_scale = torch.exp2(-exponents)
fp8 = (chunks * inv_scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
k_cache.fp8_region[slot] = fp8.view(-1)                          # 448 bytes
k_cache.scale_region[slot, :7] = (exponents.clamp(-127, 128) + 127).to(torch.uint8)
k_cache.scale_region[slot, 7]  = 0                               # pad byte

# 5. Forward GPT-J RoPE on the rope tail.  Position is window-aligned.
rope = normed[NOPE_HEAD_DIM:]                                    # 64 fp32
even, odd = rope.view(rope_pairs, 2).unbind(-1)                  # each [32]
compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
cos = cos_sin_cache[compressed_pos, : rope_dim // 2]             # [32], rope_theta=compress_rope_theta=160000
sin = cos_sin_cache[compressed_pos, rope_dim // 2 :]
new_even = even * cos - odd * sin
new_odd  = odd  * cos + even * sin
rotated  = torch.stack([new_even, new_odd], dim=-1).reshape(-1)  # [64] fp32
k_cache.bf16_region[slot] = rotated.to(torch.bfloat16)           # 128 bytes
```

Notes on fusion / numerics:
- The reference does a single `softmax(dim=2)` over the window; here, the gather index `tokens >= COMPRESS_RATIO` selects between "overlap" and "current" halves of each state slot. That's the same as `overlap_transform` (model.py:307-314) — overlap tokens get the previous-window's contribution, current tokens get this-window's.
- UE8M0 scale encoding: `byte = clamp(exponent + 127, 0, 255)`; pad byte 7 is zero. This matches V4's `act_quant(..., scale_fmt="ue8m0")` (model.py:372).
- **Compressor-RoPE-only:** `cos_sin_cache` here is the Compressor's own cache, built from `compress_rope_theta=160000` (V4-Flash-Base config), NOT the main attention RoPE's `rope_theta=10000`. The compressor passes its OWN `rotary_emb.cos_sin_cache` into this kernel (`compressor.py:334`). Mixing the two breaks indexer scoring.
- The bf16 RoPE store uses pointer reinterpretation `(fp8_ptr + NOPE_HEAD_DIM).to(tl.pointer_type(tl.bfloat16))` — the 64 RoPE elements occupy 128 bytes immediately after the 448 FP8 bytes within the same per-token slot.

## Config-dependent dispatch

- Activation condition: `head_dim == 512` AND triton path is selected.
  - On NVIDIA CUDA: the dispatcher in `compressor.py:340-347` calls `compress_norm_rope_store_cutedsl` instead (CuteDSL fast path). This Triton kernel is the **locked-alternative** fallback (Class B per D2; CuteDSL is the locked active path on B200).
  - On AMD: this kernel is the only path.
- Variants (Class B per D2 — both variants are in scope; this spec covers the Triton variant only):
  - **CuteDSL sibling — locked active path on NVIDIA**: `compress_norm_rope_store_cutedsl` at `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py:1244-1333`. Internally splits into two cases:
    - `compress_ratio == 4`: `SparseAttnCompressNormRopeStoreC4Kernel` (single fused kernel) at `sparse_attn_compress_cutedsl.py:75-460`.
    - `compress_ratio == 128`: `SparseAttnCompressKernel` (lines 463-815) + `SparseAttnNormRopeStoreKernel` (lines 818-1087), launched back-to-back.
  - This Triton kernel handles BOTH `compress_ratio` values via the `OVERLAP` constexpr (one variant for `ratio=4 → OVERLAP=True`, one for `ratio=128 → OVERLAP=False`).
- Class: **B** — Triton ↔ CuteDSL pair.
- Cache layout (per-token paged-K-cache stride 576 bytes data + 8 bytes UE8M0 scale): must match BOTH variants. If MPK replaces one, the other still consumes/produces the same slot layout.
- Downstream consumer: this writes the Compressor's K cache, read by `flash_mla_sparse_fwd` (sparse-attention prefill) and `flash_mla_with_kvcache` (decode) via the `SparseAttnIndexer` path. The TOKEN_STRIDE / SCALE_DIM choice is hard-coded across kernel + cache allocation in `compressor.py:253-255`.
