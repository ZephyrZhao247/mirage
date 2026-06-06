# fused_kv_compress_norm_rope_insert_indexer_attn

## Identity
- Source file: `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:302-474` (Triton JIT body `_fused_kv_compress_norm_rope_insert_indexer_attn`).
- Launch dispatcher: `compress_norm_rope_store_triton` at `vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py:31-106` (selects this kernel by `head_dim != 512` AND `use_fp4_cache == False`).
- Language/DSL: **Triton** (`@triton.jit`).
- Third-party dep: none (pure Triton + PyTorch).
- Registered as opaque custom op: **no**.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/compressor.py:357` (via dispatcher `compress_norm_rope_store_triton` at `fused_compress_quant_cache.py:67`) | `DeepseekCompressor.forward` for the **indexer's** compressor (`head_dim=128`, `use_fp4_cache=False`) | `state_cache: [num_blocks, block_size, 512] fp32` (`coff=2, head_dim=128`), `kv_cache: [num_blocks, kv_block_size, 1, 132] uint8` (128 FP8 bytes + 4 fp32 scale bytes per token), `block_table`, `positions`, `slot_mapping`, `cos_sin_cache: [max_pos, 64] fp32` | inputs fp32, outputs fp8e4m3 + fp32 scale (paged indexer K cache) | `head_dim == 128` AND `use_fp4_cache == False` (i.e., `attention_config.use_fp4_indexer_cache=False`); compressor is the Indexer's compressor (`DeepseekV4Indexer.compressor`, attention.py:795-804) |

V4-Flash indexer always uses `compress_ratio=4` (`attention.py:467-471` of `model.py`; vLLM sets it via `DeepseekV4Indexer.__init__`).

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `state_cache_ptr` | `[num_blocks, block_size, 2 * coff * head_dim = 512]` | fp32 | paged 3-D; `[kv_state \| score_state]` | Indexer compressor state cache (`coff=2`, `head_dim=128`). |
| `token_to_req_indices_ptr` | `[num_tokens]` | int32 | contiguous | Request index per token. |
| `positions_ptr` | `[num_tokens]` | int64 | contiguous | Global positions. |
| `slot_mapping_ptr` | `[num_tokens]` | int64 | contiguous | State-cache slot; `-1` pad. |
| `block_table_ptr` | `[num_reqs, max_blocks]` | int32 | strided | Per-request logical→physical block. |
| `block_size` | scalar | int | — | State-cache block size. |
| `rms_norm_weight_ptr` | `[HEAD_SIZE = 128]` | bf16/fp32 | contiguous | RMSNorm gain. |
| `rms_norm_eps` | scalar | fp32 | — | RMSNorm epsilon. |
| `cos_sin_cache_ptr` | `[max_pos, rope_head_dim = 64]` | fp32 | cos/sin halves | Compressor RoPE cache (uses `compress_rope_theta=160000`; **different** from main attention RoPE). |
| `k_cache_ptr` | `[num_kv_blocks, kv_cache_block_size, 1, TOKEN_STRIDE + SCALE_DIM] = [..., 132]` per token | uint8 | per token: `[0,128) FP8 data`, `[128,132) one float32 scale` | Indexer paged K cache. |
| `kv_slot_mapping_ptr` | `[num_tokens]` | int64 | contiguous | KV-cache slot; `-1` pad. |
| `kv_cache_block_size` | scalar | int | — | KV-cache tokens/block. |
| `HEAD_SIZE` | constexpr int = 128 | — | — | Indexer head dim. |
| `TRITON_BLOCK_SIZE` | constexpr int = 128 | — | — | `next_power_of_2(128) = 128`. |
| `STATE_WIDTH` | constexpr int = 256 | — | — | `coff * head_dim = 2*128`. |
| `COMPRESS_RATIO` | constexpr int = 4 | — | — | Indexer always uses ratio=4. |
| `OVERLAP` | constexpr bool = True | — | — | `compress_ratio == 4`. |
| `ROPE_HEAD_DIM` | constexpr int = 64 | — | — | RoPE dims at tail. |
| `FP8_MAX` | constexpr fp32 = 448.0 | — | — | e4m3 max. |
| `QUANT_BLOCK` | constexpr int = 128 | — | — | Single quant block (equals HEAD_SIZE; static_assert at line 451-454). |
| `TOKEN_STRIDE` | constexpr int = 128 | — | — | FP8 bytes per token. |
| `SCALE_DIM` | constexpr int = 4 | — | — | One fp32 scale per token (4 bytes). |
| `KV_BLOCK_STRIDE` | constexpr int | — | — | `kv_cache.stride(0)`. |

## Outputs

In-place writes into `k_cache_ptr`:

| Region | Shape per token | Dtype | Meaning |
| --- | --- | --- | --- |
| FP8 values | `[0, 128)` (128 elements) | `float8_e4m3fn` stored as uint8 | Per-token FP8-quantized indexer K (NoPE + RoPE-rotated tail). |
| FP32 scale | `[block_size * 128 + slot_in_block * 4, +4)` | fp32 | Per-token scalar dequant scale (`exp2(exponent)`). UE8M0-discrete (only powers of two) by construction. |

Non-boundary tokens (`(position + 1) % COMPRESS_RATIO != 0`) and padded tokens early-exit.

## Grid / Block

- `grid_dim = (num_actual,)` — one CTA per token.
- `block_dim`: `num_warps=1` (dispatcher line 68), so 32 threads/CTA.
- Autotune configs: **none** — `num_warps=1` hard-coded.
- Per-CTA work:
  1. Early-exit on `slot_id < 0` or `(position + 1) % COMPRESS_RATIO != 0`.
  2. Gather `(1 + OVERLAP) * COMPRESS_RATIO = 8` rows from state cache (same overlap-window logic as the sparse-attn kernel).
  3. Softmax + weighted sum across the window → `compressed_kv [128]` fp32.
  4. RMSNorm fp32.
  5. Forward GPT-J RoPE on the trailing `rope_head_dim = 64` lanes (positions window-aligned).
  6. **Single-block FP8 quant** (skip the 2-D `reshape((N_BLOCKS, QUANT_BLOCK))` — kernel uses flat `tl.max(tl.abs(...), axis=0)` because `TRITON_BLOCK_SIZE == QUANT_BLOCK`); compute `scale = 2^ceil(log2(amax / FP8_MAX))`, store one fp32 scale per token (not UE8M0 byte — the indexer cache uses an fp32 layout inherited from V3.2).
  7. Write 128 fp8e4m3 bytes + 1 fp32 scale.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:316-377` (`Compressor.forward`). For the indexer the same `Compressor` class is used but with `rotate=True` (model.py:398) — vLLM splits the Hadamard rotation off into the Q-side FP8 path, so this K-side kernel does NOT apply the rotation; it only does compress + RMSNorm + RoPE + FP8 quant. Compressor RoPE uses `compress_rope_theta=160000` (model.py:67 default, set to 160000 in V4-Flash-Base config) — distinct from the main attention's `rope_theta=10000`.

```python
# Per-CTA work for token t (PyTorch-operator equivalent):
slot_id  = slot_mapping[t]; position = positions[t]
if slot_id < 0 or (position + 1) % 4 != 0: return       # COMPRESS_RATIO=4

req_idx = token_to_req_indices[t]
W = (1 + True) * 4                                       # = 8
start, pos = position - W + 1, position - W + 1 + torch.arange(W)
mask = pos >= 0
blk_idx, blk_off = pos // block_size, pos % block_size
blk_no = block_table[req_idx, blk_idx]
head_off = (torch.arange(W) >= 4).to(int) * 128          # overlap vs current half

# 1. Compress (softmax + weighted sum).
sc = state_cache[blk_no, blk_off, head_off + 256 : head_off + 256 + 128]   # score half
sc = sc.masked_fill(~mask, -inf).softmax(dim=0)                            # [8, 128]
kv = state_cache[blk_no, blk_off, head_off : head_off + 128]               # kv half
compressed_kv = (kv * sc).sum(dim=0)                                       # [128] fp32

# 2. RMSNorm (fp32).
var = (compressed_kv ** 2).mean()
normed = compressed_kv * torch.rsqrt(var + rms_norm_eps) * rms_norm_weight # [128]

# 3. Forward GPT-J RoPE on the rope tail (window-aligned position).
nope, rope = normed[:64], normed[64:]                                      # NOPE=64, ROPE=64
even, odd = rope.view(32, 2).unbind(-1)
compressed_pos = (position // 4) * 4
cos = cos_sin_cache[compressed_pos, :32]
sin = cos_sin_cache[compressed_pos, 32:]
new_even = even * cos - odd * sin
new_odd  = odd  * cos + even * sin
rotated  = torch.stack([new_even, new_odd], dim=-1).reshape(-1)            # [64] fp32
full = torch.cat([nope, rotated])                                          # [128]

# 4. Single-block FP8 quant.  bf16 roundtrip to match reference.
full = full.to(torch.bfloat16).to(torch.float32)
absmax = full.abs().amax().clamp_min(1e-4)
exponent = torch.ceil(torch.log2(absmax / FP8_MAX))
inv_scale = torch.exp2(-exponent)
fp8 = (full * inv_scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
scale_fp32 = torch.exp2(exponent)                                          # power-of-two scalar

k_cache.values_region[slot] = fp8                                          # 128 bytes
k_cache.scale_region[slot]  = scale_fp32                                   # 4 bytes (one fp32)
```

Notes on fusion / numerics:
- Single quant block: `tl.static_assert(TRITON_BLOCK_SIZE == QUANT_BLOCK)` (line 451-454). The reshape-tile pattern of the sparse-attn kernel collapses to a flat reduction here.
- Scale layout is **fp32 (not UE8M0-byte)** — the indexer cache inherits the V3.2 layout: `head_dim_bytes = 128 FP8 + 4 fp32 = 132 bytes` per token (`attention.py:787`). The scale value is still discretized to powers of two via `exp2(ceil(log2(...)))`, but stored as the full fp32 value.
- The RoPE rotation uses the *Compressor's own* `cos_sin_cache` (built with `compress_rope_theta=160000`), distinct from the main attention RoPE. The Q-side counterpart (`_fused_indexer_q_rope_quant_kernel`) uses the SAME compressor RoPE cache (passed via `rotary_emb.cos_sin_cache` at `attention.py:844`) — Q-side and K-side must use the same RoPE base.
- Hadamard rotation (`rotate_activation` at model.py:369 / `rotate=True`): NOT performed in this kernel. For the indexer's compressor `rotate=True`, but vLLM's pipeline applies the rotation upstream (in the wkv GEMM or as a fused output transform) — the kernel signature matches the non-rotated path.

## Config-dependent dispatch

- Activation condition: `head_dim == 128` AND `use_fp4_cache == False`.
- Variants (**Class B** per D3):
  - **Sibling K-side kernel (this kernel's pair under `use_fp4_cache`)**: `_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn` at `fused_compress_quant_cache.py:479-667` (MXFP4 K-side cache, spec file `fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn.md`).
  - **`use_fp4_cache` coupling (D3)**: this FP8 K-side variant is selected by `use_fp4_cache=False`. The flag is read from `vllm_config.attention_config.use_fp4_indexer_cache` (`attention.py:746`) and FLIPS BOTH SIDES — picking FP8 K here means the Q-side **must** also be FP8 (`_fused_indexer_q_rope_quant_kernel`, spec `fused_indexer_q_rope_quant.md`). Conversely if `use_fp4_cache=True`, both this kernel AND the FP8 Q-side kernel are skipped in favor of the MXFP4 pair (sibling K-side + `_fused_indexer_q_rope_mxfp4_kernel`). The 4 variants form a 2-way coupled choice — not 4 independent dispatches.
  - Coupled Q-side spec: `fused_indexer_q_rope_quant.md` (same `use_fp4_cache=False` branch).
- Class: **B** — `use_fp4_cache` parameterizes a 4-variant coupled cluster.
- Downstream consumer: `fp8_fp4_mqa_logits` (prefill, `sparse_attn_indexer.py:233`) and `fp8_fp4_paged_mqa_logits` (decode, `sparse_attn_indexer.py:324`) read this paged K cache via the `kv_cache_as_quant_view(..., use_fp4_cache=False)` helper. The 132-byte/token layout (128 FP8 + 4 fp32 scale) is hard-coded across kernel + cache + DeepGEMM scalar-type tagging.
