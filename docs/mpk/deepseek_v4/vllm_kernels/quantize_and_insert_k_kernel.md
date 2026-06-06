# quantize_and_insert_k_kernel

## Identity
- Source file: `vllm/models/deepseek_v4/common/ops/cache_utils.py:23-140` (Triton JIT body `quantize_and_insert_k_kernel`); user-facing entry `quantize_and_insert_k_cache` at lines 142-194.
- Language/DSL: **Triton** (`@triton.jit`).
- Third-party dep: none (pure Triton + PyTorch).
- Registered as opaque custom op: no — exported as a Python function via `vllm/models/deepseek_v4/common/ops/__init__.py:8, 26`.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| (none on the live V4-Flash NVIDIA forward path) | — | — | — | The standalone Triton kernel is NOT called by `DeepseekV4MLAAttention.forward` or by `DeepseekCompressor.forward` on the live NVIDIA path. The live SWA-cache writer is the fused C++ op `torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` at `vllm/models/deepseek_v4/attention.py:531-541`, and the live compressor-cache writer is `compress_norm_rope_store_cutedsl` / `_triton` at `vllm/models/deepseek_v4/compressor.py:340-380`. |
| `tests/kernels/test_compressor_kv_cache.py:90, 240, 419` | unit tests | `k: [T, 512] bf16`, `slot_mapping: [T] int64`, `k_cache: [num_blocks, block_bytes] uint8` | bf16 in, FP8+bf16+UE8M0 packed out | always — reference/test path |
| `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py:231, 326, 393` | unit tests / reference | same as above | same | always — reference for the fused C++ op |

The kernel exists as the **canonical reference** for the FP8+UE8M0 K-cache packing format and as an exported building block (`common.ops.quantize_and_insert_k_cache`). Production V4-Flash forward fuses this work into the C++ op; the Triton kernel remains as the un-fused, easier-to-port equivalent. Below it is spec'd as if live, since MPK will need this exact packing for its SWA-cache write path.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `k` | `[num_tokens, 512]` | bf16 | row-major, contiguous on last dim | Per-token K vector. Layout: first 448 lanes are the "NoPE" portion (to be FP8-quantized in 64-lane blocks); last 64 lanes are the "RoPE" portion (stored as bf16 verbatim). |
| `slot_mapping` | `[num_tokens]` | int64 | contiguous | Global K-cache slot id per input token (= `block_idx * cache_block_size + pos_in_block`). Value `-1` means "skip this token" (e.g. padded request slot). |
| `k_cache` | `[num_blocks, block_bytes]` | uint8 (raw bytes) | row-major | Paged FP8 K cache as a flat byte buffer. `block_bytes = k_cache.stride(0)` is per-block stride (includes padding). |
| `block_size` (= `cache_block_size`) | scalar `64` | int | — | Tokens per paged block. Locked at 64 for the SWA cache. |
| `is_ue8m0` | scalar `True` | bool | — | Locked True (asserted line 163). UE8M0 means "unsigned 8-bit exponent, no mantissa" — the per-block scale is constrained to a power of 2 (so only the IEEE-754 exponent byte is stored). |
| (constexprs forwarded from wrapper) | — | — | — | `input_dim=512`, `fp8_dim=448`, `bf16_dim=64`, `scale_dim=8`, `quant_block=64`, `token_data_size=576`, `fp8_max=448.0`, `n_quant_blocks=8` (7 real blocks + 1 padding scale slot). |

Block layout in `k_cache` (`block_size=64`, per the docstring lines 43-51):
- bytes `[0, 64*576)` = **per-token data**, contiguous tokens. Each token's 576 bytes = `448 fp8 NoPE + 128 bf16 RoPE` (the bf16 64 lanes take 128 bytes).
- bytes `[64*576, 64*576 + 64*8)` = **per-token scales**, each token has 8 uint8 scales (7 real + 1 padding). `64*8 = 512` bytes.
- bytes `[64*576 + 64*8, block_stride)` = **block padding** (the allocator rounds the per-block size up; the kernel doesn't touch this region).

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `k_cache` (in-place) | `[num_blocks, block_bytes]` | uint8 | as above | Same buffer; each input token's FP8 NoPE, UE8M0 scales, and bf16 RoPE are written into its assigned slot. Padding scale at index 7 is zeroed. Other bytes untouched. |

## Grid / Block

- `grid_dim = (num_tokens,)` — one CTA per input token.
- `block_dim` (threads/CTA): Triton-driven. The kernel uses one `tl.arange(0, quant_block)` vector (length 64), so `num_warps` defaults to 1 (32 threads/CTA). No explicit `num_warps` / `num_stages` set at launch — Triton's defaults.
- Autotune configs: **none**.
- Per-CTA work:
  - Early exit if `pid >= num_tokens` (defensive) or `slot_idx == -1` (padded token, line 60).
  - Decompose `slot_idx` into `(block_idx, pos_in_block)` via `divmod cache_block_size`.
  - Compute byte pointers: `token_data_ptr`, `token_scale_ptr` (note the **int64** cast on `block_idx * block_stride` at line 71 — required because `block_stride ~ 37KB` × `block_idx > ~57K` overflows int32).
  - Loop `qblock_idx ∈ [0, 8)` via `tl.static_range`:
    - For `qblock_idx ∈ [0, 7)` (real blocks): load 64 bf16 lanes, compute UE8M0 scale, quantize to FP8, write FP8 bytes + scale exponent.
    - The 7th iteration (`qblock_start = 7*64 = 448`) fails the `qblock_start < fp8_dim` guard since `fp8_dim=448`, so it is a no-op.
    - Padding scale at index 7 is then explicitly zeroed at line 129 (`tl.store(token_scale_ptr + 7, 0)`).
  - Copy bf16 RoPE in 4 chunks of 16 lanes each (`bf16_dim // 16 = 4` iterations).
- `block_stride` is the **runtime** value `k_cache.stride(0)`, not a fixed constant. The kernel is bytewise so all addressing is uint8.

## Math

Reference: `deepseek_v4/DeepSeek-V4-Flash/inference/model.py:502-504` — the K production:
```python
kv = self.wkv(x)
kv = self.kv_norm(kv)
apply_rotary_emb(kv[..., -rd:], freqs_cis)
```
The reference then either copies bf16 `kv` into `self.kv_cache` directly (line 520-523, 530) — **no FP8 quant in the reference**. vLLM's `quantize_and_insert_k_cache` is the production quantizer that the QAT note at model.py:505-506, 527 alludes to (`act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)` simulates the FP8 path; vLLM does the real FP8 write).

```python
# PyTorch-operator pseudocode for one input token t.
# Inputs:
#   k:            [T, 512] bf16
#   slot_mapping: [T] int64

slot = slot_mapping[t]
if slot == -1:
    return  # skip padded slots

block_idx, pos_in_block = divmod(slot, 64)        # block_size = 64
base       = block_idx * block_stride             # int64 multiply!
token_off  = base + pos_in_block * 576            # 576 = 448 fp8 + 128 bf16 (per-token data region)
scale_off  = base + 64 * 576 + pos_in_block * 8   # scale region (after all token data)

x = k[t].to(torch.float32)                        # [512]

# ===== FP8 quant of NoPE portion (lanes [0, 448) in 7 blocks of 64) =====
for qb in range(7):
    chunk    = x[qb*64 : (qb+1)*64]               # [64]
    amax     = chunk.abs().max().clamp(min=1e-4)  # match the CUDA fused op's 1e-4 floor
    raw      = amax / 448.0                       # fp8_e4m3 range
    exponent = torch.ceil(torch.log2(raw))        # UE8M0: power-of-2 scale
    scale    = torch.exp2(exponent)               # = 2^exponent
    fp8      = (chunk / scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    # Write FP8 bytes:
    k_cache_bytes[token_off + qb*64 : token_off + (qb+1)*64] = fp8.view(torch.uint8)
    # Write UE8M0 scale exponent (biased by 127, clamped to [0, 255]):
    enc_scale = (exponent + 127.0).clamp(0, 255).to(torch.uint8)
    k_cache_bytes[scale_off + qb] = enc_scale

# Padding scale slot (index 7) — always zero.
k_cache_bytes[scale_off + 7] = 0

# ===== Pass-through bf16 RoPE portion (lanes [448, 512), 64 elements, 128 bytes) =====
rope_bf16 = k[t, 448:512]                          # [64] bf16
k_cache_bf16_view = k_cache_bytes[token_off + 448 : token_off + 576].view(torch.bfloat16)
k_cache_bf16_view[:] = rope_bf16                   # written in 4 chunks of 16 in the kernel
```

Notes on fusion / packing:
- UE8M0 encoding: `stored = exponent + 127`, decoded `scale = 2^(stored - 127)`. Exponent biased like IEEE-754 fp32 exponent but only 8 bits used. Inverse is `dequantize_and_gather_k_kernel` (see that spec): `scale = exp2(encoded_scale - 127)`.
- `block_max = max(amax, 1e-4)` floor: matches the CUDA reference fused-op behaviour; prevents scale from being `0` when the block is all-zero (would produce `log2(0) = -inf`).
- Scale storage location: scales live AFTER all 64 tokens' data in the same paged block (offset `64 * 576 = 36864 B` from block start), not interleaved with each token. This packs token data contiguously for vector loads and lets scales be read with a separate, smaller TMA box.
- 7 real + 1 padding scales per token = 8 bytes/token. The padding slot at index 7 makes the scale region naturally aligned (`64 * 8 = 512 B`, a multiple of typical TMA box sizes).
- bf16 RoPE NOT quantized — the RoPE rotation amplifies the dynamic range and rounding errors compound across the sin/cos pair; keeping bf16 here matches the V4-Flash reference (model.py:504 just `apply_rotary_emb`, no quant).
- Per-token data total = 448 FP8 + 128 bf16 = **576 bytes**. Plus 8 scale bytes (in the trailing region) = 584 bytes effective per token. Matches `get_kv_cache_shape` at `vllm/models/deepseek_v4/nvidia/flashmla.py:107-109`.

## Config-dependent dispatch

- **Live activation on V4-Flash NVIDIA path**: none — this Triton kernel is **superseded** by the fused C++ op `torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` at `vllm/models/deepseek_v4/attention.py:531-541`, which folds (q RMSNorm + q RoPE + kv RoPE + FP8 quant + cache insert) into a single megakernel. The Triton kernel kept around as the canonical un-fused reference + a building block for unit tests + the AMD/ROCm path (which doesn't have the fused C++ op).
- **Locked alternatives** (do not spec separately):
  - Fused C++ op: `torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` (live NVIDIA writer; out of this wave's scope per Class A locking — only the closed-source `.cu`).
  - CuteDSL compressor-cache writer: `compress_norm_rope_store_cutedsl` at `vllm/models/deepseek_v4/nvidia/ops/sparse_attn_compress_cutedsl.py` — different kernel, different K-shape (head_dim=512 compressor path).
- **`is_ue8m0`**: locked True (asserted at line 163). No fp8_e5m2 variant; the kernel does not branch on it.
- **`block_size = 64`**: locked by the caller (default kwarg + `DeepseekV4SWACache` config).
- **`n_quant_blocks = 8`**: locked (7 real + 1 padding). The padding slot is structural (alignment), not a config knob.
- **Output layout contract**: any downstream consumer (i.e. the FlashMLA sparse decode kernel `flash_mla_with_kvcache` consuming `swa_cache`, and the un-fused `dequantize_and_gather_k_kernel` consuming the same cache for the prefill path) MUST read 656-byte tokens at the documented offsets. Misalign and you read garbage.
- **Cross-kernel pipeline**: the kernel is the **producer** for `dequantize_and_gather_k_kernel` (prefill path, see that spec) and for `flash_mla_with_kvcache`'s in-kernel dequant (decode path). MPK must reproduce the byte layout exactly.
