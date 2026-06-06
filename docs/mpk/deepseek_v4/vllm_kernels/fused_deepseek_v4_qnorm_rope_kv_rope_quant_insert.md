# fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert

## Identity
- Source file: `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:111-534`.
  - Per-slot pipeline `processDeepseekV4Slot` (lines 137-338) — shared by both grid variants.
  - Full-grid kernel `fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel` (lines 367-444) — used when `num_tokens_full < NUM_TOKEN_CUTOFF = 1024`.
  - Reduced-grid variant `fusedDeepseekV4QNormRopeKVRopeQuantInsertKernelReducedGrid` (lines 457-538) — used when `num_tokens_full ≥ 1024`.
  - Templated launcher `launchFusedDeepseekV4Templated` (lines 543-610); runtime dispatcher `launchFusedDeepseekV4QNormRopeKVRopeQuantInsert` (lines 614-645); Torch op `fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` (lines 653-723).
- Language/DSL: **CUDA** (`__global__`), warp-cooperative; bf16/fp16 templated via `vllm::_typeConvert`. Bound through `csrc/torch_bindings.cpp:60-65`, declared in `csrc/ops.h:43`.
- Third-party dep: none on the device side (CUDA toolkit only). Adapted from TRT-LLM `applyMLARopeAndAssignQKVKernelGeneration` (header comment lines 10-14) and vLLM's `fusedQKNormRopeKernel`.
- Registered as: `torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` (CUDA-only impl key).

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/attention.py:531-541` | `DeepseekV4MultiHeadLatentAttentionWrapper._fused_qnorm_rope_kv_insert` | `q: [T, n_local_heads, 512]`, `kv: [T, 512]`, `swa_kv_cache_2d: [num_blocks, block_stride_bytes]`, `slot_mapping: [T_insert]`, `positions: [T] i64`, `cos_sin_cache: [max_pos, 64] fp32`, `padded_heads ∈ {8,16,32,64,128}`, `eps`, `block_size` | bf16 (V4-Flash) or fp16 (`VLLM_DISPATCH_HALF_TYPES`) | NVIDIA only (`current_platform.is_rocm()` callers stay on AMD `rocm_inv_rope_einsum` path); skipped in profile runs (`attn_metadata` is not a dict — attention.py:505-514) |
| `tests/kernels/test_fused_deepseek_v4_qnorm_rope_kv_insert.py:126` | unit test driver | matches above | bf16 | test-only |

Production call site fires once per attention-layer forward, downstream of `wq_b(qr)` and concurrently with the indexer/compressor on aux streams (attention.py:441-490). Output `q_out` flows directly into `self.mla_attn(q, kv, positions, output=out)` at attention.py:494.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `q_in` | `[N = num_tokens_full, num_heads_q, HEAD_DIM=512]` | bf16 / fp16 (templated) | row-major, contiguous (`TORCH_CHECK` at `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:662-663,671-672`) | Q heads from `wq_b`, RoPE not yet applied |
| `kv_in` | `[N, HEAD_DIM=512]` | bf16 / fp16 (matches `q_in`) | row-major, contiguous (`csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:664,673,674`) | KV latent post-`kv_norm`, RoPE not yet applied. Read-only on device. |
| `k_cache` | `[num_blocks, block_stride_bytes]` | `torch.uint8` (`csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:665,677`) | paged; per-block layout: `[block_size * 576 bytes token data] + [block_size * 8 bytes scales]` (header comment lines 26-29, constants lines 81-87) | Sparse-SWA paged KV cache (uint8 raw bytes); written in place. `block_stride_bytes = k_cache.stride(0)` (line 696). |
| `slot_mapping` | `[T_insert = num_tokens_insert]`, `T_insert ≤ N` | int64 (`csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:666-667`) | contiguous | Per-token cache slot index. `slot_id < 0` skips the insert (line 263). `T_insert < N` handles DP padding (only first `T_insert` tokens get cache writes). |
| `position_ids` | `[N]` | int64 (`csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:668-669`) | contiguous | Absolute token positions for forward GPT-J RoPE (note: forward, not inverse). |
| `cos_sin_cache` | `[max_pos, ROPE_DIM=64]` | **fp32** (`csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:670,678-681`) | first 32 lanes cos, second 32 lanes sin | Precomputed RoPE table; shared with `fused_inv_rope_fp8_quant` (caller passes `self.rotary_emb.cos_sin_cache`). |
| `q_head_padded` | scalar | int64 → int | — | `num_heads_q_padded` template arg dispatched at lines 631-643. Must be `≥ num_heads_q`. Supported instantiations: **{8, 16, 32, 64, 128}**; other values `TORCH_CHECK`-fail. |
| `eps` | scalar | double → fp32 | — | RMSNorm epsilon (caller passes `self.eps`). |
| `cache_block_size` | scalar | int64 → int | — | Tokens per paged-cache block (caller passes `swa_metadata.block_size`). |

Templated compile-time constants (file lines 81-98):
- `kHeadDim = 512`, `kRopeDim = 64`, `kNopeDim = 448`
- `kQuantBlock = 64`, `kNumQuantBlocks = 7`, `kScaleBytesPerToken = 8` (7 real UE8M0 bytes + 1 pad)
- `kTokenDataBytes = 576` (448 fp8 nope + 128 = 2×64 bf16 rope)
- `kFp8Max = 448.0f`
- `kNumLanes = 32`, `kElemsPerLane = 16` (one warp = 32 lanes × 16 elems = 512 = HEAD_DIM)
- `is_neox = false` (GPT-J interleaved even/odd pairs; header comment line 23)

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `q_out` (function return) | `[N, q_head_padded, HEAD_DIM=512]` | matches `q_in.dtype` | allocated via `torch::empty` (line 703-704); row-major contiguous | Q post-RMSNorm (lanes < `num_heads_q`) + RoPE on trailing `rope_dim`; padded slots in `[num_heads_q, q_head_padded)` get zero-fill (`csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:153-163`) so FlashMLA can read padded heads safely. |
| `k_cache` (in-place mutation) | unchanged shape | unchanged dtype (uint8) | per-token slot at `slot_mapping[t]` (when `≥ 0`): fp8 nope `[0, 448)`, bf16 rope `[448, 576)`, UE8M0 scale `[block_size * 576 + slot_in_block * 8, +8)` | Paged KV cache: 7 UE8M0-quantized FP8 blocks (64 elems each = 448 nope dims) + raw bf16 RoPE-rotated `kv[..., 448:512]` + 7 single-byte exponents (8th byte zero-pad). |

Per-token cache layout (header lines 26-29, also lines 268-273):
```
block_base + pos_in_block * 576                  → 448 bytes FP8 nope
block_base + pos_in_block * 576 + 448            → 128 bytes BF16 rope (64 elems × 2 B)
block_base + block_size * 576 + pos_in_block * 8 → 7 UE8M0 scale exponents + 1 pad zero
```

## Grid / Block

The kernel ships TWO variants selected by `num_tokens_full` vs `NUM_TOKEN_CUTOFF = 1024` (lines 583-600). Both share `kBlockSize = 256` threads = 8 warps/CTA; one warp owns one (token, head-slot) pair across the full 512-wide head; lane `l` owns dims `[l*16, l*16+16)`.

Slot enumeration per token (file lines 130-136):
- `slot < num_heads_q` → **live-Q**: read `q_in[t, slot]`, RMSNorm (no weight) + GPT-J RoPE on last 64 dims, write `q_out[t, slot]`.
- `num_heads_q ≤ slot < q_head_padded` → **pad-Q**: write 32 B of zeros into `q_out[t, slot]`, no read.
- `slot == q_head_padded` → **KV**: read `kv_in[t]`, GPT-J RoPE on last 64 dims, UE8M0 FP8 quant on first 448 dims (across 7 blocks of 64), insert into paged cache at `slot_mapping[t]`.

**Full-grid variant** (`fusedDeepseekV4QNormRopeKVRopeQuantInsertKernel`, used when `num_tokens_full < 1024`):
- `total_warps = num_tokens_full * (q_head_padded + 1)`; `gridDim.x = ceil(total_warps / 8)` (lines 553-556).
- Each global warp maps to one `(tokenIdx, slotIdx)` via `globalWarpIdx / (q_head_padded + 1)` (lines 392-396).
- KV slots for DP-padded tokens (`tokenIdx ≥ num_tokens_insert`) early-out (line 402).

**Reduced-grid variant** (`...KernelReducedGrid`, used when `num_tokens_full ≥ 1024`):
- `gridDim.x = num_tokens_full` (line 592); each CTA owns one token, 8 warps iterate over the slot list `[0, q_head_padded + (1 if has KV slot else 0))`.
- Per-warp loop (lines 510-529) prefetches the next slot's `[uint4, uint4]` load while computing the current slot — buffer rotation amortizes LDG latency.

PDL / programmatic stream serialization (lines 408-410, 438-440, 558-581):
- `cudaLaunchAttributeProgrammaticStreamSerialization` enabled when `sm_version ≥ 90`.
- Device side: `cudaGridDependencySynchronize()` before any global load; `cudaTriggerProgrammaticLaunchCompletion()` at exit. **Active on sm_100a** (B200 V4-Flash target).

Vectorized memory ops:
- Per-warp loads: two `uint4` (16 B each) = 32 B = 16 bf16 elements per lane (`csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:417-431`).
- RoPE cos/sin: 4× `float4` per RoPE lane (lines 215-220).
- FP8 store: one `uint4` per lane = 16 fp8 bytes (line 303).
- BF16 RoPE store: two `uint4` = 32 B = 16 bf16 elements (lines 333-334).

Int64 indexing: every offset multiplication uses `static_cast<int64_t>(tokenIdx)` (lines 154-157, 254-257, 421-426, 493-499) to prevent int32 overflow on long sequences.

Multi-arch compile guard: bf16 device body is no-op'd for `__CUDA_ARCH__ < 800` (lines 381-388, 441-443, 466-470, 535-537) so multi-arch builds compile clean. Host-side `TORCH_CHECK(sm_version >= 80)` at line 567-571 makes the failure explicit. V4-Flash B200 runs sm_100a; well above the floor.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:497-506`:
```
q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))   # line 497
q *= torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)         # line 498  ← Q head-norm
apply_rotary_emb(q[..., -rd:], freqs_cis)                              # line 499  ← Q RoPE (forward)
kv = self.wkv(x); kv = self.kv_norm(kv)                                # 502-503  (kv_norm done upstream by fused_q_kv_rmsnorm)
apply_rotary_emb(kv[..., -rd:], freqs_cis)                             # line 504  ← KV RoPE (forward)
act_quant(kv[..., :-rd], 64, scale_fmt, scale_dtype, True)             # line 506  ← KV FP8 block quant (block=64)
```
The FP8 quant and paged-cache scatter are not in the simulator (reference uses bf16 in `kv_cache` per the comment at model.py:527); the vLLM kernel adds them to match the QAT recipe and feed FlashMLA's FP8 KV path.

```python
# PyTorch-operator equivalent of one warp's work for token t, slot s:
#
# Inputs:
#   q_in:         [N, num_heads_q, 512] bf16
#   kv_in:        [N, 512]              bf16
#   positions:    [N]                   int64
#   cos_sin:      [max_pos, 64]         fp32   (cos[:32] || sin[:32])
#   slot_mapping: [T_insert]            int64
#   k_cache:      [num_blocks, block_stride] uint8
#
# Slot dispatch:
is_kv    = (s == q_head_padded)
is_pad_q = (not is_kv) and (s >= num_heads_q)
is_live_q = not is_kv and not is_pad_q

# ── Pad-Q: zero-fill output and return.
if is_pad_q:
    q_out[t, s, :] = 0
    continue

# ── Load source into fp32 register array `x` of length 16 per lane (= 512 total).
if is_kv:
    x = kv_in[t, :].to(torch.float32)
else:
    x = q_in[t, s, :].to(torch.float32)

# ── Q branch: per-head RMSNorm, NO learnable weight (model.py:498).
if not is_kv:
    rrms = torch.rsqrt((x * x).sum() / 512 + eps)
    x = x * rrms

# ── Forward GPT-J RoPE on dims [448, 512). Even/odd interleaved pairs.
#    Applied to BOTH Q and KV; non-RoPE dims pass through.
pos = positions[t]
cos = cos_sin[pos, :32]              # [32]
sin = cos_sin[pos, 32:]              # [32]
x_rope = x[448:].view(32, 2)         # 32 pairs
even = x_rope[:, 0] * cos - x_rope[:, 1] * sin
odd  = x_rope[:, 0] * sin + x_rope[:, 1] * cos
x[448:] = torch.stack([even, odd], dim=-1).view(-1)

# ── Live-Q: cast back to bf16 and write to padded q_out.
if is_live_q:
    q_out[t, s, :] = x.to(q_out.dtype)
    continue

# ── KV branch: only for tokens with a real slot.
slot_id = slot_mapping[t]
if slot_id < 0:
    continue
block_idx     = slot_id // cache_block_size
pos_in_block  = slot_id %  cache_block_size

# Round-trip bf16 to match QAT precision before quant (line 277).
x_rt = x.to(torch.bfloat16).to(torch.float32)

# UE8M0 FP8 block quant on dims [0, 448) in 7 blocks of 64 each.
# Each lane owns 16 contiguous dims; a block of 64 spans 4 consecutive lanes
# (the `warp4MaxAbs` reduction unifies absmax across those 4 lanes — line 285).
absmax_local = x_rt[:448].view(7, 64).abs().amax(dim=-1)            # [7]
absmax       = absmax_local.clamp_min(1e-4)                          # line 285
exponent     = torch.ceil(torch.log2(absmax / 448.0))                # [7] fp32, UE8M0
inv_scale    = torch.exp2(-exponent)
x_q          = (x_rt[:448].view(7, 64) * inv_scale.unsqueeze(-1))    # broadcast
x_q          = x_q.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)      # SATFINITE/E4M3

# Cache writes per token slot (layout from header lines 26-29):
block_base = k_cache[block_idx]                                       # uint8 view
token_fp8_ptr   = block_base[pos_in_block * 576 :                     # 448 B fp8 nope
                             pos_in_block * 576 + 448]
token_bf16_ptr  = block_base[pos_in_block * 576 + 448 :               # 128 B bf16 rope
                             pos_in_block * 576 + 576]
token_scale_ptr = block_base[cache_block_size * 576 +                 # 8 B scales
                             pos_in_block * 8 :
                             cache_block_size * 576 + pos_in_block*8 + 8]

token_fp8_ptr  [:]  = x_q.view(-1).view(torch.uint8)
token_bf16_ptr [:]  = x[448:].to(torch.bfloat16).view(torch.uint8)
# UE8M0 scale storage: biased 8-bit exponent (exp + 127), saturated to [0, 255]
token_scale_ptr[:7] = torch.clamp(exponent + 127.0, 0.0, 255.0).to(torch.uint8)
token_scale_ptr[7]  = 0   # explicit pad byte (kernel line 312)
```

Notes on fusion / quant:
- This is a **5-op fusion**: (1) per-head RMSNorm no-weight on Q, (2) forward GPT-J RoPE on Q[448:], (3) forward GPT-J RoPE on KV[448:], (4) UE8M0 block-FP8 quant on KV[:448] with `block=64`, (5) scatter-write into paged uint8 cache at `slot_mapping[t]`. A single grid amortizes the read of `q_in`/`kv_in` and the SM occupancy across all 5 steps.
- GPT-J interleaved pairing (`is_neox=false`, header comment line 23): pairs are `(x[2p], x[2p+1])` not `(x[p], x[p+rope_dim/2])`. Forward direction here (model.py:499,504) — contrast with `fused_inv_rope_fp8_quant` which uses `inverse=True` (model.py:534).
- UE8M0 scale encoding (lines 286-287, 308-310): `exponent = ceil(log2(absmax / 448.0))`; stored as `clamp(exponent + 127, 0, 255)` byte (IEEE-754 fp32 exponent bias). This matches the convention `fp8_einsum` reads on the consumer side.
- Block-of-64 absmax across 4 lanes (line 103-110, `warp4MaxAbs`): each lane has 16 dims, and a quant block has 64 dims, so 4 lanes cooperate via `__shfl_xor_sync(mask=1)` then `mask=2`. Scale storage happens on `laneId & 3 == 0` (line 306-310) — one lane per 4-lane group.
- bf16 round-trip before quant (line 277, `Converter::convert(Converter::convert(elements[i]))`): the round-trip rfp32→bf16→fp32 step matches the QAT precision profile (the reference `act_quant` at model.py:506 operates on bf16 input directly).

## Config-dependent dispatch

- Activation condition: NVIDIA-only call site (attention.py:531). ROCm path uses BF16 attention + `rocm_inv_rope_einsum` and never touches this kernel.
- SM dispatch:
  - Multi-arch device guard: bf16 stub no-op'd for `__CUDA_ARCH__ < 800` (pre-Ampere). V4-Flash B200 is **`sm_100a`**, well above the floor. Host-side `TORCH_CHECK(sm_version >= 80)` at line 567 makes the failure mode explicit.
  - PDL (programmatic stream serialization) enabled when `sm_version ≥ 90` (lines 581, 408-410, 438-440). Active on sm_100a; the launch sets `numAttrs = 1` and the device side calls `cudaGridDependencySynchronize` / `cudaTriggerProgrammaticLaunchCompletion`.
  - SM90 (Hopper) is a **locked alternative** — same kernel, same code, just runs without sm_100a-specific tuning. No separate spec.
  - ROCm fallback: same kernel body but `<<< >>>` launch without PDL (lines 601-609). Out of scope for this batch.
- Grid-variant dispatch (line 583): `num_tokens_full < 1024` uses the full grid (1 CTA per ~8 (token, slot) pairs); `≥ 1024` switches to the reduced-grid variant (1 CTA per token, warps loop with prefetch). Same per-slot pipeline either way.
- Template dispatch on `q_head_padded` (lines 631-643): compiled instantiations for `{8, 16, 32, 64, 128}`. V4-Flash uses one of `{64, 128}` (FlashMLA padded-head requirement); other values `TORCH_CHECK`-fail.
- Dtype dispatch via `VLLM_DISPATCH_HALF_TYPES` (lines 706-721): bf16 and fp16 both supported; V4-Flash runs bf16.
- Hard preconditions (TORCH_CHECKs at lines 662-695):
  - `q_in.dim() == 3 && q_in.size(2) == 512`; `kv.dim() == 2 && kv.size(1) == 512`.
  - `q_in.dtype() == kv.dtype()`.
  - `q_head_padded >= num_heads_q`.
  - `k_cache.dtype() == uint8`; `slot_mapping.dtype() == position_ids.dtype() == int64`.
  - `cos_sin_cache.size(1) == 64` AND `cos_sin_cache.dtype() == float32` (notably bf16 will fail — the kernel reads cos/sin via `float4` LDGs at line 215-218).
  - `kv.size(0) == position_ids.size(0) == q_in.size(0)`; `slot_mapping.size(0) ≤ q_in.size(0)` (DP-padding allowance).
- Downstream consumer constraints:
  - `q_out` feeds `self.mla_attn(q, kv, ...)` (attention.py:494) → `flash_mla_with_kvcache` / `flash_mla_sparse_fwd`. Layout `[N, q_head_padded, 512]` bf16 contiguous is required; padded-slot zeros mean FlashMLA's per-head softmax sees 0 logits (combined with `attn_sink = -inf` for those slots).
  - `k_cache` is consumed by FlashMLA via raw uint8 indexing using `head_bytes = 448 + 128 + 7 + 1 = 584`. The `+1` pad byte at `token_scale_ptr[7]` is therefore load-bearing for FlashMLA's stride math.
