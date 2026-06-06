# flash_mla_with_kvcache

## Identity
- Source file: external — compiled C++/CUDA in `vllm._flashmla_C` (built into `_flashmla_C.abi3.so`, not in this tree). Underlying op: `torch.ops._flashmla_C.sparse_decode_fwd` (sparse path) / `torch.ops._flashmla_C.dense_decode_fwd` (dense path).
- Python wrapper: `vllm/third_party/flashmla/flash_mla_interface.py:54-177` (function `flash_mla_with_kvcache`).
- Re-export shim: `vllm/v1/attention/ops/flashmla.py:87-95` (`from vllm.third_party.flashmla.flash_mla_interface import flash_mla_with_kvcache`); availability gated by `_is_flashmla_available()` (lines 33-48). When unavailable a stub at line 105 raises.
- Language/DSL: **External CUDA** (NVIDIA FlashMLA library — `deepseek-ai/FlashMLA`). DeepSeek V4-Flash uses the **sparse** path (with `indices`); see config-dependent dispatch below.
- Third-party dep: `deepseek-ai/FlashMLA` (vendored under `vllm/third_party/flashmla/`).
- Registered as opaque custom op: yes — invoked through `torch.ops._flashmla_C.sparse_decode_fwd` inside the wrapper.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/nvidia/flashmla.py:286` | `DeepseekV4FlashMLASparseImpl._forward_decode` | `q: [num_decode_tokens, 1, h_q ∈ {64,128}, 576]`, `swa_cache: [num_blocks, 64, 1, 656B]`, `swa_indices: [num_decode_tokens, 1, window_size]`, optional `extra_k_cache: [num_blocks, 256, 1, 656B]`, `extra_indices_in_kvcache: [num_decode_tokens, 1, topk]` | q bf16; cache uint8 (FP8+scale+bf16 packed); indices int32; lse fp32 | Decode-only (`num_decodes > 0`); `is_fp8_kvcache=True` is forced; uses sparse decode path (`indices = swa_indices`). For `compress_ratio == 4` and `== 128`, also passes `extra_k_cache` + `extra_indices_in_kvcache`; for `compress_ratio <= 1` (SWA-only) those are `None`. |

Single live caller on NVIDIA. Runs once per `DeepseekV4MLAAttention.forward` per decode-batch step when `swa_metadata.num_decodes > 0`.

Note: A second decode-time caller `flash_mla_with_kvcache_fp8` exists at `vllm/v1/attention/ops/flashmla.py:123-153` and routes through `_flashmla_extension_C.fwd_kvcache_mla_fp8`; it is used by `vllm/v1/attention/backends/mla/flashmla.py:306` (the **dense** FlashMLA path) and is **NOT** the V4-Flash sparse decode path. Out of scope for this spec.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `q` | `[batch_size, seq_len_q, num_heads_q, head_dim=576]` — V4-Flash passes `q` from `_forward_decode` after `q.unsqueeze(1)` so `batch_size = num_decode_tokens`, `seq_len_q = 1`, `num_heads_q ∈ {64, 128}` (padded via `get_padded_num_q_heads`), `head_dim = 576` (512 NoPE + 64 RoPE) | bf16 | contiguous on last dim | Query. Each decode token treated as its own "request" (`batch_size = num_decode_tokens`). |
| `k_cache` | `[num_blocks, page_block_size, num_heads_k=1, head_dim]` — for V4-Flash this is `swa_cache_layer.kv_cache.unsqueeze(-2)`; `head_dim` of the underlying buffer is 656 bytes packed (see layout note) | uint8 (raw bytes) | paged | FP8+scale+bf16 packed K cache. Page block size is **64** for the SWA cache (per `DeepseekV4SWACache`) and **256** for the compressed cache (per `DeepseekV4FlashMLASparseBackend.get_supported_kernel_block_sizes`). |
| `block_table` | None | — | — | Sparse mode: `block_table` is `None`; addressing comes entirely from `indices` (each entry is `block_idx * page_block_size + offset`). |
| `cache_seqlens` | None | — | — | Sparse mode: `None`; per-query valid length comes from `topk_length`. |
| `head_dim_v` | scalar `512` | int | — | Output value-projection dim. Asserted == 512 by the kernel (DeepSeek MLA). |
| `tile_scheduler_metadata` | `FlashMLASchedMeta` instance | dataclass | — | Per-layer-type tile-scheduler state cache. `tile_scheduler_metadata.tile_scheduler_metadata` is `(num_sm_parts, TileSchedulerMetaDataSize) int32`; `tile_scheduler_metadata.num_splits` is `(1,) int32`. The first call with `have_initialized=False` lets the kernel run its in-kernel planner and fill those tensors via PyTorch's graph-aware allocator (so CUDA-graph replay reuses the same addresses). Subsequent calls with matching `config` skip the planner. See `vllm/v1/attention/backends/mla/sparse_swa.py:325-363` for the per-layer-type sharing rationale. |
| `cache_seqlens` (kwarg) | None | — | — | Same as the positional — kept None in sparse mode. |
| `is_fp8_kvcache` | scalar `True` | bool | — | Locked True by V4-Flash decode path (line 293). Selects sparse FP8 K cache layout (512B FP8 NoPE + 16B fp32 scales + 128B bf16 RoPE = 656B/token). |
| `indices` | `[batch_size, seq_len_q, topk]` (V4-Flash: `[num_decode_tokens, 1, window_size]`) | int32 | contiguous | SWA window indices into `swa_cache`. Each value is `block_idx * page_block_size + offset`. Invalid entries are `-1` or `>= num_blocks * page_block_size`. |
| `topk_length` | `[batch_size]` (V4-Flash: `[num_decode_tokens]`) | int32 | contiguous | Per-query valid count for `indices`. Only leftmost `topk_length[i]` slots in `indices[i]` are attended. |
| `attn_sink` | `[num_heads_q]` | fp32 | contiguous | Adds a learned per-head logit sink: `out *= exp(lse) / (exp(lse) + exp(attn_sink))`. `+inf` zeroes the head output; `-inf` is a no-op. Does not change returned `lse`. |
| `softmax_scale` | scalar | float | — | `layer.scale = 1/sqrt(head_dim_k) = 1/sqrt(576)` (set by `DeepseekV4MLAAttention.__init__`). |
| `extra_k_cache` | `[num_blocks, 256, 1, 656B]` or None | uint8 | paged | When `compress_ratio > 1` (C4A or C128A), this is the layer's own compressed K cache (`self_kv_cache.unsqueeze(-2)`). None for SWA-only layers (`compress_ratio <= 1`). Page block size **256**. |
| `extra_indices_in_kvcache` | `[num_decode_tokens, 1, topk]` int32 or None | int32 | — | Topk indices into `extra_k_cache`. For C4A: built by `compute_global_topk_indices_and_lens` (see that spec). For C128A: `attn_metadata.c128a_global_decode_topk_indices` (pre-computed at metadata build). Each value is `block_idx * 256 + offset` (or `-1` sentinel). |
| `extra_topk_length` | `[num_decode_tokens]` int32 or None | int32 | — | Per-query valid count for `extra_indices_in_kvcache`. C4A: `compute_global_topk_indices_and_lens` returns this. C128A: `attn_metadata.c128a_decode_topk_lens`. |
| `out` | `[num_decode_tokens, 1, num_heads_q, 512]` | bf16 | contiguous on last dim | Pre-allocated output buffer (= `output.unsqueeze(1)`). |

K cache token layout (sparse FP8 mode, per FlashMLA docs at `vllm/third_party/flashmla/flash_mla_interface.py:97-102`):
- First 512 bytes: quantized NoPE — 512 `float8_e4m3` values.
- Next 16 bytes: 4 × fp32 block scales (one scale per 128 FP8 lanes).
- Last 128 bytes: 64 × bf16 RoPE values (un-quantized for precision).
- Total: 656 bytes/token.

Note: V4-Flash's actual on-disk SWA cache layout is 584B/token (see `get_kv_cache_shape` at flashmla.py:107-109: `448 NoPE + 128 RoPE + 8 fp8 scale`), but the FlashMLA kernel reads the cache as 656B/token contract; resolution of this discrepancy lives inside the closed-source kernel and is not visible at the wrapper boundary.

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `out` | `[num_decode_tokens, 1, num_heads_q, 512]` | bf16 | contiguous | Attention output. V4-Flash writes in-place into the caller-provided `output.unsqueeze(1)` (line 301). |
| `softmax_lse` | `[batch_size, num_heads_q, seq_len_q]` (V4-Flash: `[num_decode_tokens, num_heads_q, 1]`) | fp32 | — | Log-sum-exp of attention scores per (query, head). Returned but discarded by V4-Flash (`out, _ = flash_mla_with_kvcache(...)`, line 286). |
| `sched_meta.tile_scheduler_metadata` (side-effect) | `(num_sm_parts, TileSchedulerMetaDataSize)` | int32 | — | Filled in-place on the first call (`have_initialized=False`); read-only on subsequent calls. |
| `sched_meta.num_splits` (side-effect) | `(1,)` | int32 | — | Filled in-place on the first call. |

The kernel also internally produces a `max_logits` tensor (per FlashMLA docs) but the `flash_mla_with_kvcache` wrapper does not return it.

## Grid / Block

External CUDA — exact `gridDim` / `blockDim` not in this tree (compiled into `.abi3.so`). What IS recoverable from the wrapper / FlashMLA public docs:

- **SM target**: SM90 (Hopper) and SM100 (Blackwell). V4-Flash on B200 dispatches to the **SM100** codepath. Selection done inside `.abi3.so`; the wrapper exposes no SM-selector argument. See `is_flashmla_sparse_supported()` at `vllm/v1/attention/ops/flashmla.py:63-78`: requires device capability family 90 or 100.
- **Tile-scheduler planner**: on the first invocation per `FlashMLASchedMeta`, an in-kernel planner runs with `have_initialized=False`, computes `(num_sm_parts, TileSchedulerMetaDataSize)` int32 metadata and a scalar `num_splits`, and allocates them via PyTorch's graph-aware allocator. The number of CTAs (== `num_sm_parts`) is the partitioning chosen by that planner — typically near the SM count, sized to balance work across persistent CTAs. Subsequent calls re-use the same allocation (assertions on lines 142-152 enforce that `batch_size`, `seq_len_q`, `num_heads_q`, `page_block_size`, `num_heads_k`, `causal`, `is_fp8_kvcache`, `topk`, `extra_page_block_size`, `extra_topk` all match).
- **Per-tile-CTA work**: each CTA owns a tile of (query, head-group) and walks its assigned slice of `topk` + `extra_topk` indices, gathering FP8+scale+bf16 tokens from the paged caches via the `indices` lists, dequantizing into registers, and accumulating an online-softmax attention. The 656-byte token layout suggests one TMA per token-block (512B FP8 + 16B scales + 128B bf16). Per FlashMLA's published numbers: up to **410 TFLOPS** on sparse decoding (H800 SXM5).
- **Hard shape constraints** (asserted by the wrapper / V4-Flash impl):
  - `head_dim` of `q` must equal `head_dim` of `k_cache` (576 for V4-Flash MLA).
  - `head_dim_v == 512`.
  - `num_heads_k == 1` (MLA — single K head shared across query heads).
  - `num_heads_q ∈ {64, 128}` (FP8 decode kernel limitation — see `DeepseekV4FlashMLASparseImpl.get_padded_num_q_heads` at `vllm/models/deepseek_v4/nvidia/flashmla.py:120-127`). V4-Flash pads up to the next supported value.
  - `is_fp8_kvcache=True` requires `causal=False` (asserted line 121 of the wrapper); and sparse mode (`topk != None`) requires `is_fp8_kvcache=True` (line 157).
  - `causal=False` is enforced because the per-query attended set is already determined by `indices` (sparsity defines causality).
  - SWA cache page block: **64**. Compressed cache page block: **256**. (Both fixed by `DeepseekV4SWACache` / `DeepseekV4FlashMLASparseBackend.get_supported_kernel_block_sizes`.)
  - `topk` (`indices.shape[-1]`) is `window_size` for the SWA portion; `extra_topk` is the C4/C128 indexer's topk (e.g. 2048 for C4A, configured via `index_topk`).
  - Per the FlashMLA SM100 sparse prefill spec (referenced for SM100 alignment rules): `B_TOPK = 64` for h_q=64 and `B_TOPK = 128` for h_q=128. The decode kernel uses similar internal block tiling; the wrapper does not surface this directly but V4-Flash's caller pads to 128 alignment for compatibility (see `combine_topk_swa_indices` spec).

## Math

Reference: `deepseek_v4/DeepSeek-V4-Flash/inference/model.py:528, 533` — the call site `o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)`. Decode is the `start_pos > 0` branch at line 530-533.

```python
# PyTorch-operator pseudocode mirroring the FlashMLA sparse decode kernel's fusion.
# Inputs (V4-Flash decode, per query token t in [0, num_decode_tokens), per head h):
#   q[t, h]          ∈ R^576   (NoPE[:512] + RoPE[512:])
#   swa_cache        — paged FP8+bf16 buffer with token layout described above
#   swa_indices[t]   ∈ Z^window_size  (kv slot ids; -1 for invalid)
#   swa_lens[t]      ∈ Z          (count of valid swa_indices entries)
#   topk_indices[t]  ∈ Z^topk     (kv slot ids in compressed pool; -1 for invalid)
#   topk_lens[t]     ∈ Z          (count of valid topk_indices entries)
#   attn_sink[h]     ∈ R          (learned per-head logit sink)
#   softmax_scale    = 1/sqrt(576)
#
# 1) Gather + dequant K from the two paged caches into a single per-query K tensor.
def gather_one(cache, idx):  # idx: int32, may be -1
    # Each 656-byte token: 512 fp8 NoPE + 4 fp32 scales (per 128 lanes) + 64 bf16 RoPE.
    fp8_bytes     = cache[idx, 0:512]                           # uint8 viewed as fp8e4m3
    block_scales  = cache[idx, 512:528].view(torch.float32)     # [4] fp32
    rope_bf16     = cache[idx, 528:656].view(torch.bfloat16)    # [64]
    fp8           = fp8_bytes.view(torch.float8_e4m3fn)         # [512]
    nope          = fp8.to(torch.float32).view(4, 128) * block_scales[:, None]
    return torch.cat([nope.view(-1), rope_bf16.to(torch.float32)], dim=-1)  # [576]

# Per query (t, h):
swa_valid   = swa_indices[t, :swa_lens[t]]
topk_valid  = topk_indices[t, :topk_lens[t]]              # None when swa_only
K_swa       = torch.stack([gather_one(swa_cache,   i) for i in swa_valid])    # [Lswa, 576]
K_extra     = (torch.stack([gather_one(extra_cache, i) for i in topk_valid])
               if topk_indices is not None else None)                          # [Ltopk, 576]
K           = torch.cat([K_swa] + ([K_extra] if K_extra is not None else []), dim=0)

# 2) Online-softmax sparse attention.
logits = (q[t, h] @ K.T) * softmax_scale                  # [L]
lse    = torch.logsumexp(logits, dim=-1)                  # scalar
p      = torch.softmax(logits, dim=-1)                    # [L]
V      = K[:, :512]                                       # MLA: V is the first 512 lanes of K
out_th = p @ V                                            # [512]

# 3) Optional attention sink (per-head logit-sink mixing).
if attn_sink is not None:
    out_th = out_th * (torch.exp(lse) / (torch.exp(lse) + torch.exp(attn_sink[h])))

# 4) Write to caller's preallocated buffer.
output[t, h, :] = out_th.to(torch.bfloat16)
```

Notes on fusion:
- MLA collapses K and V into one 512-dim NoPE pool — `V` is sliced from `K[:, :512]`, not a separate tensor. The 64-dim RoPE tail participates in QK only (so QK uses all 576 dims, but PV uses only 512).
- The kernel fuses (gather → fp8 dequant → bf16 RoPE blend → QK → online-softmax → PV → sink-mix → bf16 write) per tile. No intermediate tensor is materialized in HBM.
- Page-table free addressing: `indices[t, 0, k]` is a global slot id `(block_idx * page_block_size + offset)`; the kernel just decodes that to (block, in-block offset) once and TMA-fetches the 656B token directly.
- Two separate K pools (`swa` + `extra`) are concatenated logically; the kernel doesn't actually copy them, it walks both index lists in sequence with shared accumulator state.

## Config-dependent dispatch

- Activation condition: `vllm/models/deepseek_v4/nvidia/flashmla.py:286` whenever `num_decodes > 0` in the current step. Decode and prefill are split by `num_decode_tokens` (sparse_swa metadata).
- **Sparse vs dense mode** (in-wrapper): selected by `indices is not None`. V4-Flash **always passes `indices=swa_indices`**, so the sparse path (`flash_mla_cuda.sparse_decode_fwd`) is the only live path. Dense path (`flash_mla_cuda.dense_decode_fwd` at line 168 of the wrapper) is **locked off** and out of scope.
- **`is_fp8_kvcache`**: locked `True` for V4-Flash decode (line 293 of `flashmla.py`). Sparse mode requires it (assertion line 157).
- **Per-layer compress_ratio** branch in V4-Flash (lines 226-285 of `flashmla.py`) — selects which `tile_metadata` to share and whether `extra_k_cache` / `extra_indices_in_kvcache` are populated:
  - `compress_ratio <= 1` (SWA-only): `tile_metadata = swa_metadata.tile_sched_swaonly`, `extra_k_cache = None`, `extra_indices_in_kvcache = None`, `topk_indices = None`, `topk_lens = None`.
  - `compress_ratio == 4` (C4A): `tile_metadata = swa_metadata.tile_sched_c4a`. `topk_indices`/`topk_lens` from `compute_global_topk_indices_and_lens(layer.topk_indices_buffer[:num_decode_tokens], …)` (see that spec). `extra_k_cache = self_kv_cache.unsqueeze(-2)`.
  - `compress_ratio == 128` (C128A): `tile_metadata = swa_metadata.tile_sched_c128a`. `topk_indices = attn_metadata.c128a_global_decode_topk_indices`, `topk_lens = attn_metadata.c128a_decode_topk_lens` (pre-computed at metadata-build time, not by a kernel). `extra_k_cache = self_kv_cache.unsqueeze(-2)`.
- **`FlashMLASchedMeta` sharing**: one meta per **layer-type** (compress_ratio family), reused across all same-type layers in the same decode step — the first per-type call triggers the in-kernel planner; subsequent calls skip it. Sharing across layer types would trip the consistency asserts at wrapper lines 142-152.
- **Padded h_q**: `get_padded_num_q_heads(num_heads)` snaps to `{64, 128}`. V4-Flash has `n_heads=64` (default), so `h_q=64` is the locked configuration on a non-TP single-GPU run; TP-split runs (e.g. TP4 → 16 local heads) round up to 64.
- **`num_heads_k`**: locked at 1 (MLA).
- **SM target**: SM90 vs SM100 dispatch internal to the `.abi3.so`; not user-selectable. SM100 codepath used on B200 (V4-Flash target).
- **Alternative paths in vLLM that are NOT this caller** (locked pointers, do not spec):
  - Dense FlashMLA decode: `vllm/v1/attention/backends/mla/flashmla.py:306, 320` (bf16 and fp8 dense variants — for non-V4 DeepSeek backends).
  - V4 DSA via `flash_mla_with_kvcache_fp8` extension: `vllm/v1/attention/ops/flashmla.py:123-153` (separate `_flashmla_extension_C` op).
  - ROCm equivalent: `vllm/models/deepseek_v4/amd/rocm.py:678` (`_forward_decode`) using a different attention library — out of scope (amd/ subtree).
