# dequantize_and_gather_k_kernel (Triton path)

## Identity
- Source file: `vllm/models/deepseek_v4/common/ops/cache_utils.py:197-305` (Triton JIT body `_dequantize_and_gather_k_kernel`); Triton wrapper `dequantize_and_gather_k_cache_triton` at lines 307-350; runtime dispatcher `dequantize_and_gather_k_cache` at lines 353-380.
- Language/DSL: **Triton** (`@triton.jit`).
- Third-party dep: none (pure Triton + PyTorch).
- Registered as opaque custom op: no — Python function.
- **Sibling (locked alternative, NOT spec'd here per D2)**: CuteDSL path `DequantGatherKCacheKernel` invoked via `dequantize_and_gather_k_cache_cutedsl` at `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py`. The dispatcher (`vllm/models/deepseek_v4/common/ops/cache_utils.py:367-380`) picks CuteDSL if `has_cutedsl()`, else falls back to this Triton kernel. MPK implementations should use this Triton variant as the semantic reference.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/nvidia/flashmla.py:373` | `DeepseekV4FlashMLASparseImpl._forward_prefill` (compressed-K gather) | `out: [PREFILL_CHUNK_SIZE=4, M, 576] bf16`, `k_cache (compressed): [num_blocks, 256, 656B]`, `seq_lens: [chunk_size] int32 (= seq_lens // compress_ratio)`, `gather_lens: None`, `block_table: [chunk_size, max_blocks_per_seq] int32`, `block_size: int (= attn_metadata.block_size // compress_ratio)`, `offset: 0` | bf16 out; uint8 cache | `compress_ratio > 1` (C4A or C128A); not called for SWA-only layers |
| `vllm/models/deepseek_v4/nvidia/flashmla.py:385` | `DeepseekV4FlashMLASparseImpl._forward_prefill` (SWA-K gather) | `out: [4, M, 576] bf16` (**same workspace, second pass**), `k_cache (SWA): [num_blocks, 64, 656B]`, `seq_lens: [chunk_size] int32`, `gather_lens: [chunk_size] int32`, `block_table: [chunk_size, ...] int32`, `block_size: 64`, `offset: N` | bf16 out; uint8 cache | always (every prefill chunk, all compress_ratio variants). Writes into rows `[N, N + gather_len)` of the same `kv` workspace already partially filled at offset 0. |

Two calls per prefill chunk (or one if `swa_only`). Both writes share the same `out` workspace (`kv`) but to disjoint row ranges (`[0, N)` for compressed, `[N, M)` for SWA), enabling `combine_topk_swa_indices` to address them with a single linearized index per query.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `out` | `[num_reqs, max_num_tokens, head_size=576]` (V4-Flash: `num_reqs = chunk_size ≤ PREFILL_CHUNK_SIZE = 4`, `max_num_tokens = M = N + window_size + max_num_batched_tokens`, `head_size = 576`) | bf16 | row-major | Dequantized K (and V — they're shared in MLA) workspace. Pre-allocated by the V4 layer's `workspace_manager.get_simultaneous(...)` call at line 362. |
| `out_stride0` | scalar | int | — | `out.stride(0)` — bytes between requests. |
| `out_stride1` | scalar | int | — | `out.stride(1)` — bytes between tokens within a request. |
| `k_cache` | `[num_blocks, block_size, head_bytes]` | uint8 (raw bytes) | paged | Source cache. **Both** call sites pass it with `head_bytes = 656` (656 = 448 fp8 NoPE + 128 bf16 RoPE + 8 scales — see `quantize_and_insert_k_kernel.md` for the byte layout). Either the layer's compressed K cache or the SWA K cache, depending on call site. |
| `seq_lens` | `[num_reqs]` | int32 | contiguous | Total sequence length per request. For the compressed-K call site (line 373), V4-Flash passes `seq_lens[chunk_start:chunk_end] // layer.compress_ratio` (effective number of compressed tokens). For the SWA-K call site (line 385), the un-divided `seq_lens` is passed. |
| `block_table` | `[num_reqs, max_blocks_per_seq]` | int32 | contiguous | Per-request paging table (logical block → physical block id). |
| `offset` | scalar | int | — | Output base-row offset. 0 for the compressed gather (writes to `out[:, 0:N, :]`), `N` for the SWA gather (writes to `out[:, N:N+gather_len, :]`). |
| `gather_lens` | `[num_reqs]` int32 or None | int32 | contiguous | When `None`: gather ALL `seq_len` tokens of the sequence (compressed case). When provided: gather only the **last** `gather_len[i]` tokens (SWA case — only the window of newest tokens). |
| `max_blocks_per_seq` (constexpr) | scalar | int | — | `block_table.shape[-1]`. |
| `fp8_dim` (constexpr) | scalar `448` | int | — | Lanes of FP8 NoPE per token. |
| `bf16_dim` (constexpr) | scalar `64` | int | — | Lanes of bf16 RoPE per token. |
| `scale_dim` (constexpr) | scalar `8` | int | — | Scale bytes per token (7 real + 1 padding). |
| `quant_block` (constexpr) | scalar `64` | int | — | FP8 block size; one UE8M0 scale per 64 lanes. |
| `cache_block_size` (constexpr) | scalar | int | — | Page block size. **64** for SWA cache, **256** for compressed cache (per `get_supported_kernel_block_sizes`). |
| `token_data_size` (constexpr) | scalar `576` | int | — | Bytes of token data (FP8+bf16 portion), not counting the trailing scale region. |
| `block_stride` (constexpr) | scalar | int | — | `k_cache.stride(0)` — runtime bytes per paged block. |
| `output_dim` (constexpr) | scalar `512` | int | — | Logical output dim per token = `fp8_dim + bf16_dim` = 512. (Not actually used as a stride — `out_stride1` covers that — but kept for the kernel's own sanity.) |
| `fp8_max` (constexpr) | scalar `448.0` | float | — | `float8_e4m3fn` max representable magnitude. |
| `n_quant_blocks` (constexpr) | scalar `7` | int | — | Real FP8 quant blocks per token. (The 8th scale slot is the padding — written by the producer, not read here.) |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `out` (in-place) | `[num_reqs, max_num_tokens, 512]` | bf16 | row-major | Dequantized K for each gathered token. NoPE 448 lanes written from `fp8_value * 2^(encoded_scale - 127)`; RoPE 64 lanes copied verbatim from the cache's bf16 region. Untouched rows are NOT zeroed by this kernel — they retain whatever the workspace last held. The caller relies on `combined_indices` + `topk_length` to never point at those rows. |

Note: the actual `out` last-dim allocation is **576** (per `flashmla.py:362-364`: `(chunk_size_const, M, q.shape[-1])` with `q.shape[-1] == 576`); the kernel only fills the first 512 lanes per token (448 dequant + 64 bf16 copy). Lanes [512, 576) appear to be unused by the downstream `flash_mla_sparse_fwd` consumer for the V portion (V is the first 512 lanes), but ARE read for QK (q has 576 dims). Either the workspace is zeroed elsewhere or the upper 64 lanes of dequantized K are written by some other path — **VERIFY**: the kernel writes `output_dim = 512` lanes but the `out` last-dim is 576, leaving 64 lanes uncovered. The caller may rely on those lanes being undefined-but-not-NaN.

## Grid / Block

- `grid_dim = (num_reqs, NUM_WORKERS=128)` — one CTA per (request, worker-stripe). `NUM_WORKERS` is hard-coded at line 329.
- `block_dim` (threads/CTA): Triton default. Vector lanes = 64 (`quant_block`) and 16 (RoPE chunk size), so `num_warps` defaults to 1 (32 threads/CTA).
- Autotune configs: **none**. No explicit `num_warps`/`num_stages` at launch — Triton defaults.
- Per-CTA work:
  - Read `seq_len = seq_lens[batch_idx]`. Read `gather_len = gather_lens[batch_idx] if gather_lens_ptr is not None else seq_len`. Compute `start_pos = seq_len - gather_len` (tail of the sequence).
  - Stripe across tokens: `for i in range(worker_id, gather_len, num_workers)` — worker_id `∈ [0, 128)` interleaves tokens at stride 128 within the gather range.
  - For each owned token at sequence position `pos = start_pos + i`:
    - Find paged block: `block_in_seq = pos // cache_block_size`, `pos_in_block = pos % cache_block_size`.
    - Look up physical block: `block_table[batch_idx, block_in_seq]` (int32).
    - Compute byte pointer: `cache_block_ptr = k_cache + int64(physical_block_idx) * block_stride` — note the **int64** cast (line 246) for the same overflow reason as in the producer kernel.
    - Dequant 7 FP8 blocks of 64 lanes each → 448 bf16 lanes; copy 64 bf16 RoPE lanes (in 4 chunks of 16).
    - Write to `out[batch_idx, offset + i, 0:512]`.
- Stride trick: `out_stride0` and `out_stride1` are passed explicitly so the kernel doesn't assume contiguous; this lets the caller pass a `.view()` of a larger workspace.

## Math

Reference: `deepseek_v4/DeepSeek-V4-Flash/inference/model.py:531-533` — the decode KV-cache read:
```python
if self.compress_ratio:
    self.compressor(x, start_pos)
o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
```
The reference reads `self.kv_cache` as bf16 directly. vLLM's `_dequantize_and_gather_k_kernel` is the **production** dequantizer that bridges the FP8 paged cache (produced by `quantize_and_insert_k_kernel` / fused C++ op) to the bf16 representation `flash_mla_sparse_fwd` consumes.

```python
# PyTorch-operator pseudocode for one CTA's work (request batch_idx, worker_id).
# Inputs:
#   k_cache      : [num_blocks, block_size, 656B] uint8 (656 = 448 + 128 + 8*1 + padding)
#   seq_lens[b]  : int32
#   gather_lens[b]: int32 or None
#   block_table  : [num_reqs, max_blocks_per_seq] int32
#   offset       : int (0 for compressed, N for SWA)

seq_len    = seq_lens[batch_idx]
gather_len = gather_lens[batch_idx] if gather_lens is not None else seq_len
start_pos  = seq_len - gather_len      # gather the LAST gather_len tokens

# Worker stripe across the gather range:
for i in range(worker_id, gather_len, 128):
    pos               = start_pos + i
    block_in_seq      = pos // cache_block_size           # 64 (SWA) or 256 (compressed)
    pos_in_block      = pos %  cache_block_size
    physical_block_id = block_table[batch_idx, block_in_seq]    # int32
    base              = int64(physical_block_id) * block_stride # MUST be int64

    # Slice the 656-byte token record:
    token_data_off  = base + pos_in_block * 576           # 576 = 448 + 128 (per-token data)
    token_scale_off = base + cache_block_size * 576 + pos_in_block * 8

    # ===== Dequant FP8 NoPE (7 blocks of 64 lanes) =====
    for qb in range(7):
        fp8_bytes     = k_cache[token_data_off + qb*64 : token_data_off + (qb+1)*64]      # uint8
        fp8           = fp8_bytes.view(torch.float8_e4m3fn)
        x_f32         = fp8.to(torch.float32)
        encoded_scale = k_cache[token_scale_off + qb].to(torch.uint8)                     # UE8M0
        scale         = torch.exp2(encoded_scale.to(torch.float32) - 127.0)
        x_dequant     = x_f32 * scale
        out[batch_idx, offset + i, qb*64 : (qb+1)*64] = x_dequant.to(torch.bfloat16)

    # ===== Copy bf16 RoPE (64 lanes, in 4 chunks of 16) =====
    rope_bf16 = k_cache[token_data_off + 448 : token_data_off + 576].view(torch.bfloat16) # [64]
    out[batch_idx, offset + i, 448:512] = rope_bf16
```

Notes on fusion:
- This is the exact inverse of `quantize_and_insert_k_kernel` (see that spec). The UE8M0 decoder is `scale = exp2(encoded_scale - 127)`.
- "Gather last `gather_len` tokens" behaviour is the SWA window: for the SWA-K call site (line 385), `gather_lens[i] = min(seq_lens[i], window_size)`, so only the most-recent window of bf16-dequantized K is loaded. For the compressed-K call site (line 373), `gather_lens` is `None`, so all `seq_len // compress_ratio` compressed tokens get loaded.
- The kernel's `offset` parameter shifts the output write position so the two calls can share one `kv` workspace: the compressed-K gather writes `out[:, 0:N, :]`, the SWA-K gather writes `out[:, N:N+gather_len, :]`. Downstream `combine_topk_swa_indices` then emits indices in `[0, N)` for compressed-pool references and `[N, M)` for SWA references, all sharing the same flat `kv` after the caller's `.view(-1, 1, 576)` reshape.
- Worker stripe of 128 ≠ a thread count; each "worker" is a separate CTA running in parallel along the `gather_len` axis. With `gather_len = 128` and 4 requests, the grid is `(4, 128)`, so 512 CTAs.

## Config-dependent dispatch

- Activation condition: always during prefill (both call sites at `vllm/models/deepseek_v4/nvidia/flashmla.py:373, 385`); not called during decode (which dequantizes inside `flash_mla_with_kvcache` itself).
- **CuteDSL vs Triton dispatch** (per **D2**): the dispatcher at `vllm/models/deepseek_v4/common/ops/cache_utils.py:367-380`:
  ```python
  if has_cutedsl():
      from vllm.models.deepseek_v4.nvidia.ops.dequant_gather_k_cutedsl import (
          dequantize_and_gather_k_cache_cutedsl,
      )
      dequantize_and_gather_k_cache_cutedsl(...)
      return
  dequantize_and_gather_k_cache_triton(...)
  ```
  Per **D2**, only the Triton path is spec'd here. The CuteDSL sibling lives at `vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py` (`DequantGatherKCacheKernel`) — same math, optimized for SM100 via CuTeDSL TMA. **Locked alternative pointer only**; MPK uses this Triton kernel's math as the reference.
- **`gather_lens` None vs provided**: in-kernel branch at line 225. None ⇒ gather all `seq_len` tokens; provided ⇒ gather last `gather_len` tokens. Compressed-K call passes `None`; SWA-K call passes `gather_lens`. No separate spec needed.
- **`cache_block_size` 64 vs 256**: in-kernel constexpr. Compressed cache uses 256-token pages (per `DeepseekV4FlashMLASparseBackend.get_supported_kernel_block_sizes`); SWA cache uses 64. Same kernel handles both.
- **`offset` 0 vs N**: in-kernel runtime parameter. Both call sites use the same compiled kernel.
- **`compress_ratio` branching at the caller**: this kernel itself is `compress_ratio`-agnostic. The caller wraps the compressed-K call inside `if not swa_only:` (line 369); SWA-only layers (compress_ratio ≤ 1) skip it entirely.
- **`is_ue8m0`**: locked True. The kernel hard-codes the UE8M0 decode `exp2(scale - 127)`; no fp8_e5m2 path.
- **Output dim coverage**: kernel writes 512 lanes, allocates 576 — see "Outputs" caveat above.
- **Pipeline note** (cross-kernel): this kernel must run BEFORE `combine_topk_swa_indices` (which emits indices into the post-gather flat KV pool) and `flash_mla_sparse_fwd` (which reads that pool). Both gather call sites complete before the index-combine call at line 403.
