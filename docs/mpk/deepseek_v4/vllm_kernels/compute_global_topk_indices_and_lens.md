# compute_global_topk_indices_and_lens

## Identity
- Source file: `vllm/models/deepseek_v4/common/ops/cache_utils.py:417-466` (Triton JIT body `_compute_global_topk_indices_and_lens_kernel`); user-facing entry `compute_global_topk_indices_and_lens` at lines 383-414.
- Language/DSL: **Triton** (`@triton.jit`).
- Third-party dep: none (pure Triton + PyTorch).
- Registered as opaque custom op: no — Python function.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/nvidia/flashmla.py:234` | `DeepseekV4FlashMLASparseImpl._forward_decode` | `topk_indices: layer.topk_indices_buffer[:num_decode_tokens]` — `[num_decode_tokens, topk]` int32 (LOCAL indices within compressed pool, produced by the C4 indexer kernel earlier in the layer); `token_to_req_indices: [num_decode_tokens]` int32; `block_table: attn_metadata.block_table[:num_decodes]` `[num_decodes, max_blocks_per_seq]` int32; `block_size: attn_metadata.block_size // layer.compress_ratio` int (= 256/4 = 64); `is_valid_token: swa_metadata.is_valid_token[:num_decode_tokens]` bool | int32 indices; bool valid-mask | Decode-only, AND only for `layer.compress_ratio == 4` (C4A). C128A path (compress_ratio == 128) uses `attn_metadata.c128a_global_decode_topk_indices` pre-computed at metadata-build time and bypasses this kernel entirely. SWA-only layers (compress_ratio ≤ 1) also skip. |

Single live caller. Runs once per C4A layer's decode forward.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `topk_indices` | `[num_tokens, topk]` (V4-Flash: `num_tokens = num_decode_tokens`, `topk = index_topk = 2048` per V4-Flash config) | int32 | row-major, contiguous on last dim | **Local** topk indices within the per-request compressed pool. Produced by the C4 indexer (`layer.indexer.forward`), which scored candidate positions and emitted the topk best LOCAL indices per token. `-1` (or any negative value) marks an invalid/padding entry. |
| `token_to_req_indices` | `[num_tokens]` | int32 | contiguous | Per-token request id (which sequence this query token belongs to). Maps decode-token slot → request slot for `block_table` lookup. |
| `block_table` | `[num_reqs, max_blocks_per_seq]` | int32 | contiguous | Per-request page table. Maps logical compressed-pool block id → physical block id in the layer's compressed K cache. |
| `block_size` | scalar | int | — | Page block size in the **compressed** cache, divided by compress_ratio. V4-Flash: `attn_metadata.block_size // 4 = 256 / 4 = 64` (256 = compressed pool's native page block, 4 = compress_ratio for C4A). This is the number of logical compressed tokens per physical paged block. |
| `is_valid_token` | `[num_tokens]` | bool (treated as int) | contiguous | Per-token validity mask — `False` for padded query slots. Padded tokens get their `topk_lens` zeroed. |
| `TRITON_BLOCK_SIZE` (constexpr) | scalar `1024` | int | — | Inner-loop tile size. Hard-coded at call site (line 412). |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `global_topk_indices` | `[num_tokens, topk]` (same shape as input `topk_indices`) | int32 | row-major | **Global** KV-cache slot ids — each value is `block_table[req, local_idx // block_size] * block_size + (local_idx % block_size)`. Invalid entries (local was `-1`) become `-1`. Suitable to pass to `flash_mla_with_kvcache` as `extra_indices_in_kvcache`. |
| `topk_lens` | `[num_tokens]` | int32 | contiguous | Count of VALID (non-negative) entries per token, AFTER masking out padded tokens (which are forced to length 0). |

Note: `global_topk_indices` is allocated via `torch.empty_like(topk_indices)` (line 398), so its dtype matches the input (int32). `topk_lens` is allocated `torch.empty(num_tokens, dtype=torch.int32, …)`.

## Grid / Block

- `grid_dim = (num_tokens,)` — one CTA per query token.
- `block_dim` (threads/CTA): Triton default. Single `tl.arange(0, TRITON_BLOCK_SIZE=1024)` vector ⇒ Triton typically picks `num_warps=4` (128 threads).
- Autotune configs: **none**. No explicit `num_warps` / `num_stages` at launch.
- Per-CTA work:
  - Load `is_valid_token[token_idx]` (scalar) and `req_idx = token_to_req_indices[token_idx]`.
  - Tile over `topk` lanes in steps of 1024:
    - Load 1024 local indices.
    - Mask `local_idx >= 0` (valid sentinel check).
    - Decompose: `block_indices = local_idx // block_size`, `block_offsets = local_idx % block_size`.
    - Vector-load physical block numbers: `block_numbers = block_table[req_idx, block_indices]` (masked by `is_valid` so invalid lanes don't IMA).
    - Compute `slot_ids = block_numbers * block_size + block_offsets`.
    - Where invalid: overwrite slot_id with `-1`.
    - Store `slot_ids` into `global_topk_indices[token_idx, tile_offset:tile_offset+1024]`.
    - Accumulate `count += sum(is_valid.to(int32))`.
  - Finally: `topk_lens[token_idx] = is_valid_token ? count : 0`.

For V4-Flash with `topk = 2048` and `TRITON_BLOCK_SIZE = 1024`, the inner loop runs exactly 2 iterations per CTA.

## Math

Reference: `deepseek_v4/DeepSeek-V4-Flash/inference/model.py:427-433`:
```python
topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
if seqlen > 1:
    mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
    topk_idxs = torch.where(mask, -1, topk_idxs + offset)
else:
    topk_idxs += offset
return topk_idxs
```
The reference's `+offset` step shifts indices into the **global window** (offset = `kv.size(1)` or `win`); in production-vLLM paging, this becomes `block_table` lookup since the global KV cache is paged. The reference also masks invalid topk slots (positions not yet attendable) to `-1` — same sentinel convention as this kernel.

```python
# PyTorch-operator pseudocode for one CTA's work (token_idx).
# Inputs:
#   topk_indices[t, :]    : [topk] int32      # LOCAL slot ids in compressed pool; -1 for invalid
#   token_to_req_indices  : [num_tokens] int32
#   block_table[r, :]     : [max_blocks] int32 # logical block → physical block
#   block_size            : int               # tokens per physical block in compressed cache
#   is_valid_token[t]     : bool

is_valid_t = is_valid_token[t]
r          = token_to_req_indices[t]
count      = 0

# Tile over topk lanes:
for tile_start in range(0, topk, 1024):
    local_idx        = topk_indices[t, tile_start : tile_start + 1024]      # [1024]
    is_valid         = local_idx >= 0                                       # mask
    block_idx_log    = local_idx.clamp(min=0) // block_size                  # masked load → safe
    block_off        = local_idx.clamp(min=0) % block_size
    block_phys       = torch.where(is_valid,
                                   block_table[r, block_idx_log],
                                   torch.full_like(block_idx_log, 0))       # masked gather
    slot_ids         = torch.where(is_valid,
                                   block_phys * block_size + block_off,
                                   torch.full_like(block_phys, -1))
    global_topk_indices[t, tile_start : tile_start + 1024] = slot_ids
    count += is_valid.to(torch.int32).sum()

# Padded query slot ⇒ zero length (regardless of how many valid local indices).
topk_lens[t] = count if is_valid_t else 0
```

Notes on fusion (the kernel's docstring at lines 390-396 enumerates the 3 fused ops):
- **Op 1 — Block-table lookup** (`local index → global slot id`): paging indirection, normally a `torch.gather`. Fused as `physical_block * block_size + offset`.
- **Op 2 — Valid-entry counting** (`count = sum(local_idx >= 0)`): would be a `(local_idx >= 0).sum(dim=-1)`. Fused into the per-tile reduction.
- **Op 3 — Padding mask** (`topk_lens = 0 for padded query tokens`): would be `topk_lens.masked_fill_(~is_valid_token, 0)`. Fused into the final store.
- Trade-off: the C128A path computes equivalent indices/lens at **metadata-build time** (deterministic from positions), avoiding this kernel entirely. C4A's topk is **data-dependent** (from `Indexer.forward`'s learned scoring), so it must be computed per-step.
- The kernel only does the paging indirection; the `+ offset` (logical → global) embedding from model.py:432 is replaced by `block_table` lookup since the production cache is paged, not contiguous.

## Config-dependent dispatch

- Activation condition: `vllm/models/deepseek_v4/nvidia/flashmla.py:234` only when `layer.compress_ratio == 4` (C4A) AND `num_decodes > 0` AND `not swa_only`.
- **C4A vs C128A**: C4A uses this kernel (data-dependent topk from `Indexer`). C128A uses `attn_metadata.c128a_global_decode_topk_indices` pre-computed during metadata build (deterministic positional pattern, identical for every step within a request). The C128A indices are computed once-per-request rather than once-per-token, so they don't need a per-step kernel. C128A also uses `attn_metadata.c128a_decode_topk_lens` for the length output.
- **SWA-only layers** (`compress_ratio ≤ 1`): no compressed pool, no topk; the kernel is bypassed by the `if not swa_only:` guard at line 226 of `flashmla.py`.
- **Decode vs prefill**: this is a **decode-only** kernel. The prefill path's analog (compute global indices for compressed pool) is folded into `combine_topk_swa_indices` (see that spec), which simultaneously concatenates SWA window indices — different fusion shape for prefill.
- **`TRITON_BLOCK_SIZE = 1024`**: hard-coded at the call site (line 412). For V4-Flash `topk = 2048`, this is 2 iterations. Not a config knob, just a tile-size constant.
- **Output consumer**: `global_topk_indices` is reshaped at line 241 to `[num_decode_tokens, 1, topk]` (adding the `h_kv=1` axis) and passed to `flash_mla_with_kvcache` as `extra_indices_in_kvcache`. `topk_lens` is passed as `extra_topk_length`. Both must agree on length semantics with the FlashMLA kernel.
- **Pipeline note** (cross-kernel): this kernel is the **decode-time analog** of part of `combine_topk_swa_indices`. Decode runs `compute_global_topk_indices_and_lens` (this) → `flash_mla_with_kvcache`. Prefill runs `dequantize_and_gather_k_cache` (×2) → `combine_topk_swa_indices` → `flash_mla_sparse_fwd`. Both ultimately produce the same input contract for the FlashMLA kernel: a (token, topk) int32 grid of pre-resolved global slot ids plus a per-token length tensor.
