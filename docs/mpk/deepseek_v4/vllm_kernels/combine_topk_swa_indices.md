# combine_topk_swa_indices

## Identity
- Source file: `vllm/models/deepseek_v4/common/ops/cache_utils.py:524-594` (Triton JIT body `_combine_topk_swa_indices_kernel`); user-facing entry `combine_topk_swa_indices` at lines 476-521. Padding alignment constant `_SPARSE_PREFILL_TOPK_ALIGNMENT = 128` at lines 468-473.
- Language/DSL: **Triton** (`@triton.jit`).
- Third-party dep: none (pure Triton + PyTorch).
- Registered as opaque custom op: no — Python function.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/nvidia/flashmla.py:403` | `DeepseekV4FlashMLASparseImpl._forward_prefill` | `topk_indices: [num_prefill_tokens_chunk, top_k]` int32 (LOCAL compressed-pool indices; for C4A from `layer.topk_indices_buffer`, for C128A from `attn_metadata.c128a_prefill_topk_indices`; for SWA-only this is a dummy placeholder with `top_k=0`); `query_start_loc: [num_chunk_reqs + 1]` int32 (the slice covers `[num_decodes + chunk_start, num_decodes + chunk_end + 1]`); `seq_lens: [chunk_size]` int32; `gather_lens: [chunk_size]` int32; `window_size: int = layer.window_size = 128`; `compress_ratio: int ∈ {1, 4, 128}` (effectively 4 or 128 in the not-`swa_only` branch; passed as 1 when SWA-only); `top_k: int (=topk_indices.shape[-1] for not-swa_only, else 0)`; `M: int = N + window_size + max_num_batched_tokens`; `N: int = ceil(max_model_len / compress_ratio) for not-swa_only, else 0` | int32 indices; int32 lens; int32 query_start_loc | Prefill-only. Always called regardless of compress_ratio: SWA-only layers pass `top_k = 0` to zero out the topk portion. |

Single live caller. Runs once per prefill chunk (inside the `for chunk_idx in range(num_chunks)` loop at line 365), AFTER the two `dequantize_and_gather_k_cache` calls and BEFORE `flash_mla_sparse_fwd`.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `topk_indices` | `[num_tokens, top_k_padded]` (V4-Flash: `num_tokens = num_prefill_tokens_in_chunk`; `top_k_padded` = caller's local-pool topk width, e.g. 2048 for C4A) | int32 | row-major | **Local** compressed-pool indices per query token. Already produced by the indexer (C4A) or metadata-build (C128A). `-1` for invalid. The `top_k` constexpr passed into the kernel may be **less** than `topk_indices.shape[-1]`: callers pass C4A's `top_k = topk_indices.shape[-1]` but SWA-only callers pass `top_k = 0` (kernel skips the topk-load loop). |
| `query_start_loc` | `[num_reqs_chunk + 1]` (V4-Flash slice: `query_start_loc[num_decodes + chunk_start : num_decodes + chunk_end + 1]`) | int32 | contiguous | Chunk-local cumulative query-token boundaries. The kernel **rebases** by subtracting `query_start_loc[0]` so per-batch ranges are zero-based within the chunk. |
| `seq_lens` | `[num_reqs_chunk]` (V4-Flash: `seq_lens[chunk_start:chunk_end]`) | int32 | contiguous | Total sequence length (including the new query tokens) per request in the chunk. |
| `gather_lens` | `[num_reqs_chunk]` (V4-Flash: `gather_lens[chunk_start:chunk_end]`) | int32 | contiguous | Number of SWA tokens gathered (≤ window_size + new query tokens). Determines `gather_start = seq_len - gather_len` (where the SWA window starts in absolute sequence coords). |
| `M` | scalar | int | — | Per-request stride within the per-chunk linearized `kv` workspace. `M = N + window_size + max_num_batched_tokens` (line 357 of flashmla.py). Each request occupies a row of size `M` in the flat `kv` (after `kv.view(-1, 1, 576)`). |
| `N` | scalar | int | — | Per-request offset of the SWA region within a single request's `M` slots. `N = ceil(max_model_len / compress_ratio)` (not-swa_only) or `0` (swa_only). Compressed-pool indices live in `[M*b, M*b + N)`; SWA indices live in `[M*b + N, M*(b+1))`. |
| `TOP_K` (constexpr) | scalar | int | — | Per-token max compressed-pool indices to write. V4-Flash: `index_topk` (e.g. 2048) for C4A/C128A, **0** for SWA-only. The actual per-token valid count is `min((pos + 1) // COMPRESS_RATIO, TOP_K)`. |
| `COMPRESS_RATIO` (constexpr) | scalar `1`/`4`/`128` | int | — | `layer.compress_ratio`. Drives the per-token topk-length calc. SWA-only passes `1` (which then `(pos+1)//1 = pos+1` would over-count, but `TOP_K = 0` caps everything at 0 — see the `minimum(...)` in the kernel). |
| `WINDOW_SIZE` (constexpr) | scalar `128` | int | — | `layer.window_size`. Per-token SWA portion length is `min(pos + 1, WINDOW_SIZE)`. |
| `PADDED_TOP_K` (constexpr) | scalar | int | — | `triton.next_power_of_2(topk_indices.shape[-1])`. Vector load width for the topk-fetch loop. |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `combined_indices` | `[num_tokens, combined_topk]` where `combined_topk = align_up(top_k + window_size, 128)` (the alignment constant `_SPARSE_PREFILL_TOPK_ALIGNMENT`) | int32 | row-major | Concatenated, **chunk-linearized-global** KV slot ids: `combined_indices[t, :topk_len_t] = topk_indices[t] + M*batch_t`, then `combined_indices[t, topk_len_t : topk_len_t + swa_len_t] = M*batch_t + N + (window positions)`. All trailing slots up to `combined_topk` are `-1` (set by the `torch.full(...)` at line 494). Note: `M*batch_t` is the per-batch shift that lets `flash_mla_sparse_fwd` operate on the flat `kv` workspace. |
| `combined_lens` | `[num_tokens]` | int32 | contiguous | Per-token valid count = `topk_len + swa_len` (= the prefix of `combined_indices[t, :]` that's not a padding sentinel). |

`combined_topk` formula (line 489-493):
```python
combined_topk = (top_k + window_size + 128 - 1) // 128 * 128   # next multiple of 128
```
For V4-Flash C4A: `top_k = 2048`, `window_size = 128` → `combined_topk = ceil(2176 / 128) * 128 = 2176`.
For SWA-only: `top_k = 0`, `window_size = 128` → `combined_topk = 128`.

The 128-alignment is mandated by `flash_mla_sparse_fwd`: the SM100 h_q=128 kernel needs `topk % 128 == 0` (`B_TOPK = 128`); h_q=64 kernel needs `topk % 64 == 0` (`B_TOPK = 64`). Padding to 128 satisfies both.

## Grid / Block

- `grid_dim = (num_reqs, NUM_WORKERS=128)` — one CTA per (request-in-chunk, worker-stripe).
- `block_dim` (threads/CTA): Triton default. Vector lanes: `PADDED_TOP_K` (up to ~2048) and `WINDOW_SIZE` (128). Triton typically picks `num_warps=4-8` for `PADDED_TOP_K ≥ 1024`.
- Autotune configs: **none**. No explicit `num_warps` / `num_stages` at launch.
- Per-CTA work:
  - Read chunk-local query range: `query_start = query_start_loc[batch_idx] - query_start_loc[0]`, `query_end = query_start_loc[batch_idx + 1] - query_start_loc[0]`. `query_len = query_end - query_start`.
  - Read `seq_len = seq_lens[batch_idx]`, `gather_len = gather_lens[batch_idx]`. Compute `start_pos = seq_len - query_len` (absolute position of the first new query token in this request), `gather_start = seq_len - gather_len` (absolute position where SWA-K gather started).
  - Worker stripe across query tokens: `for token_idx in range(query_start + worker_id, query_end, 128)`:
    - `pos = start_pos + (token_idx - query_start)` — absolute sequence position of this query token.
    - `topk_len = min((pos + 1) // COMPRESS_RATIO, TOP_K)` — number of valid compressed-pool indices (the indexer is configured to produce at most this many).
    - `swa_len = min(pos + 1, WINDOW_SIZE)` — number of valid SWA-window positions for this token.
    - **Topk portion** (lines 569-579): load `topk_indices[token_idx, :PADDED_TOP_K]` (masked to `:topk_len`), add `M * batch_idx`, store at `combined_indices[token_idx, :topk_len]`.
    - **SWA portion** (lines 580-591): for each SWA slot `k ∈ [0, swa_len)`, write `combined_indices[token_idx, topk_len + k] = M * batch_idx + N + (k + pos - swa_len + 1 - gather_start)`. This maps SWA-window position `(pos - swa_len + 1 + k)` to its row in the gathered SWA-K buffer (which lives at `[N, M)` per request).
    - Store `combined_lens[token_idx] = topk_len + swa_len`.

For V4-Flash with `chunk_size = 4` and `NUM_WORKERS = 128`, the grid is `(4, 128) = 512` CTAs per `combine_topk_swa_indices` call.

## Math

Reference: `deepseek_v4/DeepSeek-V4-Flash/inference/model.py:507-515`:
```python
topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)
if self.compress_ratio:
    offset = kv.size(1) if start_pos == 0 else win
    if self.indexer is not None:
        compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
    else:
        compress_topk_idxs = get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset)
    topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
topk_idxs = topk_idxs.int()
```
The reference concatenates SWA window indices and compressed-pool topk indices into a single `topk_idxs` tensor passed to `sparse_attn` (line 528, 533). vLLM's `combine_topk_swa_indices` performs the same concatenation, additionally:
1. Resolving local indices to chunk-linearized-flat-KV-workspace ids (the `+ M*batch_idx + N + …` arithmetic), since vLLM gathers KV into a contiguous bf16 workspace rather than reading the paged cache in-kernel.
2. Padding the output to 128-alignment for `flash_mla_sparse_fwd`'s tiling requirements.

```python
# PyTorch-operator pseudocode for one CTA's work (batch_idx, worker_id ∈ [0, 128)).
# Inputs:
#   topk_indices            : [num_tokens, top_k] int32       (LOCAL compressed-pool ids)
#   query_start_loc[…]      : int32                            (chunk-relative after rebase)
#   seq_lens[b], gather_lens[b]: int32
#   M, N                    : ints, shared across all batches
#   COMPRESS_RATIO, WINDOW_SIZE, TOP_K: constexpr

base        = query_start_loc[0]       # rebase to chunk-local
qs          = query_start_loc[batch_idx] - base
qe          = query_start_loc[batch_idx + 1] - base
query_len   = qe - qs
seq_len     = seq_lens[batch_idx]
gather_len  = gather_lens[batch_idx]
start_pos   = seq_len - query_len            # abs pos of first new query token
gather_start= seq_len - gather_len           # abs pos of first SWA-gathered token

# Stripe across query tokens of this batch in this chunk.
for t in range(qs + worker_id, qe, 128):
    tq        = t - qs                       # token offset within this batch
    pos       = start_pos + tq               # absolute sequence position
    topk_len  = min((pos + 1) // COMPRESS_RATIO, TOP_K)   # valid compressed indices
    swa_len   = min(pos + 1, WINDOW_SIZE)                 # valid SWA window length

    # === Topk portion ===
    # Shift LOCAL compressed-pool indices into the chunk-flat workspace:
    # the b-th batch's compressed-pool gather lives at rows [M*b, M*b + N).
    combined_indices[t, :topk_len] = topk_indices[t, :topk_len] + M * batch_idx

    # === SWA portion ===
    # The SWA gather for batch b lives at rows [M*b + N, M*(b+1)).
    # Inside that span, gathered rows correspond to absolute positions
    # [gather_start, gather_start + gather_len).
    # For this query token, valid SWA window covers absolute positions
    # [pos - swa_len + 1, pos], whose buffer-row indices are
    # [N + (pos - swa_len + 1 - gather_start), N + (pos - gather_start)].
    for k in range(swa_len):
        combined_indices[t, topk_len + k] = M * batch_idx + N + (k + pos - swa_len + 1 - gather_start)

    combined_lens[t] = topk_len + swa_len

# All other slots in combined_indices are -1 (filled by the torch.full at allocation).
```

Notes on fusion:
- The concat-with-offset pattern is the vLLM analog of model.py:514 `torch.cat([topk_idxs, compress_topk_idxs], dim=-1)`. The differences:
  - vLLM puts **compressed** first then **SWA**, opposite of the reference (the reference does `[swa_topk, compressed_topk]`). This swap matches the workspace layout: rows `[0, N)` are compressed, rows `[N, M)` are SWA, so iterating "small absolute slot first" puts the compressed entries first.
  - vLLM prepends `M*batch_idx` to encode the batch axis as an index offset (since `flash_mla_sparse_fwd` has no batch dim).
  - vLLM rebases `query_start_loc` against its first entry, since the kernel receives a slice of the global `query_start_loc` but needs zero-based row indices into the chunk-local `combined_indices` allocation.
- Padding to `combined_topk` (multiple of 128) is **structural** — required by `flash_mla_sparse_fwd`'s SM100 tile size. Padding slots stay `-1`, which the FlashMLA kernel treats as "skip this attended position." `combined_lens` separately caps the valid range so the kernel doesn't even visit padding lanes (cheaper than masking, per FlashMLA docs).
- The `topk_len` formula `min((pos + 1) // COMPRESS_RATIO, TOP_K)` mirrors how both the C4A indexer and the C128A metadata-builder bound their emitted indices: at sequence position `pos`, at most `(pos + 1) // compress_ratio` compressed tokens can exist (you can't sample more than what has been compressed). The `TOP_K` cap is the index_topk hyperparameter. The kernel implementation **assumes** the producer emits exactly this many valid entries in the leftmost slots; trailing slots in `topk_indices[t]` are unused.
- `gather_start` adjustment: the SWA gather (`dequantize_and_gather_k_kernel`'s SWA call) writes only the **last** `gather_len` tokens (positions `[gather_start, seq_len)`). For chunked prefill, `gather_len` may exceed `swa_len` (because the gather covers `window_size + new_chunk_tokens`, while a single query within the chunk has a smaller per-token window). The `+ N + (k + pos - swa_len + 1 - gather_start)` term computes the correct row in that pre-gathered SWA buffer.

## Config-dependent dispatch

- Activation condition: `vllm/models/deepseek_v4/nvidia/flashmla.py:403` — **always** called per prefill chunk. SWA-only layers participate (with `top_k = 0`, `N = 0`); compressed-pool branches just set the right `topk_indices` source and `top_k` value.
- **Per-layer compress_ratio** routing of `topk_indices` (lines 336-355):
  - `compress_ratio == 4` (C4A): `topk_indices = layer.topk_indices_buffer[num_decode_tokens:][:num_prefill_tokens]`; `top_k = topk_indices.shape[-1]`. Indexer-produced.
  - `compress_ratio == 128` (C128A): `topk_indices = attn_metadata.c128a_prefill_topk_indices`; `top_k = topk_indices.shape[-1]`. Pre-computed at metadata build.
  - `compress_ratio <= 1` (SWA-only): `topk_indices = layer.topk_indices_buffer[num_decode_tokens:]` (placeholder, not used inside the kernel); `top_k = 0`; `N = 0`. The kernel's `topk_len = min((pos+1)//1, 0) = 0` zeroes the topk portion entirely.
- **`compress_ratio = 1` placeholder for SWA-only**: the `topk_indices` is still allocated (so the kernel doesn't IMA on the load), but `top_k = 0` and `topk_len = 0` mean no values are actually used. The placeholder shape just needs to be valid.
- **`window_size` and `compress_ratio`**: V4-Flash defaults (`window_size = 128`, `compress_ratio ∈ {0, 4, 128}` per `compress_ratios` config) bake them in as constexpr at kernel compile time.
- **`PADDED_TOP_K = next_power_of_2(topk_indices.shape[-1])`**: vector load width. For C4A's `topk = 2048`, this is 2048; for C128A's `topk` it varies.
- **`_SPARSE_PREFILL_TOPK_ALIGNMENT = 128`**: locked by the FlashMLA SM100 kernel's B_TOPK ∈ {64, 128} requirement (see source pointer at line 468-473). Padding to 128 satisfies both h_q=64 and h_q=128 codepaths.
- **Decode counterpart**: this kernel is the **prefill** analog of `compute_global_topk_indices_and_lens` (which is decode-only). Differences:
  - This kernel additionally combines with the SWA window indices into one tensor (decode passes SWA and topk as separate arguments to `flash_mla_with_kvcache`).
  - This kernel pads to 128 alignment (prefill kernel needs it); the decode kernel doesn't.
  - This kernel writes indices into a chunk-flat workspace (`+ M*batch_idx`); the decode kernel writes paged-global indices (block-table-resolved). Different downstream consumers (`flash_mla_sparse_fwd` reads a flat bf16 workspace; `flash_mla_with_kvcache` reads paged FP8 cache directly).
- **Pipeline note** (cross-kernel): per prefill chunk, the order is:
  1. `dequantize_and_gather_k_cache(...compressed_k_cache..., offset=0)` (line 373; skipped if SWA-only)
  2. `dequantize_and_gather_k_cache(...swa_k_cache..., offset=N)` (line 385)
  3. `combine_topk_swa_indices(...)` (this kernel; line 403)
  4. `flash_mla_sparse_fwd(...)` (line 416)
  The two gathers fill the `kv` workspace; this kernel emits indices into that workspace; FlashMLA consumes both.
