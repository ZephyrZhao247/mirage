# save_partial_states

## Identity
- Source file: `vllm/models/deepseek_v4/common/ops/save_partial_states.py:48-102` (Triton JIT body `_save_partial_states_kernel`); user-facing entry `save_partial_states` at lines 9-45.
- Launch wrapper: same file, `save_partial_states` Python function (lines 9-45) directly launches the kernel with a 1-D grid sized `(num_actual,)`.
- Language/DSL: **Triton** (`@triton.jit`).
- Third-party dep: none (pure Triton + PyTorch).
- Registered as opaque custom op: **no** — called directly as a Python function from the compressor forward, not exposed under `torch.ops.vllm.*`.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/compressor.py:314` | `DeepseekCompressor.forward` (after splitting `kv_score` into `kv` and `score`) | `kv: [num_tokens, coff*head_dim]`, `score: [num_tokens, coff*head_dim]`, `ape: [compress_ratio, coff*head_dim]`, `positions: [num_tokens] int64`, `state_cache: [num_blocks, block_size, 2*coff*head_dim] fp32`, `slot_mapping: [num_tokens] int64` | fp32 throughout (`kv`/`score`/`ape` already up-cast in `forward`) | always on; runs once per compressor forward (both attention compressor with `head_dim=512` and indexer compressor with `head_dim=128`) |

Single call site. `head_size = kv.shape[-1] = coff * head_dim` (`coff = 1 + overlap`; `coff=2` when `compress_ratio==4`, `coff=1` when `compress_ratio==128`). For DeepSeek V4-Flash attention compressor: `head_size=1024` (ratio=4) or `512` (ratio=128); for the indexer compressor (head_dim=128): `head_size=256` (ratio=4 only — indexer is only built when ratio==4 per `attention.py:467-471` of `model.py`).

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `kv` | `[num_tokens, head_size]` | fp32 (already up-cast in `DeepseekCompressor.forward` line 281; bf16 raw output of the wkv GEMM up-cast at `compressor.py:282` `kv_score = kv_score.split(...)` after fp32 wkv) | row-major, contiguous on last dim | Compressed KV state proposal at each token (per token, one `coff*head_dim`-wide vector). Written verbatim into the `[0, STATE_WIDTH)` half of the slot. |
| `score` | `[num_tokens, head_size]` | fp32 | row-major, contiguous on last dim | Gating-score proposal at each token. Will be summed with positional embedding `ape[position % compress_ratio]` before storing. |
| `ape` | `[compress_ratio, head_size]` | fp32 | row-major | Absolute-position embedding shared across the compression window. Indexed by `position % compress_ratio`. |
| `positions` | `[num_tokens]` | int64 | contiguous | Per-token global position used to pick the APE row. |
| `state_cache` | `[num_blocks, block_size, 2*head_size]` | fp32 | strided 3-D; last dim packs `[kv_state \| score_state]` each `STATE_WIDTH=head_size` wide | Compressor state cache (per-request `CompressorStateCache.kv_cache`). |
| `slot_mapping` | `[num_tokens]` | int64 | contiguous | Per-token slot index into the flattened cache; `-1` marks padded/invalid tokens. |
| `block_size` | scalar | int | — | Tokens per cache block (the cache layout is paged). |
| `STATE_WIDTH` | scalar (constexpr) | int | — | `state_cache.shape[-1] // 2`, equals `head_size`. Splits the last dim into `[kv \| score]`. |
| `COMPRESS_RATIO` | scalar (constexpr) | int | — | `compress_ratio` (V4-Flash: 4 or 128 per-layer). Used only to fold `position % compress_ratio` into APE lookup. |
| `HEAD_SIZE` | scalar (constexpr) | int | — | `kv.shape[-1]`. |
| `TRITON_BLOCK_SIZE` | scalar (constexpr) | int | — | `triton.next_power_of_2(head_size)` — load width per program, masked to `HEAD_SIZE`. |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `state_cache` (in-place) | `[num_blocks, block_size, 2*head_size]` | fp32 | as above | For each non-padded token, writes `kv` into `[0, STATE_WIDTH)` and `score + ape[position % compress_ratio]` into `[STATE_WIDTH, 2*STATE_WIDTH)` of the slot `slot_mapping[token_idx]`. Padded tokens (`slot_id < 0`) are skipped entirely. |

No tensor is returned (the wrapper is `-> None`).

## Grid / Block

- `grid_dim = (num_actual,)` — one CTA per *actual* token (`num_actual = slot_mapping.shape[0]`).
- `block_dim` (threads/CTA): **default Triton** — no explicit `num_warps`/`num_threads` override at the launch site. Triton picks 4 warps by default for this body (single 1-D load/store with no reductions).
- Autotune configs: **none**.
- PDL: caller may pass `launch_pdl=False` via `pdl_kwargs` (set by `compressor.py:302-306` on CUDA/XPU). PDL is intentionally disabled because both this kernel and the downstream compress kernels depend on its output without using grid-dependency primitives (`compressor.py:308-313` comment).
- Per-CTA work: load one `[HEAD_SIZE]` slice of `kv` and one of `score`, look up `ape[position % COMPRESS_RATIO]` and one `position`, write `kv` and `score + ape` into the slot.
- Mask handling: `block < HEAD_SIZE` masks the rounded-up `TRITON_BLOCK_SIZE` load.
- Early-exit: if `slot_id < 0` the CTA returns immediately (padded tokens in CUDA-graph replay).

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:331-357`. The reference's two paths (`start_pos==0` prefill, `start_pos>0` decode) both write `kv_state` and `score_state` for non-compressed positions. The vLLM fused kernel collapses both paths to "per-token slot-mapped write of kv + (score+ape)"; the prefill/decode distinction lives entirely in the caller's `slot_mapping`/`positions` builders.

```python
# Per-CTA work for one token t (PyTorch-operator equivalent):
slot_id = slot_mapping[t]
if slot_id < 0:
    return  # padded token (CUDA graph replay)

block_idx     = slot_id // block_size
pos_in_block  = slot_id %  block_size

# kv-half write (straight pass-through)
state_cache[block_idx, pos_in_block, : STATE_WIDTH] = kv[t]

# score-half write fuses APE add
pos = positions[t]
ape_row = pos % COMPRESS_RATIO
state_cache[block_idx, pos_in_block, STATE_WIDTH : 2*STATE_WIDTH] = (
    score[t] + ape[ape_row]
)
```

Notes on fusion:
- The reference Compressor (model.py:332, 335, 338, 345, 348, 357) writes `score_state[..., r] = score[..., r] + self.ape` (with `r = start_pos % ratio` in decode, or broadcast over the window in prefill). This kernel fuses that add directly into the slot write — no separate "+= ape" pass is needed downstream.
- The kv-half and score-half are stored *concatenated along the last dim*. Downstream compress kernels (`_fused_kv_compress_norm_rope_insert_*`) re-fetch this slot as a single `row_base + STATE_WIDTH + block` for the score half, and `row_base + block` for the kv half.
- `STATE_WIDTH == HEAD_SIZE == coff * head_dim`, so the cache slot is `2 * coff * head_dim` fp32 elements wide (e.g. 4096 fp32 = 16 KiB per token for V4-Flash attention compressor with `head_dim=512, coff=2`).
- `head_size` is passed dynamically (Python int) but `STATE_WIDTH`/`HEAD_SIZE` reach the kernel as `tl.constexpr`, so the kernel specializes per `head_size` (Triton auto-caches one variant per compressor instance).

## Config-dependent dispatch

- Activation condition: always on (single path). The caller invokes this unconditionally before `compress_norm_rope_store_{triton,cutedsl}`.
- Variants: **none on NVIDIA**. The same kernel runs on Ampere/Hopper/Blackwell — no SM-specific code path. On AMD/XPU the only difference is PDL disabling via `pdl_kwargs` (see "Grid / Block").
- Class: **A** (locked path; no Class B sibling).
- Downstream consumer constraint: the score-half write (`STATE_WIDTH + block`) is read back as the softmax-input by the three `_fused_kv_compress_norm_rope_insert_*` kernels in `fused_compress_quant_cache.py`. The fused `+ ape` here must match how those kernels expect the score laid out — any change to the APE-add semantics must be coordinated with all three compress kernels.
