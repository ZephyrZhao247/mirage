# flash_mla_sparse_fwd

## Identity
- Source file: external — compiled C++/CUDA in `vllm._flashmla_C` (built into `_flashmla_C.abi3.so`, not in this tree). Underlying op: `torch.ops._flashmla_C.sparse_prefill_fwd`.
- Python wrapper: `vllm/third_party/flashmla/flash_mla_interface.py:180-217` (function `flash_mla_sparse_fwd`).
- Re-export shim: `vllm/v1/attention/ops/flashmla.py:87-95`; availability gated by `_is_flashmla_available()` (lines 33-48). When unavailable a stub at line 104 raises.
- Language/DSL: **External CUDA** (NVIDIA FlashMLA library — `deepseek-ai/FlashMLA`).
- Third-party dep: `deepseek-ai/FlashMLA` (vendored under `vllm/third_party/flashmla/`).
- Registered as opaque custom op: yes — invoked through `torch.ops._flashmla_C.sparse_prefill_fwd` inside the wrapper.
- Sibling: `flash_mla_with_kvcache` (decode) — see `flash_mla_with_kvcache.md`.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/nvidia/flashmla.py:416` | `DeepseekV4FlashMLASparseImpl._forward_prefill` | `q[query_start:query_end]: [num_prefill_tokens_chunk, h_q ∈ {64,128}, 576]`, `kv.view(-1, 1, 576): [PREFILL_CHUNK_SIZE * M, 1, 576]`, `combined_indices.unsqueeze(1): [num_prefill_tokens_chunk, 1, combined_topk]`, `combined_lens: [num_prefill_tokens_chunk]`, `output[query_start:query_end]: [num_prefill_tokens_chunk, h_q, 512]` | q bf16; kv bf16 (gathered+dequantized); indices int32; lens int32; output bf16 | Prefill-only (`num_prefills > 0`). Called once per chunk of `PREFILL_CHUNK_SIZE=4` requests inside the prefill loop. |

Single live caller on NVIDIA. Looped over `num_chunks = (num_prefills + 4 - 1) // 4` chunks per attention forward.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `q` | `[s_q, h_q, d_qk=576]` (V4-Flash: `s_q = num_prefill_tokens_in_chunk`, `h_q ∈ {64, 128}`) | bf16 | contiguous on `d_qk` | Query — already RMS-normed and RoPE-rotated by `_q_proj_path`. **No batch dimension**: per FlashMLA docs, multi-request prefill is encoded by per-token indices, not a batch axis. V4-Flash linearizes all `chunk_size` requests' query tokens into the `s_q` dim. |
| `kv` | `[s_kv, h_kv=1, d_qk=576]` (V4-Flash: `s_kv = PREFILL_CHUNK_SIZE * M = 4 * M`) | bf16 | contiguous on `d_qk` | Already-dequantized + gathered K (and V via slicing) buffer. Built by repeated `dequantize_and_gather_k_cache` calls (one for compressed pool, one for SWA pool) into a single `kv` workspace of shape `[chunk_size, M, 576]` then `.view(-1, 1, 576)` flattens chunk + token dims so each batch's KV lives at slot offset `M * batch_idx + position`. See `combine_topk_swa_indices` for how `indices` encodes this offset. |
| `indices` | `[s_q, h_kv=1, topk]` (V4-Flash: `topk = combined_topk`, padded to 128-alignment) | int32 | contiguous | Per-query KV slot ids. Invalid entries are `-1` or `>= s_kv`. Built by `combine_topk_swa_indices` (see that spec) — concatenates `topk_indices + M * batch_idx` with SWA window indices `M * batch_idx + N + (window positions)`, both shifted into the per-chunk linearized `kv` buffer. |
| `sm_scale` | scalar | float | — | `layer.scale = 1/sqrt(576)`. |
| `d_v` | scalar `512` | int | — | Output V-projection dim. Locked at 512. |
| `attn_sink` | `[h_q]` or None | fp32 | — | Per-head logit sink (same semantics as decode). V4-Flash passes `layer.attn_sink`. |
| `topk_length` | `[s_q]` or None | int32 | — | Per-query valid count for `indices`. V4-Flash passes `combined_lens` from `combine_topk_swa_indices`. Avoids attending to the padding `-1` entries. |
| `out` | `[s_q, h_q, d_v=512]` or None | bf16 | contiguous on last dim | Pre-allocated output buffer (= the slice of the caller's `output` for this chunk). |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `out` | `[s_q, h_q, 512]` | bf16 | contiguous on last dim | Attention output. Written in-place into caller's pre-allocated `output[query_start:query_end]`. |
| `max_logits` | `[s_q, h_q]` | fp32 | — | Per-(query, head) pre-softmax max logit. Returned but ignored by V4-Flash (`flash_mla_sparse_fwd(...)` discards the return value entirely — the line just calls the function, doesn't assign). |
| `lse` | `[s_q, h_q]` | fp32 | — | Log-sum-exp of attention scores per (query, head). Returned but ignored by V4-Flash. |

## Grid / Block

External CUDA — exact `gridDim` / `blockDim` not in this tree. Recoverable contracts:

- **SM target**: SM90 (Hopper) and SM100 (Blackwell). V4-Flash on B200 dispatches to **SM100**. Selection internal to the `.abi3.so`.
- **CUDA / nvcc**: per FlashMLA README, SM100 kernels require CUDA ≥ 12.9.
- **No batch dim**: the kernel does NOT take a batch axis. V4-Flash linearizes `[chunk_size, M, 576] → [chunk_size * M, 1, 576]` and embeds the batch boundary into `indices` (`combine_topk_swa_indices` adds `M * batch_idx` to every emitted index).
- **B_TOPK alignment**: per source pointer cited at `vllm/models/deepseek_v4/common/ops/cache_utils.py:468-473`:
  > FlashMLA sparse prefill asserts `params.topk % B_TOPK == 0` (see `flashmla/csrc/sm100/prefill/sparse/fwd/head{64,128}/phase1.cuh`). B_TOPK is **64** for the h_q=64 kernel and **128** for the h_q=128 kernel; pad to **128** to satisfy both.
  V4-Flash always pads `topk` to a multiple of 128 via `_SPARSE_PREFILL_TOPK_ALIGNMENT = 128` in `combine_topk_swa_indices`. Padding slots are `-1`; `topk_length` caps the valid range.
- **Per-tile-CTA work**: phase1 / phase2 split (per the cited source-tree path `flashmla/csrc/sm100/prefill/sparse/fwd/head{64,128}/phase1.cuh`). Phase1 produces partial QK + softmax-numerator/denominator per topk tile; phase2 reduces. Each query token's `topk` valid set is walked in `B_TOPK`-sized tiles. The h_q=64 vs h_q=128 kernels are **separate compiled binaries**; selection is by `q.shape[1]` inside `.abi3.so`. V4-Flash will use h_q=64 (default `n_heads=64`); TP runs may end up with the h_q=128 kernel.
- **Hard shape constraints**:
  - `d_qk == q.shape[-1] == kv.shape[-1]` (576 for V4-Flash).
  - `d_v == 512`. Locked.
  - `h_kv == 1` (kv shape's middle dim).
  - `h_q ∈ {64, 128}`. V4-Flash uses `get_padded_num_q_heads` to enforce.
  - `topk % B_TOPK == 0` (B_TOPK ∈ {64, 128}); pad to 128 in callers.
  - **`kv` must be contiguously valid across `[0, s_kv * 576 * 2)` bytes**: per FlashMLA docs, "the KV cache must be contiguously valid for sparse attention on sm100. Here 'contiguously valid' means that every byte, from the very beginning of the KV cache, till the last byte in the KV cache, is valid memory address to visit (i.e. won't IMA)." V4-Flash guarantees this by gathering into a flat workspace via `dequantize_and_gather_k_cache`.
- **Memory ordering**: kernel uses TMA loads on SM100 (head=576 needs multiple TMA boxes per query).
- **Performance**: per FlashMLA published numbers, up to **640 TFLOPS** on sparse prefill (H800 SXM5). SM100 numbers not published in README.

## Math

Reference: `deepseek_v4/DeepSeek-V4-Flash/inference/model.py:528` — the call site `o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)` (prefill branch, `start_pos == 0`). Same `sparse_attn` symbol as decode at line 533; the V4-Flash reference uses one operator for both phases, while vLLM splits into `flash_mla_with_kvcache` (decode) vs `flash_mla_sparse_fwd` (prefill) for kernel-shape reasons.

```python
# PyTorch-operator pseudocode mirroring flash_mla_sparse_fwd's fusion (per-query).
# Inputs (V4-Flash, one prefill-chunk):
#   q[t, h]         ∈ R^576              (per-token query, RoPE applied to last 64)
#   kv[s, 0]        ∈ R^576              (gathered+dequantized K/V pool, bf16)
#   indices[t,0,:]  ∈ Z^topk_padded      (kv slot ids into the flat pool; -1 = invalid)
#   topk_length[t]  ∈ Z                  (count of valid leftmost indices entries)
#   attn_sink[h]    ∈ R                  (optional per-head logit sink)
#   sm_scale        = 1/sqrt(576)
#
# Per query token t, per head h:
valid_idx = indices[t, 0, :topk_length[t]]            # [L_valid]
# (Entries past topk_length are ignored even if not -1.)
mask      = (valid_idx >= 0) & (valid_idx < kv.shape[0])
K_t       = kv[valid_idx.clamp(min=0), 0, :]          # [L_valid, 576] bf16
                                                       # invalid lanes will be masked out
logits    = (q[t, h] @ K_t.T) * sm_scale              # [L_valid]
logits    = torch.where(mask, logits, float('-inf'))  # mask invalid
lse       = torch.logsumexp(logits, dim=-1)           # scalar
p         = torch.softmax(logits, dim=-1)             # [L_valid]
V_t       = K_t[:, :512]                              # MLA: V is K's first 512 lanes
out_th    = p @ V_t                                   # [512]

# Optional per-head sink mixing.
if attn_sink is not None:
    out_th = out_th * (torch.exp(lse) / (torch.exp(lse) + torch.exp(attn_sink[h])))

out[t, h, :] = out_th.to(torch.bfloat16)
```

Notes on fusion:
- The kernel internally walks `indices[t]` in `B_TOPK`-sized phase-1 tiles, computing partial softmax numerators/denominators, and reduces in phase 2. The "online softmax" pattern is the same as `flash_mla_with_kvcache`, but with bf16 KV (already dequantized) instead of FP8 inside the kernel — that's the trade-off: prefill amortizes dequant once (via `dequantize_and_gather_k_cache`) instead of doing it inside the kernel per-token.
- Multi-request encoding via index offsets: V4-Flash's caller pre-shifts every emitted index by `M * batch_idx` (`combine_topk_swa_indices` does this). The kernel itself sees a flat KV pool of size `PREFILL_CHUNK_SIZE * M`.
- The `-1` and out-of-range sentinels are handled by the kernel (per FlashMLA docs: "invalid entries should be set to -1 or numbers >= s_kv"). V4-Flash's `combine_topk_swa_indices` uses both: `-1` for padding past `combined_topk`, and `topk_length` to cap the valid range without rewriting indices.
- **NaN warning** (from FlashMLA docs lines 203-204): if `topk_length` is provided and an index past `topk_length[i]` points to a K row containing NaN, output may contain NaN. V4-Flash sidesteps this by ensuring trailing slots are `-1`.

## Config-dependent dispatch

- Activation condition: `vllm/models/deepseek_v4/nvidia/flashmla.py:416` whenever `num_prefills > 0`. Loops over chunks of size `PREFILL_CHUNK_SIZE = 4`.
- **Per-layer compress_ratio** branches in V4-Flash `_forward_prefill` (lines 336-415) — affects what `topk_indices` source is used, but the `flash_mla_sparse_fwd` call signature itself is the same:
  - `compress_ratio <= 1` (SWA-only): `topk_indices = layer.topk_indices_buffer[num_decode_tokens:]` (placeholder; `top_k = 0`, so `combined_topk` only contains the SWA window portion). The compressed-pool `dequantize_and_gather_k_cache` call is **skipped** (`if not swa_only:` guard at line 369).
  - `compress_ratio == 4` (C4A): `topk_indices = layer.topk_indices_buffer[num_decode_tokens:][:num_prefill_tokens]` (filled by the indexer kernel earlier in the layer). Both compressed and SWA `dequantize_and_gather_k_cache` calls run.
  - `compress_ratio == 128` (C128A): `topk_indices = attn_metadata.c128a_prefill_topk_indices` (pre-computed at metadata build, deterministic from positions). Both gathers run.
- **`h_q ∈ {64, 128}` kernel select**: separate compiled kernels; chosen inside `.abi3.so` based on `q.shape[1]`. V4-Flash's `get_padded_num_q_heads` snaps to one of these.
- **`attn_sink`**: V4-Flash always passes `layer.attn_sink`. The kernel handles `None` by skipping the sink-mix step.
- **`topk_length`**: always provided by V4-Flash (= `combined_lens`). Skipping it would force the kernel to attend the full padded `combined_topk`, wasting work on the `-1` slots.
- **`d_v == 512`**: hard-locked. Other `d_v` values not supported in this kernel.
- **`PREFILL_CHUNK_SIZE = 4`**: chosen by V4-Flash to bound the bf16 workspace `(4, M, 576)` — see `vllm/models/deepseek_v4/nvidia/flashmla.py:52`. Not a kernel parameter; the kernel sees one chunk at a time as `[s_q, h_q, 576]`.
- **SM target**: SM100 (Blackwell, V4-Flash on B200). SM90 (Hopper) supported but a different compiled binary in the same `.abi3.so`. Selection internal to the kernel.
- **Pipeline note** (cross-kernel): the call must be preceded for the same chunk by:
  1. `dequantize_and_gather_k_cache(kv[:chunk_size], compressed_k_cache, ...)` at line 373 (skipped if `swa_only`),
  2. `dequantize_and_gather_k_cache(kv[:chunk_size], swa_k_cache, ..., offset=N)` at line 385 (always),
  3. `combine_topk_swa_indices(...)` at line 403.
  See `dequantize_and_gather_k_kernel.md` and `combine_topk_swa_indices.md`.
- **Alternative paths in vLLM that are NOT this caller** (locked pointers):
  - Dense prefill: `flash_attn_varlen_func` (also from FlashMLA) — not used by V4-Flash sparse path.
  - V3.2 sparse prefill backend: `vllm/v1/attention/backends/mla/flashmla_sparse.py:985` — same kernel, different caller for non-V4 models.
  - ROCm prefill: `vllm/models/deepseek_v4/amd/rocm.py:656` (`_forward_prefill`) — out of scope.
