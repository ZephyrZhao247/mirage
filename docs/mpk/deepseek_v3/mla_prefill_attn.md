# `mla_prefill_attn` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Status: NEW kernel** (replaces the reused `mla_prefill_tp8_chunked_layer`). Designed
interface-first to fit its neighbors with minimal glue — see *Design notes*.

**Semantics:** causal MLA attention over the prompt sequence, **unabsorbed** per-head
(`S = Qn·Knᵀ + Qp·Krᵀ`, online softmax, `O = P·V`). Distinct type from
[`mla_decode_attn`](./mla_decode_attn.md). Single-chunk by default; longer prompts chunk via
`Q_START` (see *Prefill length*).

**Phase:** prefill.

**grid_dim:** `(H, ceil(q_len/BM), num_requests)`, `BM=64`; block `(256,1,1)` (128 active
threads = 4 warps). Distribution: **grid.x = head** (one head/CTA), **grid.y = 64-row query
tile**, **grid.z = request**. KV is looped inside each CTA in `BN=128` tiles up to the causal
bound; TMA double-buffers K/V. Over-provisioned `(head, q-tile, request)` tasks early-exit.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `q` | `[T, H·192]` | bf16 | **ONE buffer**, committed `[all-nope (H·128) \| all-pe (H·64)]` layout — the q_b FP8-GEMM-natural order. RoPE already applied in-place to the pe region. Head `h`: nope at col `h·128`, pe at col `H·128 + h·64`. |
| `kv` | `[L, H, 256]` | bf16 | **ONE buffer** `[k_nope (128) \| v (128)]` per head, from the **fused `kv_b` GEMM**. Head-major. |
| `k_rope` | `[L, 64]` | bf16 | shared rope key, **read directly from the gather** (`kpe`). |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `o` | `[T, H, 128]` | bf16 | per-head attention output; contiguous → viewed as `[T, H·128]` by the downstream `quantize_fp8` → `o_proj` (`linear_fp8`). |

**Params:** `tp_degree` (this config: 8 → `Hd=H=16`). Per-request `Q_LEN`/`KV_LEN`/`Q_START`
come from the `qo_indptr` / `paged_kv_indptr` meta-tensors at runtime (the grid is sized for
the max; tasks read their request's actual lengths and early-exit). No `q_len` kernel param.

**Shape variants**

| tp_degree | Hd |
|---|---|
| 1 | 128 |
| 2 | 64 |
| 4 | 32 |
| 8 (this config) | 16 |

**Reuse:** **new** — borrow the attention core (flash-style `Qn·Kn + Qp·Kr`, online softmax,
`BM=64`/`BN=128` tiling, TMA double-buffer) from `mla_prefill_tp8_chunked_sm100.cuh`, but with
the cleaned interface below. The absorbed `mla_prefill_absorbed_layer` (latent-space, BMM1/BMM2)
is **not used** (heavier attention for large `q_len`).

## Design notes — why this interface (minimal glue)

Adopted folds: **fused `kv_b` GEMM** (one `[L,H,256]` kv buffer instead of two `kv_b_k`/`kv_b_v`
GEMMs + two inputs) and **single committed Q layout** (drops `qfused_mode` + the 4 stride params;
the `[all-nope|all-pe]` buffer is THE contract, not a mode). q_nope/q_pe merge into one `q` input.

Eliminated vs the old chunked kernel: `qfused_mode` + `qn/qp_head/row_stride` params; the
`q_nope`+`q_pe` split; one of the two `kv_b` GEMMs + the `k_nope`/`v` input split.

**Retained (folds *not* adopted):**
- Output is **bf16**, so a separate [`quantize_fp8`](./quantize_fp8.md) sits between this and
  `o_proj` (FP8-output epilogue not folded in).
- The gather emits **bf16 `ckv`**, so a `quantize_fp8` precedes the fused `kv_b` GEMM
  (gather-side FP8 not folded in).

**Pipeline around this kernel:**
```
gather(→ ckv[L,512] bf16, kpe[L,64] bf16)
  → quantize_fp8(ckv) → fused kv_b GEMM → kv [L,H,256] bf16
q_b GEMM → q [T,H·192] → rope(in-place pe region)
  → mla_prefill_attn(q, kv, k_rope=kpe) → o [T,H,128] bf16
  → quantize_fp8 → o_proj(linear_fp8, +residual) → all_reduce
```

**Prefill length:** the kernel has no intrinsic cap (q tiled over grid.y, KV looped). Bound is
`max_num_batched_tokens` for the per-iteration `q_len` and `max_seq_length` for `kv_len`.
Prompts longer than `max_num_batched_tokens` are chunked across iterations via `Q_START`
(the prefix grows in the paged cache) — same kernel, multiple calls.

**Open / non-interface:** the `kpe → kpe_v2` phantom-bridge `identity_layer` in the old path is
an MPK **scheduling** artifact (the gather is a fork+join producer → case-3). It is *not* part
of this contract; resolve it with a proper join-event design when wiring, not by reshaping the
kernel I/O.
