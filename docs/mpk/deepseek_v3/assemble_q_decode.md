# `assemble_q_decode` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** interleave the BMM1-absorbed latent query `q_latent[512]` with the roped
`q_pe[64]` into the per-head `[latent | rope]` layout `q[576]` that
[`mla_decode_attn`](./mla_decode_attn.md) consumes. With `pe_only` (the absorbed-decode path),
[`bmm_fp8`](./bmm_fp8.md) BMM1 already wrote the `[0:512]` region via a slice-view into the
output buffer, so this kernel only **copies the roped `q_pe` into the `[512:576]` tail** — a
thin layout task, not heavy compute.

**Phase:** decode.

**grid_dim:** `(T, 1, 1)` — **one CTA per token** (handles all H heads); grid.x partitions
the token dim; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `q_nope_abs` | `[T,H,512]` | bf16 | head-major; BMM1 output (latent query), typically a slice-view of `q[:,:,:512]` |
| `q_pe` | `[T,H,64]` | bf16 | head-major; roped query rope part |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `q` (`q_nope_pe`) | `[T,H,576]` (or `[T,H·576]`) | bf16 | per-head `[latent(512) \| rope(64)]`; feeds `mla_decode_attn` |

**Params:** `pe_only` (BMM path: nope already written in place → copy only the pe tail).

**Shape variants**

| tp_degree | Hd |
|---|---|
| 1 | 128 |
| 2 | 64 |
| 4 | 32 |
| 8 (this config) | 16 |

**Tensor-view requirement (MUST):** the output `q` is the parent `[T,H,576]` buffer and
`q_nope_abs` is its own `[:,:,:512]` slice-view (BMM1 already wrote there). The kernel copies the
roped `q_pe` into the `[512:576]` tail **honoring the 576 per-head row stride** — an in-place
strided write into the parent, not a fresh contiguous buffer.

**Reuse:** `assemble_q_decode_sm100_layer`.

**Note:** decode-only. The prefill path needs no assembly — its query is the single committed
`q[H·192]` buffer (see [`mla_prefill_attn`](./mla_prefill_attn.md)). Could potentially be folded
into the BMM1 epilogue (write latent + place pe in one pass), but it's a registered task today.
