# `mla_decode_reduce` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** log-sum-exp merge of split-KV partials → final latent attention output.

**Phase:** decode.

**grid_dim:** internal (mirrors decode); block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `attn_partial` | `[T,Hd,512,n_split]` | bf16 | per-split context |
| `lse` | `[T,Hd,n_split]` | f32 | per-split log-sum-exp |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `attn_out` | `[T,Hd,512]` | bf16 | head-major latent context (→ BMM2 in [`bmm_fp8`](./bmm_fp8.md)) |

**Params:** `tp_degree`.

**Shape variants**

| tp_degree | Hd |
|---|---|
| 1 | 128 |
| 2 | 64 |
| 4 | 32 |
| 8 (this config) | 16 |

**Reuse:** `mla_mtp_reduce_layer` + `mla_mtp_decode_tp2/4/8_reduce_layer`.
