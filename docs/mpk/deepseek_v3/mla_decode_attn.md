# `mla_decode_attn` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** latent-space MLA attention vs paged cache, split over KV length; emits
per-split partials + log-sum-exp. (Distinct type from [`mla_prefill_attn`](./mla_prefill_attn.md).)

**Phase:** decode.

**grid_dim:** internal to kernel (scales with `Hd`, `num_splits`); block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `q` | `[T,Hd,576]` | bf16 | head-major; absorbed query `[latent\|rope]`, produced by [`assemble_q_decode`](./assemble_q_decode.md) (BMM1 latent + roped `q_pe`); TMA |
| `kv` (paged) | `[L,576]` | bf16 | gathered latent KV |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `attn_partial` | `[T,Hd,512,n_split]` | bf16 | per-split context |
| `lse` | `[T,Hd,n_split]` | f32 | per-split log-sum-exp |

**Params:** `tp_degree` (→`Hd`=128/64/32/16; tp4 v-split, tp8 q-pad), `n_split`.

**Shape variants**

| tp_degree | Hd |
|---|---|
| 1 | 128 |
| 2 | 64 |
| 4 | 32 |
| 8 (this config) | 16 |

**Reuse:** `mla_mtp_decode_layer` + `mla_mtp_decode_tp2/4/8_layer` (collapse via `tp_degree`).
