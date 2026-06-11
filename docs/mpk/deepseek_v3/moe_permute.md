# `moe_permute` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** gather this rank's routed tokens into expert-contiguous, BM-padded order;
emit grouped-GEMM metadata. **Local only** — no cross-GPU exchange (EP is handled by the
downstream [`all_reduce`](./all_reduce.md)).

**Phase:** both.

**grid_dim:** `(E_loc/e_per_cta, 1, 1) = (128,1,1)` (epc=1); block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `x_fp8` | `[T,H]` | e4m3 | quantized hidden |
| `x_scale` | — | uint32/f32 | activation scale |
| `topk_weight` | `[T,Ep]` | f32 | from router |
| `routing_idx` | `[E_loc,T]` | int32 | from router |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `permuted_fp8` | `[Mtot,H]` | e4m3 | expert-contiguous tokens, `Mtot = E_loc·bm_padding = 128·128 = 16384` |
| `permuted_scale` | — | uint32/f32 | packed for grouped GEMM |
| `meta` | `[2, Mtot+…]` | int32 | `m_indices` (row→expert) + active-mask + weights |

**Params:** `E_loc=128`, `Ep`, `bm_padding=128`, `e_per_cta`.

**Shape variants**

| variant | dims |
|---|---|
| config-fixed | `E_loc=128, bm_padding=128` → `Mtot=16384`; `H=7168` |

**Reuse:** `moe_permute_sm100_layer`.
