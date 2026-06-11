# `linear_bf16` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** `C = A·Wᵀ (+residual)`, bf16.

**Phase:** both.

**grid_dim:** `(N/128, 1, 1)`; split-K `(N/128, split_k, 1)`; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `A` | `[M,K]` | bf16 | row-major activation (M=T) |
| `W` | `[N,K]` | bf16 | row-major weight |
| `residual` (opt) | `[M,N]` | bf16 | added when `with_residual` |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `C` | `[M,N]` | bf16 | row-major |

**Params:** `M,N,K`, `with_residual`, `tp_role`, `split_k`.

**Shape variants**

| role | K | N | notes |
|---|---|---|---|
| lm_head (vocab-sharded) | 7168 | `V/8=16160` | column → [`global_argmax`](./global_argmax.md) |
| lm_head (replicated) | 7168 | `V=129280` | → [`argmax`](./argmax.md) |
| bf16 residual/accum fallback | — | — | as needed |

**Reuse:** `linear_layer`, `linear_with_residual_layer`, `splitk_linear_layer`.
