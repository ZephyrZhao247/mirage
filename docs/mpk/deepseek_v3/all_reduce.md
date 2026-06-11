# `all_reduce` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** sum a tensor across all TP×EP ranks (NVSHMEM); optional fused residual at
final store. This is what synchronizes routed-expert (EP) and tensor-parallel (TP)
contributions — replacing explicit dispatch/combine.

**Phase:** both.

**grid_dim:** `(H/128,1,1) = (56,1,1)`; grid.x tiles `H`; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `x` | `[T,H]` | bf16 | local partial (post-o_proj or post-MoE) |
| `residual` (opt) | `[T,H]` | bf16 | fused at final store |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `y` | `[T,H]` | bf16 | globally summed |
| `buffer` | `[W=8,T,H]` | bf16 | scratch (per-rank staging) |

**Params:** `world_size=8`, `rank`, opt `gate_mode`.

**Shape variants**

| call site | shape |
|---|---|
| after o_proj | `[T, H=7168]` |
| after MoE / dense MLP | `[T, H=7168]` |

**Reuse:** `allreduce_layer`.
