# `argmax` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** greedy token = argmax over vocab (2-stage: per-worker partial → reduce).

**Phase:** both.

**grid_dim:** partial `(num_workers,1,1)=(128,1,1)` (grid.x splits vocab); reduce `(1,1,1)`; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `logits` | `[T,Vloc]` | bf16 | row-major; `Vloc=V` (replicated) or `V/8` (sharded) |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `value` (partial) | `[T,nt]` | bf16 | per-worker max value |
| `index` (partial) | `[T,nt]` | int32 | per-worker argmax index |
| `token` (reduce) | `[T,1]` | int64 | final greedy token |

**Params:** `num_tasks`.

**Shape variants**

| variant | Vloc | notes |
|---|---|---|
| full vocab (replicated) | 129280 | single-GPU |
| vocab-sharded | 16160 (`V/8`) | → [`global_argmax`](./global_argmax.md) |

**Reuse:** `argmax_partial_layer` + `argmax_reduce_layer`.
