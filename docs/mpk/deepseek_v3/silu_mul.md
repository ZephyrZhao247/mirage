# `silu_mul` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** `o = silu(a[:,:I]) · a[:,I:]`.

**Phase:** both.

**grid_dim:** dense `(silu_grid,1,1)` (grid.x tiles columns); MoE `(min(num_workers,Mtot),1,1)`
(grid.x tiles permuted rows = experts); block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `a` | `[*,2I]` | bf16 | gate‖up concatenated (dense `[T,2I]`; MoE `[Mtot,2I]`) |
| `meta` (opt) | `[2,*]` | int32 | MoE active-expert mask (skip inactive blocks) |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `o` | `[*,I]` | bf16 | silu(gate)·up |

**Params:** `I`, `meta_present`.

**Shape variants**

| use | I | input shape | grid | notes |
|---|---|---|---|---|
| dense MLP | 2304 (`inter_dense/8`) | `[T, 2I]` | column-split | layers 0–2 |
| shared expert | 256 (`inter_moe/8`) | `[T, 2I]` | column-split | |
| MoE routed | 512 (`I_r`) | `[Mtot, 2I]` | row-split | + `meta` active-mask |

**Reuse:** `silu_mul_layer` (dense) / `moe_silu_mul_layer` (MoE). Different grid → Python-level select.
