# `grouped_gemm_fp8` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** grouped block-scaled FP8 GEMM; permuted row group `g` uses local-expert weight `b[g]`.

**Phase:** both.

**grid_dim:** `(num_workers,1,1)`, persistent tile distribution (smallm/largem auto-select);
block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `a_fp8` | `[Mtot,K]` | e4m3 | permuted activations |
| `b_fp8` | `[E_loc,N,K]` | e4m3 | per-local-expert weights |
| `a_scale` | — | uint32/f32 | block scales |
| `b_scale` | — | f32 | block scales |
| `m_indices` | `[Mtot]` | int32 | row→expert map |
| `meta` (opt) | — | int32 | active-mask |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `C` | `[Mtot,N]` | bf16 | grouped result |

**Params:** `num_workers`.

**Shape variants** (`E_loc=128` local experts; `Mtot=16384`):

| role | K | N |
|---|---|---|
| w13 (gate+up) | 7168 | 1024 (`=2·I_r`) |
| w2 (down) | 512 (`I_r`) | 7168 |

**Reuse:** `fp8_group_gemm_layer`.
