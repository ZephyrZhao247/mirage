# `rope` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** apply rotary embedding to the decoupled rope dims, in-place.

**Phase:** both.

**grid_dim:** Q `(T,1,1)`; K `(num_requests,1,1) = (4,1,1)`; block `(256,1,1)`. No tensor
partition — every CTA sees the full slice.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `x` | `[T,*]` | bf16 | in-place Q-rope or K-rope slice (narrow view of qkv buffer) |
| `cos` | `[T,qk_rope]` | bf16 | rotary cos table |
| `sin` | `[T,qk_rope]` | bf16 | rotary sin table |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `x` | `[T,*]` | bf16 | rotated **in-place** |

**Params:** `role∈{Q,K}`, `n_heads` (Q only), opt `row_stride`/`qfused_mode` (narrow-view layout).

**Tensor-view requirement (MUST):** the rotated tensor is an `mpk.narrow` slice of a wider
buffer — e.g. `k_pe` = cols `[2048:2112)` of `qkv_a [T,2112]` (stride `[2112,1]`, offset 2048);
`q_pe` is a slice of the q_b output. The kernel rotates **in place via the view's `stride[0]` +
offset** (`row_stride`/`qfused_mode` carry the parent row width); it must not assume the rope
dims are contiguous across rows.

**Shape variants**

| role | shape | notes |
|---|---|---|
| Q | `Hd` heads × `qk_rope=64` | `Hd=16` this config |
| K | `1 × qk_rope=64` | shared across heads |

**Reuse:** `deepseek_mla_rope_q_fused_layer` / `deepseek_mla_rope_q_split_layer` /
`deepseek_mla_rope_k_layer` (collapse via `role`).
