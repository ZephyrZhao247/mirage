# `quantize_fp8` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** per-(token, 128-group) absmax scale; `x_fp8 = round(x/scale)`.

**Phase:** both.

**grid_dim:** `(T,1,1) = (128,1,1)`; one CTA per token row; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `x` | `[T,D]` | bf16 | row-major; activations |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `x_fp8` | `[T,D]` | e4m3 | row-major; quantized activations |
| `scale` | `[T, D/128]` logical | uint32 / f32 | **K-major in both** — one scale per 128-element K-group, contiguous along K, strided by token. **UE8M0**: 4 exponent-only 8-bit scales packed per `uint32` (K axis = `ceil(K_groups/4)` words), batch padded to a tile multiple (`aligned_batch`) for the swapAB MMA. **fp32**: one `float32`/group, exact batch (dense/MoE path). They differ by **packing/dtype/padding, not major-order**. Set by `scale_layout`. |

**Params:** `scale_layout∈{ue8m0,fp32}`, `group_size=128`.

**Shape variants / insertion points**

| insertion point | shape | scale fmt | notes |
|---|---|---|---|
| qkv_a in / o_proj in / MoE in | `[T, H=7168]` | UE8M0 | standalone in v1 (rmsnorm+quant unfused) |
| MoE intermediate (w13→w2) | `[Mtot, I_r=512]` | fp32 / UE8M0 | between grouped GEMMs |
| q_nope (pre-BMM1) | `[T, H, 128]` per-head | UE8M0 | may fuse into q_b `*_fp8out` |
| attn_out (pre-BMM2) | `[T, H, 512]` per-head | UE8M0 / fp32 | |
| attn_out_reduced (pre-o_proj) | `[T, H·128]` | UE8M0 | |
| q_b proj (= q_a slice) | `[T, 1536]` | UE8M0 | slice of `qkv_a`, **stride `[2112,1]`**, offset 0 |

**Tensor-view requirement (MUST):** several inputs are `mpk.narrow` column slices — e.g. the
`q_b`-input quantize reads `q_a` = cols `[0:1536)` of `qkv_a [T,2112]` (stride `[2112,1]`). The
kernel must load `D` columns per row via `stride[0]` + view offset, and form the per-128 scale
groups over the `D` **view**-columns only (not the parent width).

**Reuse:** `quantize_fp8_layer`.
