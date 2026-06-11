# `bmm_fp8` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** per-head batched matmul `C[t,h,:] = A[t,h,:]·W[h,:,:]ᵀ`; absorbed-decode MLA.

**Phase:** decode.

**grid_dim:** `(Dout/128, Hd, 1)`; grid.y = one head/CTA, grid.x tiles `Dout`; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `A_fp8` | `[T,Hd,Kin]` | e4m3 | head-major; per-head activation |
| `A_scale` | `[T,Hd,*]` | uint32/f32 | per-head UE8M0/fp32 |
| `W_fp8` | `[Hd,Dout,Kin]` | e4m3 | per-head absorbed weight |
| `W_scale` | `[Hd,*,*]` | f32 | per-head block scales |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `C` | `[T,Hd,Dout]` | bf16 | head-major |

**Params:** `Hd`, `Dout`, `Kin`, `scale_layout`.

**Shape variants** (`Hd=16` at this config):

| role | Kin | Dout | grid |
|---|---|---|---|
| BMM1 q→latent (W_UK) | 128 (qk_nope) | 512 (kv_lora) | (4,16,1) |
| BMM2 o un-absorb (W_UV) | 512 (kv_lora) | 128 (v_head) | (1,16,1) |

**Tensor-view requirement (MUST):** BMM1 writes its `[T,H,512]` latent output into the
**`[:,:,:512]` slice-view** of the `q[T,H,576]` buffer finalized by
[`assemble_q_decode`](./assemble_q_decode.md) — the store must honor the parent per-head row
stride (576), not 512. Inputs/outputs may also be reshaped 2D⇄3D views of the same bytes.

**Reuse:** `linear_fp8_bmm_sm100_layer` (UE8M0) / `linear_fp8_bmm_dense_sm100_layer` (fp32).
