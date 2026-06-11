# `kv_cache_gather` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** append new latent KV to the paged cache, then materialize a contiguous window for attention.

**Phase:** both.

**grid_dim:** `(num_requests, num_gather_splits, 1) = (4,8,1)`; grid.x = request, grid.y =
seq-split fan-out; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `c_latent_new` | `[T,kv_lora=512]` | bf16 | row-major; new compressed KV |
| `k_rope_new` | `[T,qk_rope=64]` | bf16 | row-major; new rope key |
| `paged_cache` | `[...]` | bf16 | paged KV store (page_size rows) |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `contiguous_kv` | `[L,576]` | bf16 | row-major; concatenated `[kv_lora\|qk_rope]` window (decode) |
| `ckv` (opt) | `[L,512]` | bf16 | separated compressed KV (prefill) |
| `kpe` (opt) | `[L,64]` | bf16 | separated rope key (prefill) |

**Params:** `page_size`, `num_gather_splits`.

**Tensor-view requirement (MUST):** `c_latent_new` / `k_pe_new` are `mpk.narrow` slices of the
fused `qkv_a [T,2112]` — `c_latent_new` at offset 1536, `k_pe_new` at offset 2048, both with
**stride `[2112,1]`**. The kernel must read them via `stride[0]` + offset (the
`c_latent_row_stride` / `k_pe_row_stride` params carry the parent width 2112), not as contiguous
`[T,512]` / `[T,64]` buffers.

**Shape variants**

| phase | outputs |
|---|---|
| decode | `contiguous_kv` only |
| prefill | `contiguous_kv` + `ckv` + `kpe` |

**Reuse:** `mla_kv_gather_unified_layer` (drop legacy `mla_kv_gather_layer`).
