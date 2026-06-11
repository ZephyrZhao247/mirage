# `linear_fp8` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** `C = A·Wᵀ (+residual)`, FP8 e4m3, block-scaled.

**Phase:** both.

**grid_dim:** `(N/128, 1, 1)` (e.g. N=1536→`(12,1,1)`, N=7168→`(56,1,1)`); split-K variant
`(N/128, split_k, 1)` (grid.y = K-slices, atomic/TMA reduce-add); block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `A_fp8` | `[M,K]` | e4m3 | row-major; quantized activation (M=T) |
| `A_scale` | — | uint32/f32 | UE8M0/fp32 per [`quantize_fp8`](./quantize_fp8.md) |
| `W_fp8` | `[N,K]` | e4m3 | row-major; quantized weight |
| `W_scale` | — | f32 | per-128×128-block scales |
| `residual` (opt) | `[M,N]` | bf16 | added when `with_residual` |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `C` | `[M,N]` | bf16 | row-major; result (optionally residual-fused, TP-rank-gated) |

**Params:** `M,N,K`, `with_residual`, `scale_layout`, `tp_role∈{replicated,column,row}`, `split_k`.

**Shape variants** (M=T; per-rank shapes at TP=4×EP=2):

| role | K | N (per-rank) | residual | tp_role |
|---|---|---|---|---|
| q_a (down) | 7168 | 1536 | – | replicated |
| kv_a (down, +mqa) | 7168 | 576 | – | replicated |
| q_b (prefill up) | 1536 | 3072 (=16·192) | – | column |
| kv_b (prefill up, **fused k+v**) | 512 | 4096 (=16·256) | – | column |
| o_proj (down) | 2048 (=16·128) | 7168 | ✓ | row +AR |
| dense gate_up | 7168 | 4608 (=2·18432/8) | – | column |
| dense down | 2304 (=18432/8) | 7168 | ✓ | row +AR |
| shared gate_up | 7168 | 512 (=2·256) | – | column |
| shared down | 256 | 7168 | ✓ | row +AR |

**Reuse:** `fp8_gemm_dense_smallm_layer` / `fp8_gemm_dense_mediumm_layer` (small/medium-M —
the common decode+prefill GEMMs), `fp8_gemm_dense_decode_splitk_layer` /
`linear_splitk_swapAB_fp8_layer` (split-K). Pick by M/shape — different grid → Python-level
select. FP8-out epilogue via the `*_fp8out_layer` variants (fuses the next `quantize_fp8`).
*(v1: input is a standalone [`quantize_fp8`](./quantize_fp8.md), not the fused rmsnorm+quant.)*

**Tensor-view requirement (MUST):** the activation `A_fp8` is often a `mpk.narrow` slice of a
wider buffer — e.g. `q_b` / `kv_b` read the post-norm `q_a` / `c_latent` slices of `qkv_a [T,2112]`
(stride `[2112,1]`). The GEMM must load A via `stride[0]` + offset; some outputs likewise write
into slices of a wider buffer (e.g. decode BMM1 → `q[:,:,:512]`, see [`bmm_fp8`](./bmm_fp8.md)).

**Layouts feeding [`mla_prefill_attn`](./mla_prefill_attn.md):** `q_b` emits the committed
`[all-nope (H·128) | all-pe (H·64)]` buffer (`N=3072`); `kv_b` is the **fused k+v** GEMM
(`kv_b_k`/`kv_b_v` weights concatenated → `[H,256,512]`) emitting per-head `[k_nope | v]`
(`N=4096`). These are the single-buffer contracts the new prefill kernel consumes directly.

**Runtime fusions (not separate contracts):**
- **`qkv_a` is one fused GEMM**: the `q_a` and `kv_a` rows above are produced by a single
  runtime GEMM `H → 2176` = `[q_a(1536) | c_latent(512) | k_pe(64) | pad(64)]`; downstream
  consumers read `mpk.narrow` views. (Listed separately for clarity of roles.)
- **`o_proj` residual is in [`all_reduce`](./all_reduce.md), not the GEMM** (TP path): `o_proj`
  emits a per-rank partial; the residual is fused at the all-reduce's final store. The
  `with_residual` flag applies on the single-GPU/non-TP path.
- **decode `q_b` splits into `q_b_nope` + `q_b_pe`** (two GEMMs → `q_nope[H,128]`, `q_pe[H,64]`)
  feeding [`bmm_fp8`](./bmm_fp8.md) BMM1 + [`assemble_q_decode`](./assemble_q_decode.md); prefill
  uses the single fused `q_b`.

**Open:** confirm fused gate_up vs separate gate/up emit.
