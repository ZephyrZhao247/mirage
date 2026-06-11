# DeepSeek V3 — Kernel Specification Category

A top-down, contract-first specification of the kernels needed to run the DeepSeek V3 MPK
demo. Each file in this directory specifies **one layer type** — its semantics, exact I/O
(layout · dtype · meaning), `grid_dim`, supported shape variants, and the existing kernel
to reuse. These contracts are the debugging anchor: a wrong demo is diagnosed by checking
each layer against its contract, not by reverse-engineering the existing code.

> **Scope:** specification only — no kernel implementation. The "Reuse" line on each spec
> names the existing `PersistentKernel` method that fulfills the contract.

## Target config (v1)
- **TP=4 × EP=2, world_size=8.** Decode + non-chunked prefill. NEW grouped-GEMM MoE. Greedy.
- **Attention:** heads split across all 8 ranks → `Hd = 128/8 = 16` heads/rank (decode = tp8 variant).
- **Routed experts:** `routed_tp = world/ep = 4`, `ep = 2` → `E_loc = 256/2 = 128` experts/rank;
  routed intermediate `I_r = 2048/4 = 512`/rank.
- **Shared expert:** TP across all 8 → shared intermediate `I_s = 2048/8 = 256`/rank.
- **Deferred:** MTP, probabilistic sampling, chunked prefill, OLD per-expert MoE.

## Typing rules
- Type = structural kernel identity, not weight role. Projections collapse into
  `linear_fp8`/`linear_bf16` (shape variants); **prefill MLA ≠ decode MLA**. Per-head
  batched matmul (`bmm_fp8`) is its own type. `rmsnorm`/`quantize_fp8` are separate
  composable types (the fused rmsnorm+quant kernel is a reuse-only hot-path optimization).
- **EP needs no token exchange:** each rank computes its local experts for all tokens; a
  single `all_reduce` sums routed contributions across TP×EP ranks. So MoE permute/unpermute
  are purely **local** — no dispatch/combine kernels.

## DSv3 constants
`H=7168`, `n_heads=128`, `q_lora=1536`, `kv_lora=512`, `qk_nope=128`, `qk_rope=64`,
`qk_head=192`, `v_head=128`, `q_absorbed=576 (=512+64)`, `inter_dense=18432`,
`inter_moe=2048`, `E=256`, `Ep=8`, `n_group=8`, `topk_group=4`, `V=129280`,
`n_layers=61` (0–2 dense, 3–60 MoE). `T`=batched tokens (`mbt`, e.g. 128).

## Manifest (20 types)

| # | Spec | One-line semantics | Phase | Reuse |
|---|---|---|---|---|
| 1 | [`embed`](./embed.md) | token-id → hidden lookup | both | reuse |
| 2 | [`rmsnorm`](./rmsnorm.md) | RMS normalize → bf16 | both | reuse |
| 3 | [`quantize_fp8`](./quantize_fp8.md) | bf16 → per-group FP8 + scale | both | reuse |
| 4 | [`linear_fp8`](./linear_fp8.md) | block-scaled FP8 GEMM (+residual) | both | reuse |
| 5 | [`linear_bf16`](./linear_bf16.md) | bf16 GEMM (+residual) | both | reuse |
| 6 | [`bmm_fp8`](./bmm_fp8.md) | per-head batched FP8 matmul | decode | reuse |
| 7 | [`assemble_q_decode`](./assemble_q_decode.md) | interleave q_latent\|q_pe → q[576] | decode | reuse |
| 8 | [`rope`](./rope.md) | rotary on decoupled rope dims | both | reuse |
| 9 | [`kv_cache_gather`](./kv_cache_gather.md) | append latent KV + materialize | both | reuse |
| 10 | [`mla_decode_attn`](./mla_decode_attn.md) | latent attention vs paged cache | decode | reuse |
| 11 | [`mla_decode_reduce`](./mla_decode_reduce.md) | merge split-KV partials | decode | reuse |
| 12 | [`mla_prefill_attn`](./mla_prefill_attn.md) | full prompt attention | prefill | **new** (clean iface) |
| 13 | [`silu_mul`](./silu_mul.md) | silu(gate)·up | both | reuse |
| 14 | [`moe_router`](./moe_router.md) | sigmoid group top-k | both | reuse |
| 15 | [`moe_permute`](./moe_permute.md) | local token→expert permute | both | reuse |
| 16 | [`grouped_gemm_fp8`](./grouped_gemm_fp8.md) | grouped FP8 GEMM over local experts | both | reuse |
| 17 | [`moe_unpermute`](./moe_unpermute.md) | unpermute + weighted sum + residual | both | reuse |
| 18 | [`all_reduce`](./all_reduce.md) | sum across TP×EP ranks | both | reuse |
| 19 | [`argmax`](./argmax.md) | greedy token (partial+reduce) | both | reuse |
| 20 | [`global_argmax`](./global_argmax.md) | cross-rank argmax (sharded vocab) | both | reuse (cond.) |

**Spec template:** `Semantics` · `Phase` · `grid_dim` · `Inputs` · `Outputs` · `Params` ·
`Shape variants` · `Reuse` · `Open`. I/O rows are `name [shape] dtype — layout / meaning`.

## Forward pass (TP=4×EP=2)

```
token_ids → embed
[×61]
  rmsnorm(input)
  q:  linear_fp8(q_a) → rmsnorm(q_a) → linear_fp8(q_b)     # → q_nope[H,128] + q_pe[H,64]
  kv: linear_fp8(kv_a) → rmsnorm(kv_a) → [prefill: linear_fp8(kv_b, fused k+v)]
  rope(Q: q_pe); rope(K: k_rope); kv_cache_gather
  decode:  quantize_fp8(q_nope) → bmm_fp8(BMM1) → assemble_q_decode[q_latent|q_pe]
           → mla_decode_attn(tp8) → mla_decode_reduce(tp8) → bmm_fp8(BMM2)
  prefill: mla_prefill_attn(tp8)            # consumes q[H·192], kv[L,H,256], k_rope
  linear_fp8(o_proj,+res) → all_reduce
  rmsnorm(post-attn)
  dense(0-2): linear_fp8(gate_up) → silu_mul → linear_fp8(down,+res) → all_reduce
  MoE(3-60):  moe_router; quantize_fp8 → moe_permute
              grouped_gemm_fp8(w13) → silu_mul → quantize_fp8 → grouped_gemm_fp8(w2)
              moe_unpermute(+shared, +res)      [shared = linear_fp8→silu_mul→linear_fp8]
              all_reduce                         (sums routed across TP×EP=8 ranks)
rmsnorm(final) → linear_bf16(lm_head) → argmax | global_argmax → token
```

## Reuse targets
- `python/mirage/mpk/models/deepseek_v3/builder.py` — call sites, TP/EP topology, grid_dim.
- `python/mirage/mpk/persistent_kernel.py` — existing layer methods.

## Resolved during design
- **Topology:** attention 16 heads/rank (decode=tp8); routed experts `E_loc=128`, `I_r=512`; shared `I_s=256`.
- **MoE:** local permute/grouped-GEMM/unpermute + single `all_reduce`; no EP dispatch/combine.
- **BMM dims:** BMM1 `128→512`, BMM2 `512→128`, per-head `Hd=16`.
- **Prefill:** non-chunked prefill = `mla_prefill_tp8_chunked_layer` invoked as one full chunk.
- All 19 specs map to an existing reuse target.
