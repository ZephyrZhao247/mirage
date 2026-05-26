# Wave 1 — Consolidated Open Questions

Aggregated from `overview.md`, `hc.md`, `attention.md`, `sparse.md`, `moe.md`, `mtp.md`
(all under `/raid/user_data/zepengz/projects/mirage_2/docs/mpk/deepseek_v4/`).

**Total: 33 questions.**
Classification breakdown:
- **17 RESOLVED**
- **6 STRONG-GUESS**
- **6 NEEDS-USER-DECISION**
- **4 DEFER-TO-WAVE-2**

Citation conventions: file paths are absolute; line numbers refer to the
file as it exists in the repository at the time of writing.

---

## Summary table

| # | Spec | Topic (one line) | Class | Resolution / Recommended answer |
|---|---|---|---|---|
| 1 | hc.md L1346 | TaskType ID range exhausted at `TASK_SM100_TASK_END=298` | RESOLVED | Bump `TASK_SM100_TASK_END` to `320` in `runtime_header.h:197`; first V4 PR bundles the bump. |
| 2 | hc.md L1358 | `TaskMetadata` union — new `token_offset` field vs. alias `task_offset` | STRONG-GUESS | Add a new `struct { int token_offset; }` branch to the union (zero-cost, the `static_assert` only checks total 8B size). |
| 3 | hc.md L1369 | Thread count for `mhc_pre` (vLLM=96, MPK conv=256) | DEFER-TO-WAVE-2 | Recommend 128; final pick belongs to the kernel author once they profile B200. |
| 4 | hc.md L1377 | Output dtype/shape of `comb_mix` (`[N, hc*hc]` flat vs `[N, hc, hc]`) | RESOLVED | Use `[N, hc, hc]` to match the PyTorch reference; bytes are identical, only DTensor metadata differs. |
| 5 | hc.md L1386 | Fuse `mhc_pre` + `mhc_prenorm_gemm` into single task? | STRONG-GUESS | Keep separate in v1 to mirror vLLM boundaries; revisit in v2. |
| 6 | hc.md L1396 | `hc_post_mult_value` — parameter vs. compile-time constant | RESOLVED | Bake `2.0` as compile-time constant in v1; matches `kernel.py:394` and `model.py:686`. |
| 7 | hc.md L1404 | Test fixture for HC weights — random vs real | RESOLVED | Use random fp32 in unit + module tests; real-weight check is Wave-3. |
| 8 | hc.md L1411 | `mhc_prenorm_gemm` `splits > 1` in v1? | RESOLVED | v1 `splits=1` only; leading `splits` dim in DTensor shape preserved for v2. |
| 9 | moe.md L342 | Single-CTA topology for `hash_route_lookup_layer` | RESOLVED | Adopt `grid_dim=(1,1,1)`, `block_dim=(256,1,1)` — matches V3 sigmoid kernel in `builder.py:909-911`. |
| 10 | moe.md L1178 | Is `topk_sigmoid_sm100.cuh` correct for `T > 8`? | RESOLVED | **No** — the kernel only processes the first `WARPS_PER_CTA * ROWS_PER_WARP = 8` rows; `T > 8` silently drops rows. Latent bug. Backport fix to V3 when V4 lands. |
| 11 | moe.md L1186 | HC space wiring of `residual` in `moe_mul_sum_add_layer` | RESOLVED | The MoE writes a flat `[T, D]` output; `mhc_post` (Wave-2 task in `hc.md`) is responsible for the HC reshuffle. Cross-spec consistency confirmed. |
| 12 | moe.md L1193 | `e_score_correction_bias` missing for hash layers — `_zero_bias_E` equivalent? | STRONG-GUESS | Pass a zeroed `[E]` bias buffer — semantically identical to `bias=None` since the score-add is `score + bias` then topk. v2 may add a `nullptr`-allowed code path. |
| 13 | moe.md L1200 | Partial-batch (`T < max_num_batched_tokens`) initialization for hash kernel | RESOLVED | Already adopted in §1.9 — per-token Phase-0 zeroing over `E * T` covers padded columns, mirroring V3 sigmoid (`topk_sigmoid_sm100.cuh:104-114`). |
| 14 | moe.md L1207 | Does layer 43 (MTP) use MoE or its own FFN? | RESOLVED | MTP uses the same `MoE` (with shared + routed experts) by inheriting from `Block`. Checkpoint keys confirm: `mtp.0.ffn.shared_experts.*`, `mtp.0.ffn.experts.{0..255}.*`, `mtp.0.ffn.gate.*` all exist. The MTP block runs the standard MoE pipeline. |
| 15 | attention.md L1498 | Second paged-cache triple in meta-tensors | NEEDS-USER-DECISION | Recommend the **parallel-triple** approach (`paged_compressed_kv_indptr_buffer`, etc.) for clarity. Audit task per plan §Phase A. |
| 16 | attention.md L1506 | FP8 vs bf16 SWA cache row layout for v1 | NEEDS-USER-DECISION | Recommend **bf16 throughout** for v1 (avoids quant kernel during cache insert; matches V3's MPK convention). |
| 17 | attention.md L1514 | `combine_topk_swa_indices` location — host vs device | STRONG-GUESS | Keep on host (Python builder) in v1, fuse into gather task in v2. Matches vLLM's `deepseek_v4_attention.py:969-981`. |
| 18 | attention.md L1521 | Attn_sink semantics across splits | DEFER-TO-WAVE-2 | Implementer must read FlashMLA-Sparse reference and pick one of: (a) one split adds sink, others add 0; (b) all splits add `sink/num_splits`. Document choice in generated kernel comments. |
| 19 | attention.md L1527 | Padded heads — H=64 always? | RESOLVED | V4-Flash has `num_attention_heads=64` exactly (`config.json:23`). Padding is a no-op. Hard-code `padded_heads=64` for v1. |
| 20 | attention.md L1533 | MTP decode reuse | RESOLVED | v1: MTP reuses `mla_v4_decode` with `compress_ratio=0` (per `compress_ratios[43]=0`). Confirmed by official `model.py:792` instantiating `MTPBlock(args.n_layers + layer_id, args)` which calls `super().__init__` (Block) using `args.compress_ratios[43]`. |
| 21 | attention.md L1538 | Inverse-RoPE conjugate sign convention | DEFER-TO-WAVE-2 | Implementer validates by round-trip (forward RoPE → inverse RoPE → original) in the §5.9 test. PyTorch source: `model.py:232-242` (`apply_rotary_emb` with `inverse=True` takes the conjugate of `freqs_cis`). |
| 22 | attention.md L1544 | `tma_aligned_T = round_up_4(B)` rounding in decode | DEFER-TO-WAVE-2 | Implementer verifies vs `deepseek_v4_attention.py:326-334` consumer; likely fine because `o_fp8` stride `(d, T*d, 1)` is the variable-T layout. |
| 23 | mtp.md L600 | HF checkpoint key prefix for MTP weights | RESOLVED | Prefix is **`mtp.0.*`**, NOT `layers.43.*`. Confirmed by direct grep on `/raid/catalyst/models/DeepSeek-V4-Flash-Base/model.safetensors.index.json` — keys `mtp.0.hc_head_base`, `mtp.0.attn.*`, `mtp.0.ffn.*`, `mtp.0.enorm.weight`, `mtp.0.hnorm.weight`, `mtp.0.e_proj.{weight,scale}`, `mtp.0.h_proj.{weight,scale}` all present in shard 46. |
| 24 | mtp.md L610 | Does MTP use FP8 for `e_proj` and `h_proj`? | RESOLVED | **Yes.** Index has `mtp.0.e_proj.weight` + `mtp.0.e_proj.scale` and `mtp.0.h_proj.weight` + `mtp.0.h_proj.scale`. Presence of `.scale` (UE8M0 block scales per `config.json:35-43`) confirms FP8 storage. Use `linear_fp8_layer` in MPK. |
| 25 | mtp.md L620 | Strategy A1 weight replication (128 MiB) — converter vs A2 add-only | NEEDS-USER-DECISION | Default **A1 in v1** (per spec). Memory cost is bounded (128 MiB for 4 hc copies), task count stays minimal. Revisit after v1 profiling. |
| 26 | mtp.md L625 | MTP `attn_norm` / `ffn_norm` use base Block path? | RESOLVED | **Yes.** `model.py:739-746`: `MTPBlock(Block).__init__` calls `super().__init__` which builds `self.attn_norm` and `self.ffn_norm` (line 659-660). Checkpoint keys `mtp.0.attn_norm.weight`, `mtp.0.ffn_norm.weight` confirm. |
| 27 | mtp.md L633 | Shared paged-KV cache vs MTP-private cache | RESOLVED | **Shared cache, MTP slot = layer_id 43.** Official `model.py:741` calls `super().__init__(layer_id, args)` with `layer_id = args.n_layers + 0 = 43`. The base `Block.__init__` constructs a per-layer `Attention(layer_id, args)` whose `kv_cache` is buffered per layer — but in MPK we already allocate per-layer slices in `paged_kv_indptr_buffer`. Adopt the same per-layer slot pattern for MTP. |
| 28 | mtp.md L644 | `_mtp_hidden_buffer` in vLLM implies MPK buffer? | RESOLVED | **No buffer needed.** vLLM's separate buffer exists because vLLM runs base and MTP in two captured CUDA graphs. MPK runs both in one persistent kernel, so the base model's pre-`mhc_head` HC tensor is wired directly as `prev_x_hc` input to the MTP block. |
| 29 | overview.md L774 | MTP `compress_ratios[43]` interpretation | RESOLVED | `config.json:66` → `compress_ratios[43] = 0`. Official `model.py:453` passes `args.compress_ratios[layer_id]` directly to `Attention`, so MTP is **dense SWA** (no compressor/indexer). v1 follows official semantics (not the vLLM `max(1, ...)=1` workaround). |
| 30 | overview.md L784 | `wq_a + wkv` fused (vLLM) vs separate (official) | RESOLVED | Checkpoint stores them **separately** (`mtp.0.attn.wq_a.weight` + `mtp.0.attn.wq_a.scale`, `mtp.0.attn.wkv.weight` + `mtp.0.attn.wkv.scale` — same for `layers.{0..42}`). vLLM's `fused_wqa_wkv` is a runtime fusion (`deepseek_v4.py:966, 1344-1345`). v1 follows the **separate** on-disk layout; future fusion is an MPK-side optimization. |
| 31 | overview.md L792 | `mhc_prenorm_gemm` fused vs decomposed fallback | NEEDS-USER-DECISION | Python signature must accept both. Recommend v1 ships the **decomposed** path (RMSNorm-then-linear) because porting the CUTLASS prenorm-GEMM is high-risk per plan §Wave-2. |
| 32 | overview.md L801 | `repeat_to_hc` materialization — broadcast vs allocate | STRONG-GUESS | Fold the `hc` broadcast into the first `mhc_prenorm_gemm` consumer (set hc-axis stride to 0) — avoids the `hc_mult`× memory blowup. Wave-2 author confirms kernel can handle zero-stride input. |
| 33 | overview.md L809 | Paged-KV-cache extensions for V4 | NEEDS-USER-DECISION | Per plan §Phase A, a dedicated audit task is gated before Wave 2. This question is a pointer to that audit, not a question to resolve here. |
| 34 | overview.md L817 | UE8M0 scale conversion for routed experts | RESOLVED | Use the existing MPK pipeline. `config.json:35-43` says `scale_fmt=ue8m0`, `weight_block_size=[128,128]`. MPK already packs ue8m0 into `uint32` per `persistent_kernel.py:1974-1988`. The convert script must view the on-disk `float8_e8m0fnu` bytes (treated as `uint8` in vLLM `deepseek_v4.py:1383-1388`), reshape into `[128,128]` blocks, and pack 4 bytes/block into a `uint32`. |
| 35 | overview.md L825 | Hadamard transform impl in `indexer_q_transform` | RESOLVED | The Indexer-Q Hadamard is **absorbed offline into `wq_b`** by the convert script (see sparse.md §F.2.3, vLLM `deepseek_v4_attention.py:1066-1072`). The kernel itself contains **no runtime Hadamard**. The Compressor's `head_dim=128` Hadamard cannot be absorbed (per-token softmax mass varies), so it remains as a runtime matmul against a `__constant__` table. Stored shape is `[128, 128]` bf16 = 16 KiB. |
| 36 | sparse.md L181 | FP8 indexer cache for v1 vs match vLLM FP4 default | NEEDS-USER-DECISION | Recommend **FP8** for v1 per plan `i-want-to-add-dapper-pascal.md:432-433` ("v1 may compute Hadamard naively"). MXFP4 task variants stay v2. |
| 37 | sparse.md L234 | Hadamard constant — `cudaMemcpyToSymbol` vs literal codegen | STRONG-GUESS | Recommend **literal-embedded constant table in codegen** (option b). Avoids new runtime wiring; 16 KiB embedded in a `__device__ __constant__` is well within limits. |
| 38 | sparse.md L432 | Two-task design (`indexer_score_sm100` → `indexer_topk_sm100`) compatibility with MPK | RESOLVED | Compatible. Precedent: `mla_kv_gather_layer` (`persistent_kernel.py:1401-1420`) → `mla_decode_layer` (`persistent_kernel.py:1451-1481`) is the existing producer→consumer chain registered via `kn_graph.register_task` calls on two task names. The scheduler enforces ordering via the kernel-graph DAG. |
| 39 | sparse.md L976 | Indexer Q RoPE: `rope_theta=10000` or `compress_rope_theta=160000`? | RESOLVED | **`compress_rope_theta=160000`** (config.json:65). Trace: `model.py:493-494` assigns `self.indexer.freqs_cis = self.freqs_cis`. The Attention layer's `freqs_cis` is built at `model.py:480-481` with `rope_theta = args.compress_rope_theta` when `compress_ratio > 0` (and Indexer only exists when `compress_ratio == 4`). |
| 40 | sparse.md L1058 | Compressor head=128 Hadamard absorbable into `wkv`? | RESOLVED | **No, cannot be absorbed.** The Compressor rotates `kv` *after* the gated-softmax pool over the `coff*head_dim` input. The pool weights are per-token, so pre-multiplying the Hadamard into `wkv` is not algebraically equivalent. Runtime matmul against the `__constant__` table is required. |
| 41 | sparse.md L1217 | `weights_proj` output sign — can it be negative? | RESOLVED | **Yes.** `model.py:394` constructs `weights_proj` as `ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)` — a plain linear with **no activation, no bias clamping, no abs/relu**. Output is bf16, can be negative. The relu-then-weight ordering (official `model.py:421`) is therefore numerically distinct from vLLM's weight-then-relu. v1 follows the official's ordering. |
| 42 | sparse.md L1317 | Heap-based fused score+topk vs split into two tasks | DEFER-TO-WAVE-2 | For v1 the spec already recommends the single-task heap form (simpler, no large intermediate buffer when `K=512` is fixed and per-token candidate count ≤ 4096). Implementer profiles after first port. |
| 43 | sparse.md L1588 | (Duplicate of #36 — same FP8 vs FP4 indexer cache question) | NEEDS-USER-DECISION | Same answer as #36: FP8 in v1. |
| 44 | sparse.md L1594 | (Duplicate of #38 — two-task producer/consumer chain) | RESOLVED | Same answer as #38: compatible. |
| 45 | sparse.md L1599 | (Duplicate of #37 — Hadamard constant init) | STRONG-GUESS | Same answer as #37: literal-embedded. |
| 46 | sparse.md L1603 | (Duplicate of #39 — Indexer Q rope theta) | RESOLVED | Same answer as #39: `compress_rope_theta=160000`. |
| 47 | sparse.md L1612 | (Duplicate of #40 — Compressor Hadamard absorbable) | RESOLVED | Same answer as #40: not absorbable. |
| 48 | sparse.md L1620 | (Duplicate of #41 — `weights_proj` sign) | RESOLVED | Same answer as #41: can be negative. |
| 49 | sparse.md L1625 | DeepGEMM `fp8_fp4_paged_mqa_logits` source path | RESOLVED | Path is `deps/vllm/vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp8_paged_mqa_logits.cuh` (and `sm100_fp4_paged_mqa_logits.cuh` for the MXFP4 path). Both confirmed to exist via `find`. v1 does not port these — it writes a naive equivalent — but the spec cites them for v2. |
| 50 | sparse.md L1631 | `kv_cache_block_size=64` (V3 value) vs `block_size=4` for compressed-KV | NEEDS-USER-DECISION | Need to confirm with V4 reference. Recommend **block_size=64** (V3 value) for the compressed-KV cache in v1 to maximize reuse; if a mismatch with vLLM's `SlidingWindowMLASpec` is observed in module tests, switch to `block_size=4`. |
| 51 | sparse.md L1637 | Corner test "head_dim=128, compress_ratio=128" | DEFER-TO-WAVE-2 | Skipping the corner test is fine for v1 — it doesn't exist in real config. Implementer decides based on whether `head_offset` off-by-one logic gets touched. |

Counts after deduplication of sparse.md repeated questions:
- Of the 33 *unique* questions in the spec set, the count is:
  - **17 RESOLVED** (Q1, Q4, Q6, Q7, Q8, Q9, Q10, Q11, Q13, Q14, Q19, Q20, Q23, Q24, Q26, Q27, Q28, Q29, Q30, Q34, Q35, Q38=44, Q39=46, Q40=47, Q41=48, Q49)
  - **6 STRONG-GUESS** (Q2, Q5, Q12, Q17, Q32, Q37=45)
  - **6 NEEDS-USER-DECISION** (Q15, Q16, Q25, Q31, Q33, Q36=43, Q50)
  - **4 DEFER-TO-WAVE-2** (Q3, Q18, Q21, Q22, Q42, Q51)

(The duplicate sparse.md tail items §J reuse earlier answers; the listing above
covers each *distinct* topic once and re-references duplicates by number.)

---

## Detail (one section per question)

---

### Q1: TaskType ID range exhausted (hc.md L1346)

**Question (verbatim from spec):**
> `TASK_SM100_TASK_END = 298` in `runtime_header.h:197` leaves only `296, 297`
> free for new tasks before we hit the end sentinel. We have 4 new mHC tasks
> **plus** the V4 attention pipeline (5 tasks per `attention.md`), V4 sparse
> (3 tasks per `sparse.md`), V4 MoE (3 tasks per `moe.md`), and V4 MTP (1
> task per `mtp.md`). That's 16 new SM100 tasks total against 2 free slots.

**Spec-author guess:** bump `TASK_SM100_TASK_END` to at least 320 (15 new
slots + room for v2). First V4 task PR bundles the bump.

**Resolution:** RESOLVED.

**Answer:** Confirmed by direct read.
`/raid/user_data/zepengz/projects/mirage_2/include/mirage/persistent_kernel/runtime_header.h:144`
defines `TASK_SM100_TASK_BEGIN = 230` and line 197 defines
`TASK_SM100_TASK_END = 298`. Range 230..298 is the SM100 reserved span. We
need ≥16 new task IDs. Bump `TASK_SM100_TASK_END` to **320** (giving 22
free slots after the existing TMA tasks). The grep
`grep -n 'TASK_SM100_TASK_END' src/ include/` reveals only the enum
definition references this name, so the bump is safe.

**Citation:** `include/mirage/persistent_kernel/runtime_header.h:144, 197`.

---

### Q2: `TaskMetadata` union — new `token_offset` field (hc.md L1358)

**Question (verbatim):**
> `TaskMetadata` union: should `token_offset` be a new field, or alias the
> existing `task_offset`?

**Spec-author guess:** add a new `struct { int token_offset; }` branch.

**Resolution:** STRONG-GUESS.

**Answer:** Add a new union branch. The union at
`runtime_header.h:257-270` is exactly 8 bytes (`unsigned long long
raw_payload`), and the existing branches are:
- `struct { int expert_offset; }` (MoE)
- `struct { int16_t request_id; uint16_t kv_idx; int merge_task_offset; }` (paged attention)
- `struct { int task_offset; }` (nvshmem)

Adding a 4th branch `struct { int token_offset; }` is free (fits in 4
bytes, far under the 8-byte total). The `static_assert` at lines 273-275
only checks `sizeof == sizeof(unsigned long long)`, so it doesn't block.
Aliasing `task_offset` works but readers would be confused.

**Citation:** `include/mirage/persistent_kernel/runtime_header.h:257-275`.

---

### Q3: Thread count for `mhc_pre` (hc.md L1369)

**Question (verbatim):**
> Thread count for `mhc_pre`: vLLM uses 96 threads (3 warps; 1 warp for
> post/comb/Sinkhorn + 2 warps for pre/mix). MPK's Blackwell convention is
> `WORKER_NUM_THREADS = 256`. We can: (a) use 128 threads, (b) use 256
> threads, (c) match vLLM at 96. Recommendation: **128 threads**.

**Resolution:** DEFER-TO-WAVE-2.

**Answer:** This is a pure performance tuning question; correctness is
not affected. The implementer should pick 128 (per spec recommendation),
profile occupancy, and adjust if needed. MPK's Blackwell `WORKER_NUM_THREADS`
convention (256) does not dictate the in-task thread count for register-bound
kernels — it merely bounds it. The mHC tasks are small enough that 128
threads should reach high occupancy.

---

### Q4: Output dtype/shape of `comb_mix` (hc.md L1377)

**Question (verbatim):**
> Output dtype of `comb_mix`: vLLM stores it as flat `[N, hc*hc] f32`
> (`mhc.py:78, 148-149`); PyTorch's `hc_pre` semantically treats it as
> `[N, hc, hc]`. Pick one shape.

**Resolution:** RESOLVED.

**Answer:** Use `[N, hc, hc]`. With `hc_mult = 4` (from
`config.json:9 hc_mult=4`), `[N, hc, hc] = [N, 4, 4]`, total
`hc * hc = 16 f32 elts per token`. The flat-vs-3D distinction is purely
DTensor metadata: bytes are identical. The PyTorch reference in
`deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:686` uses 3D
indexing, so matching that simplifies reference comparison.

**Citation:** `/raid/catalyst/models/DeepSeek-V4-Flash-Base/config.json:9`
(`hc_mult: 4`); `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:686`.

---

### Q5: Fuse `mhc_pre` + `mhc_prenorm_gemm`? (hc.md L1386)

**Question (verbatim):**
> Should `mhc_pre` and `mhc_prenorm_gemm` be fused into a single MPK task?

**Spec-author guess:** keep separate to match vLLM boundaries.

**Resolution:** STRONG-GUESS.

**Answer:** Keep separate. Fusion is theoretically possible (eliminates
`gemm_out_mul` / `gemm_out_sqrsum` global writeback) but would diverge
from vLLM's well-tested kernel boundaries. v1 prioritizes faithfulness;
v2 profiles the writeback cost and decides.

---

### Q6: `hc_post_mult_value` parameter vs constant (hc.md L1396)

**Question (verbatim):**
> `hc_post_mult_value`: vLLM passes this as a parameter (`mhc.py:61, 189,
> 204`), default not in the signature. PyTorch reference (`kernel.py:394`)
> hardcodes `2`.

**Resolution:** RESOLVED.

**Answer:** Bake `2.0` as a compile-time constant in v1. The two reference
implementations both use `2 * sigmoid(...)`:
- Official: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:686`
  and `kernel.py:394` show `2 * sigmoid(...)`.
- vLLM exposes it as a parameter but the only caller passes `2.0`.

If a future variant uses a different multiplier, promote to a template
parameter. v1 is single-config (Flash-Base) so the constant is safe.

**Citation:** `deps/deepseek_v4/DeepSeek-V4-Flash/inference/kernel.py:394`,
`model.py:686`.

---

### Q7: Test fixture for HC weights (hc.md L1404)

**Question (verbatim):**
> Test fixture for HC weights: the prenorm `fn` is fp32 and `hc3 × hc*H =
> 24 × 16384 = ~1.5MB` per layer.

**Resolution:** RESOLVED.

**Answer:** Generate random fp32 weights in unit tests and module tests.
Real-weight comparison against the safetensors shard belongs in Wave-3
end-to-end testing. The 1.5 MiB per layer is small enough that synthetic
tests fit in unit-test mem budgets.

---

### Q8: `mhc_prenorm_gemm` `splits > 1` in v1? (hc.md L1411)

**Question (verbatim):**
> Should `mhc_prenorm_gemm` honor `splits > 1` in v1?

**Resolution:** RESOLVED.

**Answer:** v1 supports `splits = 1` only. The scalar v1 fallback
naturally has `splits = 1` (one CTA reduces fully per token). The DTensor
shape still allocates the leading `splits` dim so `mhc_pre` does not
require re-registration when split-K lands in v2.

---

### Q9: Single-CTA hash route lookup (moe.md L342)

**Question (verbatim):**
> the V3 `moe_topk_sigmoid_routing_layer` runs as a **single CTA** over
> all rows (`grid_dim=(1, 1, 1)`, `block_dim=(256, 1, 1)`).

**Resolution:** RESOLVED.

**Answer:** Confirmed by direct read of
`python/mirage/mpk/models/deepseek_v3/builder.py:909-911`:
```python
self.mpk.moe_topk_sigmoid_routing_layer(
    input=router_logits, bias=w_bias,
    output=(moe_topk_weights, moe_routing_indices, moe_mask),
    grid_dim=(1, 1, 1),
    block_dim=(256, 1, 1),  # 8 warps required by topk kernel
)
```
`hash_route_lookup_layer` adopts the identical pattern (single CTA, 256
threads, kernel iterates all `T` tokens internally), eliminating cross-CTA
races on `moe_active_expert_ids`.

**Citation:** `python/mirage/mpk/models/deepseek_v3/builder.py:905-911`.

---

### Q10: `topk_sigmoid_sm100` correctness for `T > 8` (moe.md L1178)

**Question (verbatim):**
> is the existing `topk_sigmoid_sm100.cuh` kernel correct for `T > 8` (its
> compile-time `ROWS_PER_WARP=1, WARPS_PER_CTA=8` only covers 8 rows)?

**Resolution:** RESOLVED.

**Answer:** **The kernel only processes the first
`WARPS_PER_CTA * ROWS_PER_WARP = 8` rows.** A row is selected at line
`int const warp_base_row = warp_idx * ROWS_PER_WARP;` and consumed inside
`if (thread_row < num_rows)` — there is **no outer loop** over rows.
Specifically:
- `include/mirage/persistent_kernel/tasks/blackwell/topk_sigmoid_sm100.cuh:157`:
  `warp_base_row = warp_idx * ROWS_PER_WARP`
- Line 160: `thread_row = warp_base_row + thread_row_in_warp`
- Line 165: `if (thread_row < num_rows) { ... }` — single guard, no loop.

The Phase-0 zeroing at line 107 *does* loop over `num_rows` (`for (int row
= 0; row < num_rows; ++row)`), so the routing-indices buffer is correctly
zeroed even when `num_rows > 8`. But the **scoring loop itself** silently
drops any rows beyond 8.

This is a latent V3 bug if V3 has ever been called with `T > 8` per
invocation. The reason V3 demos pass is probably that V3 launches one CTA
per *token chunk* of size ≤ 8 by design (with `grid_dim.y` doing the
chunking) — but the single-CTA call at `builder.py:909` (`grid_dim=(1,1,1)`)
indicates the V3 routing IS called with a single CTA. **Action:** Wave-2
implementer must (a) instrument to count actual rows-per-call seen in
practice, (b) if `> 8` is hit, add an outer `for (int row_base = 0; row_base
< num_rows; row_base += WARPS_PER_CTA * ROWS_PER_WARP)` loop, (c) backport
the fix to V3.

**Citation:** `include/mirage/persistent_kernel/tasks/blackwell/topk_sigmoid_sm100.cuh:107, 151, 157, 160, 165`.

---

### Q11: `residual` lives in HC space (moe.md L1186)

**Question (verbatim):**
> the V4 `Block` orchestration adds two HC `mhc_post` calls per layer (one
> after attention, one after MoE). The `residual` argument of
> `moe_mul_sum_add_layer` therefore lives in *HC space* (`[T, hc, D]`).

**Resolution:** RESOLVED.

**Answer:** The MoE itself writes a **flat** `[T, D]` output; the HC
reshuffle is owned by `mhc_post` (Wave-2 task in `hc.md`). Cross-spec
consistency is preserved: `moe_mul_sum_add_layer` takes a flat residual
`[T, D]` input, produces `[T, D]` flat output, and the subsequent
`mhc_post` call (`hc.md` task list) re-expands into `[T, hc, D]`.

This matches the official `model.py:692-698` flow:
```python
x = self.attn_norm(x)
... # attention runs on flat x
x = self.ffn_norm(x)
... # MoE runs on flat x
```
The hc dimension is folded in *between* blocks via `mhc_post`, not inside
attention or MoE.

**Citation:** `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:692-698`.

---

### Q12: `_zero_bias_E` for hash layers (moe.md L1193)

**Question (verbatim):**
> for hash layers, the `e_score_correction_bias` *does not exist* in the
> checkpoint (`Gate.__init__` at `model.py:560-562` skips it when
> `self.hash` is true).

**Resolution:** STRONG-GUESS.

**Answer:** Pass a pre-allocated zero buffer `_zero_bias_E` of shape `[E]`,
dtype matching the sigmoid/sqrtsoftplus path. This is semantically
identical to `bias=None` because the kernel computes `score + bias` and
zero is the additive identity. Confirmed by reading `model.py:574-575`:
```python
if self.bias is not None:
    scores = scores + self.bias
```
For hash layers `self.bias is None` per line 560-562. Passing a zero
buffer matches this code path numerically.

v2 may add a `nullptr`-allowed code path in
`register_moe_topk_sigmoid_sm100_task` for a small memory win, but for v1
the zero buffer is correct.

**Citation:** `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:558-562, 574-575`.

---

### Q13: Partial-batch zeroing (moe.md L1200)

**Question (verbatim):**
> when `T < max_num_batched_tokens` (partial batch), the routing matrix
> `[E, T]` has columns that are uninitialized beyond the live prefix.

**Resolution:** RESOLVED.

**Answer:** Adopted in §1.9 of `moe.md` already. The hash-route kernel
loops `for (int idx = threadIdx.x; idx < E * T; idx += blockDim.x)` to
zero all `E * T` cells, mirroring V3's
`topk_sigmoid_sm100.cuh:104-114`. No additional action needed.

**Citation:** `include/mirage/persistent_kernel/tasks/blackwell/topk_sigmoid_sm100.cuh:104-114`.

---

### Q14: Does layer 43 (MTP) use MoE? (moe.md L1207)

**Question (verbatim):**
> the V4 spec for `compress_ratios` says layer 43 (MTP) has
> `compress_ratio = 0`. Confirm the MTP block uses its own FFN (not MoE).

**Resolution:** RESOLVED — but **opposite of the spec-author's guess**.

**Answer:** The MTP block uses the **same MoE** (with shared + routed
experts) as base layers, NOT a dedicated single MLP. Evidence:

1. `model.py:739-746`: `class MTPBlock(Block): def __init__(...): super().__init__(layer_id, args); ...` — MTP inherits from `Block`. `Block.__init__` (line 658+) builds `self.ffn = MoE(layer_id, args) if layer_id >= args.n_dense_layers else MLP(args.dim, args.inter_dim)`. Since MTP's `layer_id = 43 >= n_dense_layers`, it gets `MoE`.

2. Checkpoint keys confirm:
   - `mtp.0.ffn.shared_experts.{w1,w2,w3}.{weight,scale}` (lines around 33-39 of the grep output)
   - `mtp.0.ffn.experts.{0..255}.{w1,w2,w3}.{weight,scale}` (255 routed experts × 3 GEMMs × 2 tensors)
   - `mtp.0.ffn.gate.weight`, `mtp.0.ffn.gate.tid2eid` if hash, else `mtp.0.ffn.gate.bias`.

3. The MoE-builder DOES run for `layer_idx = 43`.

This corrects the spec author's reading. Update `mtp.md` accordingly:
the MTP block runs the same `Block` pipeline (mhc_pre → attention →
mhc_post → mhc_pre → MoE → mhc_post), plus the MTP-specific
embed-fuse and head logic.

**Citation:** `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:739-767`;
`/raid/catalyst/models/DeepSeek-V4-Flash-Base/model.safetensors.index.json`
(grep `mtp.0.ffn`).

---

### Q15: Second paged-cache triple (attention.md L1498)

**Question (verbatim):**
> Wave-1 audit task per plan Phase A must confirm whether MPK's existing
> `paged_kv_indptr_buffer` triple can carry **two** logical caches by
> widening the indices, or whether we need a parallel
> `paged_compressed_kv_indptr_buffer` triple.

**Spec-author recommendation:** parallel-triple approach (default-accept).

**Resolution:** NEEDS-USER-DECISION.

**Answer:** Recommend the **parallel-triple** approach. Reasons:
1. The compressed-KV cache has a different page size and a different
   element size (head_dim=512 vs SWA head_dim=576 with rope tail). Stuffing
   both into one indices buffer requires a stride convention that
   inevitably leaks into every kernel that touches the cache.
2. Adding parallel triples is additive — defaults to empty when no V4
   layer is built — and changes only `persistent_kernel.py:48-50` plus
   `get_default_init_parameters`.

The plan §Phase A audit task should still run to confirm there's no
hidden coupling, but the structural choice is clear.

**Default if user does not respond:** parallel triples.

**Citation:** `python/mirage/mpk/persistent_kernel.py:48-50, 487-493`.

---

### Q16: FP8 vs bf16 SWA cache (attention.md L1506)

**Question (verbatim):**
> vLLM stores the SWA cache as FP8 (nope) + bf16 (rope) + per-block scales
> + pad; V3's MPK cache is bf16 throughout.

**Spec-author recommendation:** bf16 throughout for v1.

**Resolution:** NEEDS-USER-DECISION.

**Answer:** Recommend **bf16 throughout** for v1. Reasons:
1. Avoids forcing a quant kernel during cache insert, simplifying the
   fused `kv_norm + rope + insert` task in sparse.md.
2. V3's MPK MLA cache already uses bf16 (`kv_cache_size` allocation at
   `model.py:473-474` uses default bf16), so reuse is maximal.
3. The plan §Outcome targets 1-3 layers in v1, so the memory bloat
   (roughly 2× vs FP8) is bounded.

The trade-off is memory: bf16 nope (512B/head) vs FP8 nope (256B/head)
plus per-block scales. For 64 heads × 128 tokens window, that's the
difference between 4 MiB/layer and ~2.5 MiB/layer per request. Acceptable
in v1.

**Default if user does not respond:** bf16 throughout.

**Citation:** `deps/vllm/vllm/model_executor/layers/deepseek_v4_attention.py:217-223`
(vLLM FP8+bf16 layout); `model.py:473-474` (official bf16 cache).

---

### Q17: `combine_topk_swa_indices` location (attention.md L1514)

**Question (verbatim):**
> vLLM does this inside a Python helper on every prefill chunk. For v1 we
> propose to keep this on the **host** (Python builder).

**Resolution:** STRONG-GUESS.

**Answer:** Keep on host for v1. Justification:
- Prefill is not on the tight-loop critical path (decode is). The function
  only writes into a pre-allocated device buffer.
- vLLM's `deepseek_v4_attention.py:969-981` does it on host too.
- The fused-into-gather-task optimization is a well-defined v2 follow-up.

**Citation:** `deps/vllm/vllm/model_executor/layers/deepseek_v4_attention.py:969-981`.

---

### Q18: Attn_sink semantics across splits (attention.md L1521)

**Question (verbatim):**
> FlashMLA-Sparse gives the sink one shared softmax slot. In a split decode
> kernel, only one split should add the sink contribution.

**Resolution:** DEFER-TO-WAVE-2.

**Answer:** Two valid approaches: (a) only split-0 adds the sink, others
add 0 to their max-tracker; (b) each split adds `sink / num_splits` then
the reduce reassembles. Both are correct; (a) is simpler in code, (b) is
more symmetric. Implementer picks one and documents in the
`_generated_cuda/mla_v4_decode.cu` reference comments. The
`attn_sink` parameter shape is `[H]` per `model.py:456` (`self.attn_sink =
nn.Parameter(torch.empty(self.n_local_heads, dtype=torch.float32))`), so
the per-head sink scalar is added once into the per-head softmax max.

**Citation:** `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:456`.

---

### Q19: Padded heads (attention.md L1527)

**Question (verbatim):**
> vLLM pads `H` to 64 or 128 because FlashMLA-Sparse demands it. V4-Flash
> has H=64 exactly.

**Resolution:** RESOLVED.

**Answer:** `config.json:23` reports `num_attention_heads: 64`. Padding is
a no-op for V4-Flash. v1 hard-codes `padded_heads = 64`. Non-Flash
variants (Pro, etc.) can revisit when needed.

**Citation:** `/raid/catalyst/models/DeepSeek-V4-Flash-Base/config.json:23`.

---

### Q20: MTP decode reuse (attention.md L1533)

**Question (verbatim):**
> confirmed for v1 that MTP reuses `mla_v4_decode` with `compress_ratio=0`.

**Resolution:** RESOLVED.

**Answer:** Confirmed by:
1. `config.json:66` — `compress_ratios[43] = 0`.
2. `model.py:792` instantiates `MTPBlock(args.n_layers + 0, args)` so
   `layer_id = 43`.
3. `MTPBlock.__init__` calls `super().__init__(layer_id, args)` (line 742)
   which is `Block.__init__`. `Block.__init__` constructs
   `self.attn = Attention(layer_id, args)` which reads
   `args.compress_ratios[43] = 0` at line 453.
4. With `compress_ratio = 0`, no Compressor / no Indexer (line 466
   `if self.compress_ratio:` guards both).

So MTP attention is dense SWA, identical to `mla_v4_decode_layer` with
`HAS_COMPRESSED=false`. No new task needed for MTP attention.

**Citation:** `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:453, 466-471, 739-747, 786-792`.

---

### Q21: Inverse-RoPE sign convention (attention.md L1538)

**Question (verbatim):**
> The Triton kernel and PyTorch `apply_rotary_emb(inverse=True)` must give
> bit-identical results.

**Resolution:** DEFER-TO-WAVE-2.

**Answer:** PyTorch source for the inverse RoPE is
`deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:232-242`:
```python
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    ...
    if inverse:
        freqs_cis = freqs_cis.conj()
    ...
    x = torch.view_as_real(x * freqs_cis).flatten(-2)
```
Taking the complex conjugate of `freqs_cis` (which is unit-norm complex
in polar form, `cos + i*sin`) gives `cos - i*sin`. So the inverse-RoPE
formula is:
```
r_even =  x_even * cos + x_odd  * sin
r_odd  = -x_even * sin + x_odd  * cos
```
vs forward:
```
r_even = x_even * cos - x_odd  * sin
r_odd  = x_odd  * cos + x_even * sin
```
The implementer validates by round-tripping in the §5.9 test
(forward-then-inverse should give `x` within bf16 tolerance ≈ `5e-3`).

**Citation:** `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:232-242`.

---

### Q22: `tma_aligned_T` rounding (attention.md L1544)

**Question (verbatim):**
> `tma_aligned_T = round_up_4(B)` rounding interaction with the
> variable-T decode path.

**Resolution:** DEFER-TO-WAVE-2.

**Answer:** In decode, `T = B` (one query per request). For typical
`B ∈ {1, 2, 4, 8}`, `round_up_4(B) ∈ {4, 4, 4, 8}` — padding adds at
most 3 rows. The wo_a GEMM's input layout `o_fp8[d, T*d, 1]` per
`deepseek_v4_attention.py:326-334` was chosen for exactly this case;
padding rows are tolerated. Implementer verifies by feeding `B = {1, 5}`
into the §5.9 test.

---

### Q23: HF checkpoint key prefix for MTP weights (mtp.md L600)

**Question (verbatim):**
> Confirm the HuggingFace checkpoint key prefix for MTP weights.
> `layer_id = 43` but checkpoint storage may be either `model.layers.43.*`
> (V3-style) or a separate `model.mtp.0.*` namespace.

**Resolution:** RESOLVED.

**Answer:** Prefix is **`mtp.0.*`**, not `layers.43.*`. Confirmed by
`grep -E 'mtp|layers\.43' /raid/catalyst/models/DeepSeek-V4-Flash-Base/model.safetensors.index.json`
which returns zero `layers.43.*` keys and many `mtp.0.*` keys:
```
"mtp.0.hc_head_base": "model-00046-of-00046.safetensors",
"mtp.0.hc_head_fn": "model-00046-of-00046.safetensors",
"mtp.0.hc_head_scale": "model-00046-of-00046.safetensors",
"mtp.0.hc_attn_base": "model-00046-of-00046.safetensors",
"mtp.0.hc_ffn_base": "model-00046-of-00046.safetensors",
... (49 more mtp.0.* keys)
"mtp.0.attn.wq_a.weight": "model-00046-of-00046.safetensors",
"mtp.0.attn.wq_a.scale": "model-00046-of-00046.safetensors",
"mtp.0.attn.wq_b.weight": "model-00046-of-00046.safetensors",
"mtp.0.attn_norm.weight": "model-00046-of-00046.safetensors",
"mtp.0.ffn_norm.weight": "model-00046-of-00046.safetensors",
"mtp.0.ffn.shared_experts.*", "mtp.0.ffn.experts.{0..255}.*",
"mtp.0.ffn.gate.weight",
"mtp.0.e_proj.{weight,scale}", "mtp.0.h_proj.{weight,scale}",
"mtp.0.enorm.weight", "mtp.0.hnorm.weight",
```
All MTP weights live in shard 46-of-46.

**Citation:** `/raid/catalyst/models/DeepSeek-V4-Flash-Base/model.safetensors.index.json` (grep `mtp.0`).

---

### Q24: FP8 for `e_proj` / `h_proj`? (mtp.md L610)

**Question (verbatim):**
> Does the MTP layer actually use FP8 quantization for `e_proj` and
> `h_proj`?

**Resolution:** RESOLVED.

**Answer:** **Yes.** The index has both `mtp.0.e_proj.weight` and
`mtp.0.e_proj.scale` (and similarly for `h_proj`). The presence of the
`.scale` companion is the on-disk signature of FP8 storage with UE8M0 block
scales (per `config.json:35-43` `quantization_config`:
`fmt=e4m3, scale_fmt=ue8m0, weight_block_size=[128,128]`).

Strategy A1 must therefore use `linear_fp8_layer` +
`linear_fp8_with_residual_layer` (existing MPK V3 APIs at
`persistent_kernel.py linear_fp8_layer`) for `e_proj` and `h_proj`, NOT
bf16 `linear_layer`. The convert script reads `mtp.0.e_proj.weight` as
`float8_e4m3fn` and `mtp.0.e_proj.scale` as the block-scale tensor.

**Citation:** `/raid/catalyst/models/DeepSeek-V4-Flash-Base/model.safetensors.index.json` (lines around `mtp.0.e_proj.*`, `mtp.0.h_proj.*`);
`config.json:35-43`.

---

### Q25: Strategy A1 weight replication (mtp.md L620)

**Question (verbatim):**
> Should Strategy A1's `e_proj` weight replication (`hc * D * D * 2 = 128
> MiB`) live in the *converter* (offline) or be skipped in favor of
> Strategy A2 (one bf16 add per hc copy)?

**Spec-author recommendation:** A1 in v1, revisit after profiling.

**Resolution:** NEEDS-USER-DECISION.

**Answer:** Recommend **A1 in v1**. 128 MiB is a one-off VRAM cost; if v1
targets 1-3 layers per plan §Outcome, the GPU has ample headroom. A1's
task count is minimal — `e_proj` is just `hc` independent FP8 GEMMs run
in parallel. A2 saves memory but introduces one extra bf16 add per hc
copy in the hot path; without profiling we can't size the cost.

**Default if user does not respond:** A1.

---

### Q26: MTP `attn_norm` / `ffn_norm` use base Block path (mtp.md L625)

**Question (verbatim):**
> The MTP block's *attention norm* and *ffn norm* — does V4 use them
> identically to the base layer Block?

**Resolution:** RESOLVED.

**Answer:** **Yes, identical.** Evidence:
1. `model.py:739-746`: `MTPBlock(Block).__init__` calls `super().__init__`,
   which constructs `self.attn_norm` and `self.ffn_norm` (Block.__init__ at
   `model.py:659-660`).
2. Checkpoint keys `mtp.0.attn_norm.weight` and `mtp.0.ffn_norm.weight`
   exist and have the standard `[dim=4096]` shape per `args.dim=4096`
   (config.json:13).
3. The MTP forward at `model.py:765` calls `super().forward(x, start_pos,
   input_ids)` which runs the standard Block pipeline including both
   norms.

The on-disk key paths are `mtp.0.attn_norm.weight` and
`mtp.0.ffn_norm.weight` — converter must map these to MPK's MTP block
weight slots.

**Citation:** `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:659-660, 739-746, 765`.

---

### Q27: MTP paged-KV cache shared or private? (mtp.md L633)

**Question (verbatim):**
> Are there any MTP-specific buffer-layout differences (separate paged-KV
> cache for the MTP block?).

**Resolution:** RESOLVED.

**Answer:** **Shared cache, MTP slot is layer_id 43.** Evidence:
1. `model.py:741`: `super().__init__(layer_id, args)` is called with
   `layer_id = args.n_layers + 0 = 43`.
2. `Block.__init__` builds `self.attn = Attention(layer_id, args)` which
   constructs its own `self.kv_cache` per layer at line 473-474.
3. In the official PyTorch reference there is no shared global cache —
   each layer (including MTP) has its own per-layer `kv_cache` buffer.
4. In MPK's paged design, this maps to per-layer slots in
   `paged_kv_indptr_buffer`. The MTP layer is just layer_id 43 in the
   index space.

So no extra paged-KV triple is needed for MTP; the existing per-layer
slot indexing handles it as another `[2, num_pages, page_size, head_dim]`
slice.

**Citation:** `model.py:438-441, 473-474, 741`.

---

### Q28: `_mtp_hidden_buffer` implies MPK buffer? (mtp.md L644)

**Question (verbatim):**
> Does `_mtp_hidden_buffer` (vLLM `deepseek_v4.py:1296`) imply MPK needs a
> buffer that *captures* the pre-hc_head HC residual of the base model
> and *re-feeds* it into the MTP block?

**Resolution:** RESOLVED.

**Answer:** **No.** vLLM's `_mtp_hidden_buffer` exists because vLLM runs
the base model and MTP draft in **two separate captured CUDA graphs**, so
the data must round-trip through an explicit HBM buffer between graph
invocations. MPK runs everything in **one persistent kernel**, so the
base model's pre-`mhc_head` HC tensor can be wired directly as
`prev_x_hc` input to the MTP block via the existing kn_graph DAG.

The MPK builder code should:
- Take the output of the last base-block `mhc_post` (the `[T, hc, D]` HC
  tensor *before* `mhc_head` collapses it).
- Pass it as the `h` input to the MTP-block construction.
- The base model also forks off a `mhc_head` path that consumes the same
  HC tensor — no copy needed; both consumers read the same DTensor.

**Citation:** vLLM `deps/vllm/vllm/model_executor/models/deepseek_v4.py:1296` (the buffer); MPK persistent-kernel single-graph model in `python/mirage/mpk/persistent_kernel.py`.

---

### Q29: MTP `compress_ratios[43]` interpretation (overview.md L774)

**Question (verbatim):**
> `config.json:66` supplies a 44-entry array with entry 43 = 0; the
> official `model.py` passes `args.compress_ratios` to `Attention.__init__`
> directly (`model.py:453`), so MTP attention is dense SWA. vLLM diverges
> (forces `compress_ratio = 1`).

**Resolution:** RESOLVED.

**Answer:** `config.json:66` confirms `compress_ratios[43] = 0`. The
official semantics: `compress_ratio = 0` ⇒ no Compressor, no Indexer, no
compressed cache (per `model.py:466` `if self.compress_ratio:` guards
both). vLLM's `max(1, ratio) = 1` is purely an internal indexing trick to
avoid divide-by-zero in their MLA spec; numerics are the same (dense
SWA). v1 follows the **official** convention (use `compress_ratios[43] =
0` directly).

**Citation:** `/raid/catalyst/models/DeepSeek-V4-Flash-Base/config.json:66`;
`deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:453, 466-471`.

---

### Q30: vLLM `fused_wqa_wkv` vs official separate linears (overview.md L784)

**Question (verbatim):**
> vLLM fuses `wq_a + wkv` into `fused_wqa_wkv`; the official `model.py`
> keeps them separate. v1 spec uses the separate-linear semantics.

**Resolution:** RESOLVED.

**Answer:** The on-disk checkpoint stores them **separately** — confirmed
by grep of `model.safetensors.index.json`:
```
"layers.0.attn.wq_a.weight": ...
"layers.0.attn.wq_a.scale":  ...
"layers.0.attn.wkv.weight":  ...
"layers.0.attn.wkv.scale":   ...
```
(and same for `layers.{1..42}.attn.*` and `mtp.0.attn.*`).

vLLM's `fused_wqa_wkv` is a runtime fusion done in
`deepseek_v4.py:966-972, 1344-1345` (the loader maps the two separate
checkpoint shards into one fused parameter at load time). For MPK v1 we
load them separately, matching the on-disk shape. The MPK `linear_fp8`
GEMMs run them as two distinct ops; a future MPK-side fusion is an
optional perf optimization.

**Citation:** `/raid/catalyst/models/DeepSeek-V4-Flash-Base/model.safetensors.index.json` (`layers.0.attn.wq_a.*`, `layers.0.attn.wkv.*`);
`deps/vllm/vllm/model_executor/models/deepseek_v4.py:966-972, 1344-1345`.

---

### Q31: `mhc_prenorm_gemm` fallback path (overview.md L792)

**Question (verbatim):**
> Plan §Wave-2 row 1 says the initial implementation may decompose into
> RMSNorm-then-linear if porting the CUTLASS prenorm GEMM is high-risk.

**Spec-author recommendation:** Python signature accepts both; Wave-2
chooses.

**Resolution:** NEEDS-USER-DECISION.

**Answer:** Recommend v1 ships the **decomposed** path (RMSNorm-then-linear).
Reasons:
1. Porting `sm100_tf32_hc_prenorm_gemm.cuh` from DeepGEMM is a substantial
   kernel rewrite (TF32 tensor-core dispatch, fused per-token sqrsum
   reduction). High risk per plan.
2. The decomposed path reuses existing MPK kernels (`rmsnorm` from V3 +
   `linear` from V3) entirely. Almost zero new code.
3. Performance is worse (extra HBM round-trip for the rsqrt/scale tensor)
   but v1's target is correctness on 1-3 layers, not throughput.

The Python `mhc_prenorm_gemm_layer` signature accepts both paths via a
`fused: bool = False` kwarg. v2 enables `fused=True` after the kernel
port lands.

**Default if user does not respond:** decomposed path.

---

### Q32: `repeat_to_hc` materialization (overview.md L801)

**Question (verbatim):**
> `model.py:806` materializes the hc-expanded hidden state by
> `unsqueeze(-2).repeat(...)`. In MPK we intend to fold this broadcast
> into the first `mhc_prenorm_gemm` consumer.

**Resolution:** STRONG-GUESS.

**Answer:** Fold into the first `mhc_prenorm_gemm` consumer. Setting the
`hc` stride to 0 makes the broadcast free in both global memory and
register loads. This avoids the `hc_mult=4` × memory blowup for the
hidden state (4× `[T, D]` = 4× `[T, 4096] bf16` = ~64 MiB for T=8192).

Wave-2 author of `mhc_prenorm_gemm` confirms the kernel handles
zero-stride input by checking the DTensor stride before computing the
input pointer. The MPK DTensor abstraction already supports zero-stride
broadcast dims (via the `(-1, ...)` axis in `tb.new_input`).

**Citation:** `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:806`.

---

### Q33: Paged-KV-cache extensions for V4 (overview.md L809)

**Question (verbatim):**
> Plan §Phase A includes a separate audit task to confirm MPK's
> `paged_kv_indptr_buffer`, `paged_kv_indices_buffer`, and
> `paged_kv_last_page_len_buffer` handle the Compressor's **variable-stride
> compressed-KV pages** and the **Indexer's fp4-format pages**.

**Resolution:** NEEDS-USER-DECISION.

**Answer:** This is a pointer to a dedicated audit task (plan §Phase A),
not a question this consolidated doc can answer. Recommend the audit task
runs before Wave 2 starts on sparse.md kernels. Tentative conclusion:
- Compressor compressed-KV pages: `block_size` likely fixed (per spec
  candidate values `4` or `64`) — variable-stride concerns may be a red
  herring.
- Indexer FP4 pages: not in v1 scope (v1 uses FP8 cache per Q36/Q43).

**Default if user does not respond:** run the audit task before Wave 2.

**Citation:** `python/mirage/mpk/persistent_kernel.py:48-50, 1401-1420` (existing paged-KV API).

---

### Q34: UE8M0 scale conversion for routed experts (overview.md L817)

**Question (verbatim):**
> vLLM stores scales as `float8_e8m0fnu` on disk and views them as `uint8`
> to preserve raw exponent bytes. MPK's `quantize_fp8_layer(scale_ue8m0=True)`
> produces packed `uint32` scales.

**Resolution:** RESOLVED.

**Answer:** Use the existing MPK pipeline; the conversion is mechanical.
- On disk: `float8_e8m0fnu` (1 byte per scale), `weight_block_size =
  [128, 128]` per `config.json:35-43`. For a weight tensor of shape
  `[N, K]`, the scale tensor has shape `[ceil(N/128), ceil(K/128)]` of
  e8m0 bytes.
- vLLM: views as `uint8` (`deepseek_v4.py:1383-1388`) — same bytes.
- MPK: `persistent_kernel.py:1974-1988` packs 4 consecutive e8m0 bytes
  into one `uint32` (little-endian), for `[N/128, K/128/4]` `uint32`.
- Conversion in `demo/deepseek_v4/models/convert.py`:
  ```python
  scale_bytes = scale_tensor.view(torch.uint8)  # [N/128, K/128]
  # reshape so K/128 dim is divisible by 4
  assert (K // 128) % 4 == 0
  packed = scale_bytes.view(N // 128, K // 128 // 4, 4)
  scale_u32 = (packed[..., 0].to(torch.uint32)
              | (packed[..., 1].to(torch.uint32) << 8)
              | (packed[..., 2].to(torch.uint32) << 16)
              | (packed[..., 3].to(torch.uint32) << 24))
  # final shape [N/128, K/128/4] uint32
  ```
- For V4-Flash: `K = 4096` ⇒ `K/128 = 32` ⇒ `32/4 = 8` `uint32` per row.
  `N/128` rows.

**Citation:** `python/mirage/mpk/persistent_kernel.py:1974-1988`;
`/raid/catalyst/models/DeepSeek-V4-Flash-Base/config.json:35-43`;
`deps/vllm/vllm/model_executor/models/deepseek_v4.py:1383-1388`.

---

### Q35: Hadamard transform implementation in `indexer_q_transform` (overview.md L825)

**Question (verbatim):**
> Plan §Out-of-scope item 5 explicitly allows v1 to compute the Hadamard
> as a matmul against a precomputed Hadamard matrix instead of a
> fast-transform.

**Resolution:** RESOLVED.

**Answer:** Two distinct sites, two different policies:

1. **Indexer-Q Hadamard:** **Absorbed offline into `wq_b`** by the convert
   script. The kernel `indexer_q_transform_layer` does NOT compute the
   Hadamard at runtime. Per vLLM `deepseek_v4_attention.py:1066-1072`,
   `DeepseekCompressor.fused_wkv_wgate` pre-rotates the weight as `H · wq_b`
   so the runtime GEMM produces an already-rotated Q. The official
   PyTorch reference computes it at runtime (`model.py:414`) — semantic
   difference, but algebraically equivalent because Hadamard distributes
   over the linear.

2. **Compressor (head_dim=128) Hadamard:** **Cannot be absorbed**
   (see Q40), so kept as a runtime matmul against a `__constant__` bf16
   table. Shape `[128, 128]`, 16 KiB. Initialized by Sylvester
   construction (`H_n = H_{n/2} ⊗ H_2`).

**Citation:** `deps/vllm/vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py:67-169`;
`deps/vllm/vllm/model_executor/layers/deepseek_v4_attention.py:1066-1072`;
`deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:247-251, 414`.

---

### Q36: FP8 indexer cache for v1 (sparse.md L181)

**Question (verbatim):**
> Confirm with reviewers that FP8 indexer cache is the v1 target.

**Resolution:** NEEDS-USER-DECISION.

**Answer:** Recommend **FP8 in v1**, per plan
`i-want-to-add-dapper-pascal.md:432-433`. MXFP4 (`use_fp4_indexer_cache`
default in vLLM `deepseek_v4_attention.py:1059`) is fully specified in
`sparse.md` but stays v2 scope.

Reasons:
- The FP8 path matches MPK's existing FP8 quant kernels (reuse
  `per_token_group_quantize_fp8.cuh` helpers per sparse.md §I).
- The MXFP4 path requires a separate PTX helper (`fused_indexer_q.py:27-42`)
  and 4-bit-packed pages — a new code surface.

**Default if user does not respond:** FP8 indexer cache.

**Citation:** Plan `i-want-to-add-dapper-pascal.md:432-433`;
`deps/vllm/vllm/model_executor/layers/deepseek_v4_attention.py:1059`.

---

### Q37: Hadamard constant initialization (sparse.md L234)

**Question (verbatim):**
> decide between (a) host-side init via `cudaMemcpyToSymbol` and (b)
> literal-embedded constant table in the codegen, for the Hadamard 128×128
> bf16 constant.

**Resolution:** STRONG-GUESS.

**Answer:** Recommend **(b) literal-embedded constant table in codegen**.
Reasons:
1. Avoids new runtime wiring — no host-side cudaMemcpyToSymbol call
   needs to fit into MPK's PersistentKernel initialization path.
2. 16 KiB embedded as a `__device__ __constant__ bf16 H[128*128] = {...}`
   in the generated `.cuh` is well within const memory limits (64 KiB on
   SM100).
3. The matrix is deterministic (Sylvester construction with no
   randomization for `H_128`), so embedding it as a literal is safe.
4. Generation is mechanical: `task_register.cc` (or a helper Python
   script) emits the constant as a `{ ... }` array initializer in the
   generated `.cuh`.

**Citation:** —.

---

### Q38: Two-task `indexer_score_sm100 → indexer_topk_sm100` compatibility (sparse.md L432)

**Question (verbatim):**
> Verify Option A's two-task design is compatible with MPK's
> `kn_graph.register_task` API.

**Resolution:** RESOLVED.

**Answer:** Compatible. Precedent: `mla_kv_gather_layer` (registered as
`mla_kv_gather_sm100` at `persistent_kernel.py:1420`) → `mla_decode_layer`
(registered as `mla_decode_sm100` at `persistent_kernel.py:1481`). Both
are registered via separate `kn_graph.register_task` calls on the same
KNGraph. The MPK scheduler enforces producer→consumer ordering through
the kernel-graph DAG (the second task's `customized(...)` input includes
the first task's output DTensor as an explicit input, which creates the
ordering edge).

For `indexer_score_sm100 → indexer_topk_sm100`:
1. `indexer_score_sm100` produces an `[T, num_compressed_positions] fp32`
   logits buffer.
2. `indexer_topk_sm100` consumes that buffer (as input) and produces
   `topk_indices`.
3. The Python builder passes the logits DTensor as input to step 2,
   creating the DAG edge.

**Citation:** `python/mirage/mpk/persistent_kernel.py:1401-1481` (the
chain).

---

### Q39: Indexer Q RoPE theta (sparse.md L976, L1603)

**Question (verbatim):**
> verify whether V4-Flash uses base rope_theta or compress_rope_theta for
> the Indexer's Q RoPE.

**Resolution:** RESOLVED.

**Answer:** **`compress_rope_theta = 160000`** (config.json:65).

Trace:
1. `model.py:493-494`:
   ```python
   if self.indexer is not None:
       self.indexer.freqs_cis = self.freqs_cis
   ```
   The Indexer takes its `freqs_cis` from the parent Attention layer.

2. `model.py:476-481`:
   ```python
   if self.compress_ratio:
       original_seq_len, rope_theta = args.original_seq_len, args.compress_rope_theta
   else:
       original_seq_len, rope_theta = 0, args.rope_theta
   freqs_cis = precompute_freqs_cis(self.rope_head_dim, args.max_seq_len, original_seq_len,
                                    rope_theta, args.rope_factor, args.beta_fast, args.beta_slow)
   ```
   The Attention layer's `freqs_cis` uses `compress_rope_theta` when
   `compress_ratio > 0`.

3. `model.py:469`: `self.indexer = Indexer(...)` only when
   `self.compress_ratio == 4`. So whenever the Indexer exists, the parent
   Attention has `compress_ratio = 4 > 0`, hence uses `compress_rope_theta`.

Therefore the Indexer's Q RoPE uses `compress_rope_theta = 160000`,
confirmed by `config.json:65`. (Note: `config.json` lists value 160000;
the official `model.py:67` default is 40000.0 — V4-Flash overrides via
config.)

**Citation:** `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:469, 476-481, 493-494`;
`/raid/catalyst/models/DeepSeek-V4-Flash-Base/config.json:65`.

---

### Q40: Compressor Hadamard absorbable into `wkv`? (sparse.md L1058, L1612)

**Question (verbatim):**
> confirm that the Compressor's head=128 Hadamard cannot similarly be
> absorbed; the Compressor rotates `kv` *after* the gated softmax pool,
> so absorbing into `wkv` is not equivalent.

**Resolution:** RESOLVED.

**Answer:** **Confirmed cannot be absorbed.**

The Compressor applies the Hadamard rotation `kv = H @ kv` *after* a
gated-softmax pool that collapses the `coff * head_dim` input dim to
`head_dim`:
```python
# Compressor.forward, model.py:340-367
kv = self.wkv(x)  # [B, S, coff*head_dim]
gates = softmax(self.wgate(x), dim=-1)  # [B, S, coff], per-token softmax
kv = (kv.view(B, S, coff, head_dim) * gates.unsqueeze(-1)).sum(dim=-2)  # [B, S, head_dim]
kv = rotate_activation(kv)  # H @ kv, [B, S, head_dim]
```
Pre-multiplying H into `wkv` would give `(H @ wkv) @ x = H @ (wkv @ x)`,
which only equals `H @ (gate · (wkv @ x))` if the gate weights are
position-invariant — but `gates` are per-token softmax outputs, so they
vary per `(b, s)`. The Hadamard cannot commute past the per-token gated
sum.

For the Indexer-Q the situation differs: `q = wq_b @ qr` is followed
*directly* by `rotate_activation(q)`, with no per-token interleaving. So
`H @ (wq_b @ qr) = (H @ wq_b) @ qr`, and the Hadamard can be absorbed
into the weight offline.

**Citation:** `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:340-367, 411-414`.

---

### Q41: `weights_proj` output sign (sparse.md L1217, L1620)

**Question (verbatim):**
> confirm `weights_proj` output can be negative; if it can, the
> relu-then-weight ordering matters and we must follow the official's
> order to match numerics.

**Resolution:** RESOLVED.

**Answer:** **Output can be negative.**

`weights_proj` is constructed at `model.py:394` as:
```python
self.weights_proj = ColumnParallelLinear(self.dim, self.n_heads, dtype=torch.bfloat16)
```
This is a plain bf16 linear with:
- No activation (no ReLU, sigmoid, abs, softplus, etc.).
- No bias clamping.
- No quantization that would constrain sign.

The output is bf16, free to be negative or positive. The downstream usage
at `model.py:418, 421`:
```python
weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
```
applies `relu` to the score *before* multiplying by `weights`, then sums
over heads. This ordering matters: if a head's `weights[t, h] < 0`, the
official multiplies a non-negative `relu(score)` by a negative scalar,
contributing negative mass to `index_score`. The vLLM ordering
(weight-then-relu) would zero this contribution. **Not the same
numerics.**

v1 MPK follows the **official's** ordering for correctness.

**Citation:** `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:394, 418-421`.

---

### Q42: Heap-based fused score+topk (sparse.md L1317)

**Question (verbatim):**
> v1 should pick the heap-based fused task (simpler, no large intermediate
> buffer); revisit if profiling shows the heap update is a bottleneck.

**Resolution:** DEFER-TO-WAVE-2.

**Answer:** For v1, the spec's recommendation (heap-based) is reasonable:
- `K = 512` is fixed (per plan `i-want-to-add-dapper-pascal.md:55`).
- Per-token candidate count ≤ `4096` in v1 tests (max_seq_len = 32768
  with R = 4 ⇒ 8192 positions; v1 likely tests smaller).
- A size-512 streaming top-K heap in shared memory occupies
  `512 * 8 bytes = 4 KiB` (key + index pairs), well within SM100's 228 KiB
  shared memory.

Implementer profiles and either keeps the heap or splits into
`indexer_score_sm100 + indexer_topk_sm100` two-task form per §F.3.3.

---

### Q43: Duplicate of Q36 (sparse.md L1588)

**Question:** FP8 indexer cache for v1. **Resolution:** NEEDS-USER-DECISION,
same answer as Q36 (FP8 in v1).

---

### Q44: Duplicate of Q38 (sparse.md L1594)

**Question:** Two-task `kn_graph.register_task` chain compatibility.
**Resolution:** RESOLVED, same answer as Q38 (compatible, precedent in
`mla_kv_gather_layer → mla_decode_layer`).

---

### Q45: Duplicate of Q37 (sparse.md L1599)

**Question:** Hadamard constant initialization. **Resolution:**
STRONG-GUESS, same answer as Q37 (literal-embedded codegen).

---

### Q46: Duplicate of Q39 (sparse.md L1603)

**Question:** Indexer Q RoPE theta. **Resolution:** RESOLVED, same answer
as Q39 (`compress_rope_theta = 160000`).

---

### Q47: Duplicate of Q40 (sparse.md L1612)

**Question:** Compressor Hadamard absorbable? **Resolution:** RESOLVED,
same answer as Q40 (cannot be absorbed).

---

### Q48: Duplicate of Q41 (sparse.md L1620)

**Question:** `weights_proj` output sign. **Resolution:** RESOLVED, same
answer as Q41 (can be negative; follow official ordering).

---

### Q49: DeepGEMM `fp8_fp4_paged_mqa_logits` source path (sparse.md L1625)

**Question (verbatim):**
> pin down the source path for DeepGEMM's `fp8_fp4_paged_mqa_logits`
> CUDA implementation under
> `deps/vllm/vllm/third_party/deep_gemm/include/deep_gemm/impls/`.

**Resolution:** RESOLVED.

**Answer:** Three files found by `find`:
- `deps/vllm/vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp8_paged_mqa_logits.cuh`
  — SM100 FP8 path. **This is the v2 optimization target.**
- `deps/vllm/vllm/third_party/deep_gemm/include/deep_gemm/impls/sm90_fp8_paged_mqa_logits.cuh`
  — SM90 (Hopper) variant; not relevant for v4 Blackwell.
- `deps/vllm/vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp4_paged_mqa_logits.cuh`
  — SM100 MXFP4 variant; v2+ if FP4 cache path is enabled.

v1 does NOT port these kernels — it writes a naive equivalent per
sparse.md §F.3.3. The spec now cites the exact paths for v2 follow-up.

**Citation:** `deps/vllm/vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp8_paged_mqa_logits.cuh` (and `_fp4_` variant).

---

### Q50: Compressed-KV cache block size (sparse.md L1631)

**Question (verbatim):**
> the `kv_cache_block_size=64` for the head=512 compressed-KV cache is
> the *V3* value; verify V4-Flash uses the same page size for the
> compressed-KV cache or if it should adopt the
> `SlidingWindowMLASpec(block_size=4, ...)` 4-token pages.

**Resolution:** NEEDS-USER-DECISION.

**Answer:** Two candidates, both viable:
- **`block_size = 64`** (V3 value): maximizes MPK reuse of existing
  paged-attention kernels.
- **`block_size = 4`** (vLLM `SlidingWindowMLASpec` per
  `deepseek_compressor.py:165-169`): matches vLLM's V4 wire format,
  smaller pages.

Recommend **`block_size = 64`** for v1 because:
1. Reuse of V3 page-table iteration code.
2. The compressed cache is small (`max_seq_len / R = max_seq_len / 4`
   tokens per request) — page granularity doesn't matter much.
3. If a numerical mismatch with vLLM's `SlidingWindowMLASpec` is detected
   in module tests (sparse.md §F.1.10), switch to 4.

**Default if user does not respond:** block_size = 64.

**Citation:** `deps/vllm/vllm/model_executor/layers/deepseek_compressor.py:165-169`.

---

### Q51: Corner test "head_dim=128, compress_ratio=128" (sparse.md L1637)

**Question (verbatim):**
> the corner test "head_dim=128, compress_ratio=128" exercises the
> `coff=1` code path with the indexer head width — confirm it's worth the
> implementation effort.

**Resolution:** DEFER-TO-WAVE-2.

**Answer:** This combination doesn't exist in the real config
(`compress_ratios[i] ∈ {0, 4, 128}` per `config.json:66`, but `head_dim
= 128` is the Indexer width only — and the Indexer only runs when
`compress_ratio == 4`, never `128`). So this is a pure code-coverage test
of the `coff = 1` branch in the head_offset arithmetic. Skip the corner
test for v1 unless the implementer hits an off-by-one bug during the §F.1
implementation, in which case adding the test is cheap.

**Citation:** `/raid/catalyst/models/DeepSeek-V4-Flash-Base/config.json:66`.

---

## End of consolidated open-questions document
