# DeepSeek V3 on the new `mirage.mpk.layers` catalog — FP8, decode-only (HANDOFF)

> Self-contained status + roadmap for continuing this work in a fresh session
> on a new machine. Read this top-to-bottom; it assumes no prior context.

Branch: **`feat/new-api`**. Conda env: **`mirage`**. Target GPU: **B200 / SM100 (CC 10.0)**.

---

## 1. Goal

Replace the lengthy, hard-to-review legacy DeepSeek V3 builder
(`python/mirage/mpk/models/deepseek_v3/builder.py`, a monolith of direct
`pk.*_layer()` calls) with a clean, HF-aligned, testable model definition on the
**new object-oriented catalog API** (`python/mirage/mpk/layers/`), where every
block is an `MPKModule` with:
- `forward()` — a pure-PyTorch reference (the correctness oracle), and
- `compile()` — MPK task registration (composes catalog layers / `pk.*` primitives).

Reference template for the new API: `python/mirage/mpk/models/qwen3/modeling.py`
+ `demo/qwen3/demo_new.py`.

PyTorch reference to mirror (module decomposition + `forward()` math):
**`<conda>/envs/mirage/lib/python3.12/site-packages/transformers/models/deepseek_v3/modular_deepseek_v3.py`**
(classes: `DeepseekV3MLP`, `DeepseekV3TopkRouter`, `DeepseekV3MoE`,
`DeepseekV3Attention` (MLA), `DeepseekV3DecoderLayer`, `DeepseekV3Model`,
`DeepseekV3ForCausalLM`).

---

## 2. Locked design decisions (v1 scope)

1. **FP8 weights** (matches the legacy demo; DSv3 ships FP8).
2. **Tested single-GPU first, but TP/EP-aware via context**: read
   `current_pk().parallel_config` (tp_size/ep_size/ranks) in `__init__`; only
   build `AllReduce` / shard weights when `world_size>1` (mirror Qwen3). At
   `world_size==1` these are no-ops, so single↔multi-GPU is a config change.
3. **MoE = new permute-based path** (the `MPK_DSV3_NEW_MOE=1` composition).
4. **Match legacy-demo default fusions** where they matter; `KV_GATHER_SPLITS=8`
   is prefill-only (no-op in decode-only v1).
5. **Decode-only, no MTP, no prefill construction, naive pre-allocated KV.**
6. **Oracle = HF `modular_deepseek_v3.py`**: each `MPKModule.forward()` is a
   faithful HF port (dequantizing its own FP8 weights). Validate **bottom-up**:
   leaf/module `compile()` output vs its `forward()`, climbing leaf → module →
   decoder layer → model → end-to-end.

---

## 3. CRITICAL: FP8 scale-layout map (the biggest trap — root-caused from kernel source)

The UE8M0 scale layout is path-specific and the catalog docstrings are partly
WRONG. Ground truth (from `.cuh` source):

- **`quantize_fp8_sm100`** (catalog `QuantizeFP8UE8M0`, `pk.quantize_fp8_layer(scale_ue8m0=True)`)
  writes UE8M0 scale **COLUMN-major / K-outer** `[packed_k, aligned_batch]`
  — `tasks/blackwell/per_token_group_quantize_fp8.cuh:284-300`. (Docstrings
  claiming "M-outermost" / "TransposeScale bridges to K-outer" are backwards.)
- **`linear_fp8_swapAB_sm100`** reads `input_scale` **ROW-major / M-outer**
  `[batch, packed_k]` — `linear_fp8_swapAB_sm100.cuh:286`. ⇒ `QuantizeFP8UE8M0 →
  swapAB` is **BROKEN for packed_k>1** (they coincide only at packed_k=1).
  `TransposeScale` goes M→K (wrong direction to fix swapAB). **Do not pair them.**
- **`fp8_group_gemm_*`** wants K-outer ⇒ `QuantizeFP8UE8M0 → fp8_group_gemm` is
  CONSISTENT (the MoE permute path; no transpose).
- **f32-scale path** (`scale_ue8m0=False`) writes **M-outer row-major
  `(M, K/128)`** — `per_token_group_quantize_fp8.cuh:262-263` — unambiguous.

### The two verified FP8 GEMM recipes
- **Dense (single) GEMM** (MLP, shared expert, MLA dense proj):
  `QuantizeFP8F32Scale` (f32 1×128 M-outer) →
  `pk.fp8_gemm_dense_smallm_layer(input_fp8, weight_fp8, input_scale, weight_scale, output, num_workers)`.
  Weight scale = **f32 128×128 block `(N/128, K/128)`** (DSv3 native). Residual:
  GEMM→partial then `pk.elementwise_add_layer`. (`max_seq_length<=512` → smallm,
  else mediumm.) Running recipe:
  `tests/runtime_python/blackwell/sm100_fp8_gemm_dense/test_fp8_gemm_dense_smallm_pk_testmode.py`.
- **Grouped (per-expert) GEMM** (routed MoE): `FP8GroupGEMMSmallM` (catalog,
  owns `(E,N,K)` uint8 weight + `(num_sf_k, E*N)` K-outer uint32 scale, has
  `forward()`); activation scale is K-outer (from the permute or from
  `pk.quantize_fp8_layer(scale_ue8m0=True)` with output declared `(K_PACKED, M)`).

`num_sf_k = ceil((K/128)/4)`. `_packed_scale_k(K)` helper in modeling.py.

---

## 4. DONE + VERIFIED on B200 (with test commands)

All under `tests/runtime_python/test_mode/`, run as:
`CUDA_VISIBLE_DEVICES=<free-gpu> conda run -n mirage python <test>`

| Component | Test | Result |
|---|---|---|
| `FusedRMSNormQuantizeFP8` (NEW catalog layer) | `test_fused_rmsnorm_quantize_fp8_testmode.py` | PASS (fp8 rel 0.033, bf16 rel 0.003) |
| `DeepseekV3MLP` (FP8 dense) | `test_dsv3_mlp_fp8_testmode.py` | PASS (rel 0.039) |
| MoE permute **chain** (isolation) | `test_dsv3_moe_permute_chain_testmode.py` | PASS (rel 0.0075) |

- **New catalog layer added**: `python/mirage/mpk/layers/norm/rmsnorm_quantize_fp8.py`
  (`FusedRMSNormQuantizeFP8`, wraps `pk.fused_rmsnorm_quantize_fp8_layer` →
  `fused_rmsnorm_quantize_fp8_sm100`). Exported from `norm/__init__.py` +
  `layers/__init__.py`. **This is the only genuinely-missing catalog layer**;
  everything else (LinearFP8*, FP8GroupGEMM*, MoEPermute/Unpermute, MLA*,
  QuantizeFP8*, TransposeScale) already existed.
- `DeepseekV3MLP` (`modeling.py`): concat-fused `gate_up` FP8 weight →
  `fp8_gemm_dense_smallm` → `silu_mul(grid.x=1, plain [gate|up] concat — CONFIRMED)`
  → quantize → `fp8_gemm_dense_smallm`(down) → `elementwise_add`. `forward()`
  dequantizes the stored FP8 weights (128×128 block).

Note: Python-only changes need **no C++ rebuild**. Only rebuild
(`pip install -e . -v --no-deps`) if you touch `.cuh`/`src/`.

---

## 5. WRITTEN, PENDING verification

- **`DeepseekV3MoE`** module (`modeling.py` ~line 685; `DeepseekV3MoEMLP =
  DeepseekV3MoE` alias kept so the not-yet-rewritten decoder layer imports).
  Wraps the **verified** permute chain + bf16 router GEMM +
  `MoETopkRouting(variant="sigmoid")` + shared expert (`DeepseekV3MLP`).
  `forward()` = HF gate + per-expert routed loop + `shared(x)` + residual
  (residual folded into shared-expert output via the unpermute `residual` arg ⇒
  **the decoder layer must NOT re-add the residual**).
  Test: `test_dsv3_moe_testmode.py` (ready). **Run this first in the new session.**
  Risk: bf16 router GEMM vs fp32 `forward` logits could flip top-k selection;
  mitigated with separated gate weights. If it fails on selection, align the
  test's `forward` to use bf16 logits or widen logit separation.

---

## 6. REMAINING work (in order)

1. **`DeepseekV3Attention` (MLA, un-absorbed decode)** — `modeling.py` currently
   still has the OLD draft `DeepseekV3MLA` (BF16, stubbed `forward`). Rewrite to:
   - `forward()` = faithful HF `DeepseekV3Attention.forward`
     (`modular_deepseek_v3.py:257-323`): q_a→q_a_layernorm→q_b, split nope/rope;
     kv_a→kv_a_layernorm→kv_b reconstruct full K/V; RoPE; SDPA with YARN
     `scaling`; o_proj. **Un-absorbed** (reconstructs K/V) — matches the BMM
     decode compile path, so NO KV-absorption is needed in the driver.
   - `compile()` decode chain (verified pk primitives, world_size=1):
     `fp8_gemm_dense`(qkv_a fused) → `FusedRMSNormQuantizeFP8`/RMSNorm(q_a) →
     `fp8_gemm_dense`(q_b) → `deepseek_mla_rope_q_fused`/`rope_k` →
     `pk.mla_kv_gather_layer(c_latent_new, k_pe_new, paged_cache, contiguous_kv,
     mla_params=(qk_head_dim, v_head_dim, page_size))` →
     `pk.mla_mtp_decode_layer(q_nope_pe, decode_kv, decode_out, partial_lse,
     decode_q_len, kv_len_max)` → `[mla_mtp_decode_reduce_layer if split>1]` →
     o_proj. Catalog `MLADecode`/`MLAKVGather`/`MLARopeQ`/`MLARopeK` leaf tests
     RUN (`tests/runtime_python/layers/test_mla_*`). Builder reference:
     `builder.py:_build_mla_attention_layer` (~2401-3020), decode else-branches.
   - Test needs KV-cache + meta-tensor scaffolding — crib from
     `tests/runtime_python/layers/test_mla_decode.py` / `test_mla_kv_gather.py`
     and `demo/qwen3/demo_new.py`.
2. **`DeepseekV3DecoderLayer`** (dense layers `< first_k_dense_replace`, else
   MoE) + multi-layer test (catches per-layer-buffer / MPK "case-3 fork+join"
   issues — allocate per-layer-unique intermediates).
3. **`DeepseekV3Model` + `DeepseekV3ForCausalLM`** — mirror qwen3 (`Embed` →
   layers → `RMSNorm`; lm_head + `ArgmaxPartial`/`ArgmaxReduce`; vocab pad to
   256-multiple + alignment-safe argmax task count, see
   `qwen3/modeling.py:_aligned_lm_head_tasks` + `process_weights`).
   YARN RoPE: real DSv3 uses YARN-scaled cos/sin + mscale; plain
   `RotaryEmbedding` will fail e2e token match. Port `_precompute_rope_embeddings`
   (builder.py ~1812) or add a YARN path.
4. **Driver `demo/deepseek_v3/demo_new.py`** — currently a BF16 draft. Rewrite
   FP8 weight load by reusing the legacy `demo/deepseek_v3/demo.py` conversion
   block (~lines 657-999) + `demo/deepseek_v3/models/convert.py`
   (`dequantize_fp8`, `is_fp8`, `get_model_params`). Do UE8M0 weight-scale
   packing in each module's `process_weights()` (override `_base.py`). Naive KV:
   combined `ckv_kpe_cache (num_layers, max_num_pages, page_size,
   kv_lora_rank+qk_rope_head_dim)` bf16. Keep `--layers`, `--model-path`,
   `--skip-weight-load`, `--save-tokens`. With un-absorbed BMM decode, **drop**
   the draft's `absorb_kv_into_q().to(bf16)` block.
5. **Stage 3 smoke**: `demo_new.py --layers 0-3 --skip-weight-load` compiles &
   runs; then with real weights. **Stage 4 e2e**: token-diff `demo_new.py` vs
   legacy `demo.py --use-mirage` on the same prompt/layers.

---

## 7. How to build / test / run (local rules)

- **conda env `mirage`** for everything. Python-only edits: NO rebuild. After
  `.cuh`/`src/` changes: `pip install -e . -v --no-deps`.
- **GPU rule (IMPORTANT): never share a GPU** — MPK persistent kernels busy-wait
  and two on one GPU hang. Find a FREE GPU first
  (`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits`;
  free = used<~8GB AND util<5%). A poll-watcher background loop works well on a
  shared cluster.
- Test-mode pattern: `params = PersistentKernel.get_default_init_parameters();
  params["test_mode"]=True; ...; pk = PersistentKernel(**params)`; attach inputs;
  build the module **inside `with pk.compile_scope():`** (composite `__init__`
  reads `current_pk().parallel_config`); `pk.compile(output_dir=...)`; `pk()`;
  compare vs `module.forward()`. See `/test-mode` skill +
  `tests/runtime_python/test_mode/test_dsv3_*`.
- Legacy correctness oracle still runs:
  `python demo/deepseek_v3/demo.py --use-mirage --model-path <DSv3> --layers 0-8 ...`.

---

## 8. Key files

- `python/mirage/mpk/models/deepseek_v3/modeling.py` — **the work** (new-API model).
- `python/mirage/mpk/layers/norm/rmsnorm_quantize_fp8.py` — new catalog layer (added).
- `python/mirage/mpk/layers/{linear/fp8_group_gemm.py, linear/linear_fp8.py, moe/permute.py, quantize_fp8.py, mla/}` — catalog leaves (read their tensor-contract docstrings; trust the kernel source over docstrings on scale layout).
- `tests/runtime_python/test_mode/test_dsv3_*` — the bottom-up composite tests.
- `tests/runtime_python/blackwell/common/sm100_fp8_scale_layout.py` — canonical UE8M0 quantize/pack/dequant helpers (used by running blackwell tests).
- `python/mirage/mpk/models/deepseek_v3/builder.py` — legacy reference for per-op recipes (FP8 dispatch, MLA decode chain, MoE `_new_moe_dispatch_inline`).
- `demo/deepseek_v3/{demo.py, demo_new.py, models/convert.py}` — legacy driver + FP8 conversion to reuse.
- `python/mirage/mpk/models/qwen3/{modeling.py}` + `demo/qwen3/demo_new.py` — new-API template.

---

## 9. Methodology reminders

- Bottom-up: don't build the next level until the current level's test passes.
- The FP8 scale layout is the recurring trap — when a GEMM gives ~50% error with
  sign flips (not NaN), suspect a scale K-outer/M-outer mismatch first.
- Trust kernel `.cuh` source over catalog docstrings for scale layouts.
- Match the legacy `builder.py` recipe for any `pk.*` FP8 primitive (it's the
  end-to-end-verified path); the catalog FP8 GEMM *wrappers* (`LinearFP8`,
  `FP8GroupGEMM` standalone tests) are partly compile-only/unverified.
