# DeepSeek V3 on the new `mirage.mpk.layers` catalog — FP8, decode-only (HANDOFF)

> Self-contained status + roadmap for continuing this work in a fresh session.
> Read top-to-bottom; assumes no prior context.

Branch: **`ref/new-demo`**. Run host: **gpu0** (1× B200 / SM100; home `/home/zepeng`,
edit-on-CPU / sync-and-run via `ssh gpu0 'bash ~/sync-mirage.sh && source ~/env.sh
&& cd ~/mirage && source .venv/bin/activate && …'`). 8-GPU nodes are gpu1/gpu2.

---

## 0. STATUS (2026-06-06): subset DONE + precision-aligned

The new-API DeepSeek V3 (`python/mirage/mpk/models/deepseek_v3/modeling.py`) is
**numerically verified bottom-up in test-mode at production `hidden=7168`**, **runs
end-to-end on the real `/mnt/shared/models/DeepSeek-V3` checkpoint** (layer subset,
batch 1/4/16, 1× B200), and is **precision-aligned with the HF reference** at bf16
tolerance. ~14 commits on `ref/new-demo`. The only thing left for *full* token-level
e2e is the 61-layer multi-GPU (~8× B200) run — 671B FP8 won't fit on one GPU
(a 4-layer subset is mathematically aligned but produces gibberish, so free-running
tokens don't match — confirmed an argmax-cascade artifact, not a precision bug).

---

## 1. Goal

Replace the legacy DSv3 builder (`builder.py`, a monolith of direct `pk.*_layer()`
calls) with a clean, HF-aligned, testable model on the **object-oriented catalog API**
(`python/mirage/mpk/layers/`), every block an `MPKModule` with `forward()` (pure-PyTorch
oracle) + `compile()` (MPK task registration). Template: `models/qwen3/modeling.py`.
PyTorch reference to mirror: `.venv/lib/python3.12/site-packages/transformers/models/deepseek_v3/modular_deepseek_v3.py`.

---

## 2. Locked design decisions (v1)

1. **FP8 weights** (DSv3 ships FP8). 2. **Single-GPU tested, TP/EP-aware via
`current_pk().parallel_config`** (AllReduce/sharding are no-ops at world_size==1).
3. **MoE = permute-based path**. 4. **Decode-only, no MTP, no prefill, naive
pre-allocated KV** (combined `ckv_kpe_cache (num_layers, max_num_pages, page_size,
kv_lora_rank+qk_rope_head_dim)` bf16). 5. **MLA is ABSORBED** (W_UV fused into o_proj,
`mla_mtp_decode_sm100`+`mla_mtp_reduce_sm100`; the driver applies KV-absorption +
W_UV→o_proj fusion at weight load). *(Supersedes earlier "un-absorbed BMM" plans —
do NOT drop absorption.)* 6. **Oracle = HF**; validate bottom-up.

---

## 3. CRITICAL: FP8 scale-layout map (the recurring trap — root-caused from `.cuh`)

The UE8M0 scale layout is path-specific; catalog docstrings are partly WRONG. Ground truth:
- **`quantize_fp8_sm100`** (`QuantizeFP8UE8M0`) writes UE8M0 scale **COLUMN-major /
  K-outer** `[K_PACKED, MBT_ALIGNED]`, `MBT_ALIGNED = round_up(batch,4)`
  (`per_token_group_quantize_fp8.cuh`; `task_register.cc` `aligned_batch`). This is the
  canonical layout the dense FP8 GEMM AND group GEMM read. **`moe_permute_sm100` had read
  it ROW-major** → MoE garbage at hidden≥1024 (coincide only at K_PACKED==1); fixed
  (commit `f3ebc113`).
- **`linear_fp8_swapAB_sm100`** reads `input_scale` ROW-major / M-outer → `QuantizeFP8UE8M0
  → swapAB` is BROKEN for K_PACKED>1. Don't pair them.
- **f32-scale path** (`scale_ue8m0=False`) writes M-outer row-major `(M, K/128)`.
- **HAZARD — fp32 scale params silently bf16-downcast**: the dense GEMM reads `weight_scale`
  as `float*`. `DeepseekV3MLP.gate_up_scale/down_scale` were fp32 `nn.Parameter`s; the
  driver's `model.to(dtype=bf16)` downcast them, so each 4-byte read straddled two bf16
  entries → garbage with real (high-variation) DeepSeek scales, invisible with uniform/
  random. Fixed via `DeepseekV3MLP._apply` fp32-restore (commit `b8fb9427`). Same class as
  the MLA-layernorm dtype guard. `num_sf_k = ceil((K/128)/4)` → `_packed_scale_k(K)`.

### Two verified FP8 GEMM recipes
- **Dense GEMM** (MLP, shared expert): `QuantizeFP8F32Scale` (f32 1×128 M-outer) →
  `fp8_gemm_dense_smallm` (weight scale f32 128×128 block); residual via `elementwise_add`.
- **Grouped GEMM** (routed MoE): `FP8GroupGEMMSmallM` (K-outer uint32 scale) on the
  permuted+quantized tokens.

---

## 4. DONE + VERIFIED (test-mode @ hidden=7168, B200)

| Level | rel / metric |
|---|---|
| MLA module (`test_dsv3_mla_testmode.py`) | rel 0.0075 |
| dense MLP (`test_dsv3_mlp_fp8_testmode.py`) | rel 0.039 (now uses ~199× scale variation + `.to(bf16)`) |
| MoE block (`test_dsv3_moe_testmode.py`) | rel 0.045 @7168 |
| decoder-layer dense / MoE (`test_dsv3_decoder_layer_testmode.py`) | 0.0385 / 0.0534 |
| 4-layer model dense / MoE (`test_dsv3_model_testmode.py`) | 0.086 / 0.091 |
| standalone MLA decode kernel wrapper (`blackwell/sm100_mla_mtp_decode/`) | cosine 1.0 |
| standalone dense GEMM wrapper (`blackwell/sm100_dense_gemm_smallm/`) | cosine 1.0 |

**Precision vs HF** (`modular_deepseek_v3.py`, real weights, 4-layer subset): layer-by-layer
(`compare_hf_layerwise.py`) embed/MLA exact, dense MLP 0.9995, 3-layer 0.9992; teacher-forced
output logits (`compare_hf_teacher_forced.py`) mean cosine 0.996 / median 1.0, top-1 86%.
**End-to-end on real checkpoint** (`demo_new.py`): runs batch 1/4/16. Qwen3-8B regression green
(~4.4 ms/tok) throughout.

### Bugs fixed (commits)
`a8ddd4f4` gather→decode dependency edge (graph.cc (4,0)→(3,1) + output_ptrs[0]) ·
`f3ebc113` moe_permute column-major scale · `f353ecb3` FP8 dense GEMM mbarrier desync
across output tiles (global k-step counter) · `3d9cf8d7` MLA decode `MAX_SK` 32→8 (B200 smem
optin budget; deadlock) · `ce3aec75` topk_sigmoid warp-mask · `9691bbfc` new MLAMtpDecode/
Reduce leaves + disable buggy MLADecode / reroute MLAReduce · `2ac09d5e` modeling: single-split
decode skips reduce, lm_head/argmax 16B align, gather page_size, bf16 dtypes · `7baa733c`
demo_new FP8 weight-load layouts + build-in-compile-scope · `955c9cd2` YARN RoPE + interleaved
convention (catalog RotaryEmbedding opt-in `rope_scaling`/`interleaved`, matches HF rel 5e-8) ·
`b8fb9427` FP8-MLP fp32-scale dtype · tests `7be001b4`/`768ba324`/`95427d80`/`f4cc1ddc`.

**Key technique:** when an MPK kernel fails in the runtime but might be correct, build a
standalone `__global__` + pybind11 wrapper (`blackwell/sm100_*` dirs) to test it outside the
megakernel — this proved the MLA decode and dense GEMM kernels correct and redirected each hunt
to the real (system-level) bug.

---

## 5. REMAINING

1. **Full 61-layer token-level e2e** — needs ~8× B200 (671B FP8 > one 180 GB GPU). Wire
   TP/EP multi-GPU (`world_size>1`, vocab-parallel lm_head, NVSHMEM all-reduce — the
   `ColumnParallelLinear`/`RowParallelLinear` + `ep_size` catalog pieces exist) on gpu1/gpu2,
   then token-match vs HF. **The deferred end-to-end goal.**
2. **Prefill attention path** — the model is decode-only; a real MLA prefill is needed for
   faithful multi-token prompt processing. Doable on gpu0.
3. **decode/reduce split-count mismatch** — decode launches with runtime `sk_rt`, reduce with
   compile-time `num_splits` (`task_register.cc` ~3688/3752), strides `*sk`. Only bites
   sequences >128 tokens (multi-split); the single-split subset never hits it (the standalone
   wrapper passes at num_splits=2 because it uses one `sk`). Needs a multi-split MPK repro to
   confirm + fix (pass `sk_rt` to the reduce, or drop the decode clamp + init LSE to −inf for
   inactive splits).
4. **Generalize test-mode scaffolding** — replace the `#ifdef MPK_TEST_MODE` single-iteration
   MLA-decode special case in `persistent_kernel.cuh` with a general MODE_OFFLINE "run task
   graph once" mechanism.

---

## 6. Build / test / run

- **Run host gpu0** (1× B200). Pattern:
  `ssh gpu0 'bash ~/sync-mirage.sh && source ~/env.sh && cd ~/mirage && source .venv/bin/activate && cd <dir> && CUDA_VISIBLE_DEVICES=0 timeout <s> python -u <test> 2>&1 | tail -N'`.
  Never share a busy GPU (persistent kernels busy-wait). Bracket-kill hung procs: `pkill -9 -f "[t]est_…"`.
- Python / task-`.cuh` edits need **no native rebuild** (megakernel nvcc-compiled fresh each
  run). After `src/`/cython/cmake/rust edits: `build-if-needed.sh` / `pip install -e . --no-deps`.
- Megakernel compile <120s; longer = a bug.
- Test-mode pattern: `params=PersistentKernel.get_default_init_parameters(); params["test_mode"]=True`;
  attach inputs; build the module **inside `with pk.compile_scope():`**; `pk.compile(output_dir=…)`;
  `pk()`; compare vs `forward()`. MLA test-mode needs the single-iteration scaffolding
  (`MPK_TEST_MODE` + `max_seq_length>1 && max_num_batched_tokens==1`).
- Regression oracle (must stay green): Qwen3-8B `demo/qwen3/demo.py --use-mirage --model
  /mnt/shared/models/Qwen3-8B --max-num-batched-requests 1` (coherent + ~4.3 ms/tok).

---

## 7. Key files

- `python/mirage/mpk/models/deepseek_v3/modeling.py` — the new-API model.
- `python/mirage/mpk/layers/mla/mtp_decode.py` (MLAMtpDecode/Reduce), `mla/decode.py`,
  `mla/{rope.py,kv_gather.py}`, `rotary.py` (YARN+interleaved), `linear/fp8_gemm_dense.py`,
  `linear/fp8_group_gemm.py`, `moe/permute.py` — catalog leaves (trust kernel `.cuh` over
  docstrings on scale layout).
- `tests/runtime_python/test_mode/test_dsv3_*` — bottom-up composite tests.
- `tests/runtime_python/blackwell/sm100_mla_mtp_decode/`, `sm100_dense_gemm_smallm/` — standalone
  kernel-wrapper isolation harnesses.
- `demo/deepseek_v3/{demo_new.py, compare_hf_layerwise.py, compare_hf_teacher_forced.py,
  compare_hf_precision.py, models/convert.py}` — driver + HF precision harnesses.
- `python/mirage/mpk/models/deepseek_v3/builder.py` — legacy reference for per-op recipes
  (FP8 dispatch, MLA decode chain, `_precompute_rope_embeddings` YARN).

---

## 8. Methodology reminders

- Bottom-up: don't build the next level until the current passes.
- FP8 scale layout is the recurring trap — ~50% error with sign flips (not NaN) ⇒ suspect a
  scale K-outer/M-outer mismatch or an fp32-param bf16-downcast first.
- fp32 `nn.Parameter`s read by a kernel as `float*` will be silently bf16-downcast by
  `model.to(bf16)` — guard them in `_apply` (declare/restore fp32).
- A kernel that fails in the megakernel but might be correct ⇒ test it standalone with a
  `__global__`+pybind11 wrapper to split "kernel bug" from "invocation/runtime bug".
- Free-running token-diff on a truncated subset is gibberish-unstable; use teacher-forced
  per-position logit cosine / top-k as the precision metric.
- Match the legacy `builder.py` recipe for any proven FP8/MLA/RoPE primitive.
