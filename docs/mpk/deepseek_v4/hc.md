# MPK port spec — DeepSeek V4-Flash Hyper-Connections (mHC) kernels

**Wave**: 1 (spec only — no code changes in this wave)
**Scope**: 4 MPK tasks, mirroring the vLLM-canonical 4-kernel mHC layout
**Status**: spec
**Owners**: 4 Wave-2 subagents (one per task, each in its own worktree)

Numerical oracle is the official PyTorch reference
`deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`. Design template
(kernel boundaries, fusion choices, math expressions) is vLLM's TileLang
implementation `deps/vllm/vllm/model_executor/layers/mhc.py`. The
prenorm-GEMM has an additional CUDA reference in
`deps/vllm/vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh`.

This spec strictly follows the "Mandatory kernel-authoring requirements"
in `~/.claude/plans/i-want-to-add-dapper-pascal.md` (§"Mandatory
kernel-authoring requirements", lines 75–132). Each kernel section
contains: math formula, I/O tables, source-to-migrate citation, generated
CUDA reference path, grid_dim subsection, `/add-mpk-task` conformance
checklist, naive v1 implementation strategy, test-mode plan, and reuse
notes.

---

## 0. Conventions, symbols, and global constants

### 0.1 Symbols used throughout

| Symbol | Value (for Flash-Base) | Meaning |
|---|---|---|
| `N` | dynamic (= `num_tokens` in vLLM, `b*s` in `model.py`) | flat token count |
| `H` | 4096 (`hidden_size`) | per-copy hidden dim (`args.dim`) |
| `hc` | 4 (`hc_mult`) | number of HC copies of hidden state |
| `hc_dim` | 16384 (= `hc * H`) | flattened HC hidden dim |
| `hc3` | 24 (= `(2 + hc) * hc`) | width of `fn` rows / `base` (mix_hc in model.py) |
| `splits` | dynamic; `compute_num_split(64, hc_dim, ceil(N/64))` | split-K factor for prenorm GEMM; v1 fallback uses `splits=1` |
| `hc_sinkhorn_iters` | 20 | Sinkhorn iterations (config.json) |
| `hc_eps` | 1e-6 | numerical eps for Sinkhorn + sigmoid offset (config.json) |
| `rms_eps` | 1e-6 (`norm_eps`) | RMSNorm epsilon |
| `hc_post_mult_value` | 2.0 | multiplier in `post = 2*sigmoid(...)` (matches `model.py:694` `2 * T.sigmoid(...)`) |

### 0.2 Dtypes (critical)

Per plan corrections (binding):

- **`residual` (HC stream): `bfloat16`** — propagated through layers.
- **HC weights `hc_*_fn`, `hc_*_base`, `hc_*_scale`: `float32`** — stored
  fp32 even in fp8 checkpoint (`model.py:666-672` uses
  `with set_dtype(torch.float32)`).
- **Intermediate `gemm_out_mul`, `gemm_out_sqrsum`, `mixes`, `pre`,
  `post`, `comb`: `float32`** (matches
  `deps/vllm/vllm/model_executor/layers/mhc.py:71-78`).
- **`layer_input` (output of `mhc_pre`, input to attn-norm /
  ffn-norm): `bfloat16`** (matches `mhc.py:79, 254-259`).
- **`mhc_post` output: same as `residual`, i.e. `bfloat16`** (matches
  `mhc.py:417` `torch.empty_like(residual)`).

### 0.3 PyTorch reference orchestration (the "what we are implementing")

From `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:674-701`
(reproduced verbatim for context — this is the spec's numerical oracle):

```python
def hc_pre(self, x, hc_fn, hc_scale, hc_base):
    # x: [b,s,hc,d], hc_fn: [mix_hc,hc*d], hc_scale: [3], hc_base: [mix_hc]
    shape, dtype = x.size(), x.dtype
    x = x.flatten(2).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
    mixes = F.linear(x, hc_fn) * rsqrt
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base,
                                        self.hc_mult, self.hc_sinkhorn_iters, self.hc_eps)
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=2)
    return y.to(dtype), post, comb

def hc_post(self, x, residual, post, comb):
    # x: [b,s,d], residual: [b,s,hc,d], post: [b,s,hc], comb: [b,s,hc,hc]
    y = post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=2)
    return y.type_as(x)
```

And the Block uses them as (`model.py:689-701`):

```python
def forward(self, x, start_pos, input_ids):
    residual = x                                 # [b,s,hc,H]
    x, post, comb = self.hc_pre(x, self.hc_attn_fn, ...)
    x = self.attn_norm(x)
    x = self.attn(x, start_pos)
    x = self.hc_post(x, residual, post, comb)
    residual = x
    x, post, comb = self.hc_pre(x, self.hc_ffn_fn, ...)
    x = self.ffn_norm(x)
    x = self.ffn(x, input_ids)
    x = self.hc_post(x, residual, post, comb)
    return x
```

This produces two `mhc_pre/mhc_post` pairs per layer. The transformer
forward (`model.py:802-810`) finishes with `mhc_head` collapsing
`[N, hc, H] → [N, H]` for the lm_head GEMM.

The `hc_split_sinkhorn` semantics (the math driving `pre`/`post`/`comb`)
is in `deps/deepseek_v4/DeepSeek-V4-Flash/inference/kernel.py:371-427`.

### 0.4 vLLM kernel boundary choice (why 4 tasks)

vLLM fuses mHC into exactly four kernels (`mhc.py:48,181,359,460`):

1. `tf32_hc_prenorm_gemm` (CUDA, `sm100_tf32_hc_prenorm_gemm.cuh`):
   produces `gemm_out_mul[splits, N, hc3] fp32` and
   `gemm_out_sqrsum[splits, N] fp32` from `residual` and `fn`.
2. `mhc_pre_big_fuse_tilelang` (TileLang, `mhc.py:48`): consumes the two
   gemm outputs + `hc_scale, hc_base, residual`, produces `post_mix[N,
   hc] f32`, `comb_mix[N, hc, hc] f32`, `layer_input[N, H] bf16`. Folds
   reduce-across-splits, RMSNorm rsqrt, sigmoid (pre/post), Sinkhorn,
   AND the `y = sum(pre * x, dim=hc)` mix into one persistent grid of
   `(num_tokens,)` CTAs.
3. `mhc_post_tilelang` (TileLang, `mhc.py:359`): the residual-update
   step. `out[n, hc, h] = post[n, hc] * x[n, h] + sum_k(comb[n, k, hc] *
   residual[n, k, h])`.
4. `hc_head_fuse_tilelang` (TileLang, `mhc.py:460`): two-pass per-token
   kernel for `ParallelHead.hc_head`. Pass 1 accumulates squared-sum +
   `hc_mult` dot-products. Pass 2 applies the sigmoid-gated weighted sum
   into `out[N, H] bf16`.

MPK adopts the **same 4 task boundaries**. Names are:

| MPK task name | Implementation file | vLLM oracle |
|---|---|---|
| `mhc_prenorm_gemm` | `include/mirage/persistent_kernel/tasks/blackwell/mhc_prenorm_gemm_sm100.cuh` | `sm100_tf32_hc_prenorm_gemm.cuh` (CUDA) |
| `mhc_pre` | `include/mirage/persistent_kernel/tasks/blackwell/mhc_pre_sm100.cuh` | `mhc_pre_big_fuse_tilelang` (`mhc.py:48-178`) |
| `mhc_post` | `include/mirage/persistent_kernel/tasks/blackwell/mhc_post_sm100.cuh` | `mhc_post_tilelang` (`mhc.py:359-408`) |
| `mhc_head` | `include/mirage/persistent_kernel/tasks/blackwell/mhc_head_sm100.cuh` | `hc_head_fuse_tilelang` (`mhc.py:460-551`) |

Python layer methods (added additively to
`python/mirage/mpk/persistent_kernel.py`):

- `mhc_prenorm_gemm_layer(self, residual, fn, gemm_out_mul, gemm_out_sqrsum, grid_dim, block_dim)`
- `mhc_pre_layer(self, gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual, post_mix, comb_mix, layer_input, grid_dim, block_dim)`
- `mhc_post_layer(self, x, residual, post_mix, comb_mix, output, grid_dim, block_dim)`
- `mhc_head_layer(self, residual, fn, hc_scale, hc_base, output, grid_dim, block_dim)`

### 0.5 The cardinal MPK rule: `blockIdx`-agnostic

`CLAUDE.md` / "Key Concepts" / "Task" entry — and the plan §"Mandatory
kernel-authoring requirements" §4 — make this binding. The vLLM TileLang
code uses `with T.Kernel(num_tokens, threads=...) as i:` which lowers to
`blockIdx.x` indexing the token. **MPK ports MUST NOT use `blockIdx.*`
for routing.** Instead each CTA reads its token slice from
`task_desc->task_metadata` (we will add a `token_offset` /
`num_tokens_this_task` packing — see §"Wiring" below) and uses base
pointers from `task_desc->input_ptrs[]` / `output_ptrs[]`. The MPK
scheduler dispatches each token (or chunk-of-tokens) as a separate task
to whatever worker thread block is free.

This is a behavior change versus the vLLM TileLang kernel — every port
in this spec calls it out explicitly in its "Grid design" subsection.

### 0.6 Reuse-from-V3 inventory (do not reimplement)

These already-shipped MPK pieces are used directly by the new layer
methods (verified against `python/mirage/mpk/persistent_kernel.py`):

- `TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))` builder pattern
  (e.g. `rmsnorm_layer:1026`, `linear_fp8_layer:2001`).
- `kn_graph.customized([...], tb_graph)` + `kn_graph.register_task(...)`
  invocation (e.g. `rmsnorm_layer:1030-1031`).
- `tb_graph.new_input(tensor, (dim_map), tile_idx, is_input)` partition
  syntax. v1 ports use `(-1, -1, -1), -1, True` for "global, not
  grid-partitioned" since each CTA picks its slice from
  `task_metadata` rather than from grid partitioning.
- `attach_input(...)` and the `test_mode` harness
  (`get_default_init_parameters()`, `params["test_mode"]=True`,
  `pk.compile`, `pk.run_test_mode`, `pk.finalize`) — pattern verified in
  `tests/runtime_python/test_mode/test_rmsnorm_testmode.py`.
- `rmsnorm_layer` and `linear_layer` — these are the **decomposed v1
  fallback** for `mhc_prenorm_gemm` (see §1.8).
- `elementwise_add_layer` — used for `mhc_post` v1 fallback if we
  decompose (see §3.8).
- Norm helper `rms_norm_sm100` in
  `include/mirage/persistent_kernel/tasks/blackwell/norm_sm100.cuh:27` —
  semantics reference for the rsqrt + finalize.

---

## 1. Task 1 — `mhc_prenorm_gemm`

### 1.1 Math formula

This kernel is the "prenorm" half of `hc_pre`. Following vLLM (`mhc.py:275-283` and `sm100_tf32_hc_prenorm_gemm.cuh`), it fuses **the
projection by `fn` and the squared-sum needed by the subsequent RMSNorm
rsqrt** into a single TF32-accumulator GEMM with a side `sqr_sum`
reduction. It does **not** apply the rsqrt itself — that is folded into
`mhc_pre`. This is a deliberate vLLM choice: the rsqrt is the only step
that requires the full K-axis reduction to land before it can be
applied; by emitting partial sums per K-split and finishing the
reduce-and-divide in `mhc_pre`, the GEMM is straight TF32 with no
synchronization tail.

Let `R[n, hc, h] : bf16` be the residual stream (the per-token HC
state). Let `F[hc3, hc*H] : f32` be the prenorm weight (`hc_attn_fn`,
`hc_ffn_fn`, etc. — `model.py:667-668`). Let `R_flat[n, k] = R[n, k //
H, k % H]` be its `[N, hc*H]` view.

Per token `n`, the kernel computes:

```
gemm_out_mul[s, n, j]    = Σ_{k ∈ split_s} (R_flat[n, k]_f32) * F[j, k]
                                                  for j ∈ [0, hc3), s ∈ [0, splits)
gemm_out_sqrsum[s, n]    = Σ_{k ∈ split_s} (R_flat[n, k]_f32)^2
                                                  for s ∈ [0, splits)
```

with `split_s = [s * (hc_dim / splits), (s+1) * (hc_dim / splits))`. The
final accumulation across splits and the divide-by-`hc*H` happens in
`mhc_pre` (see §2.1). v1 may use `splits=1`, in which case the
"per-split" axis collapses (see §1.6 and §1.8).

### 1.2 I/O tables

**Inputs**

| Name | Shape | Dtype | Source |
|---|---|---|---|
| `residual` | `[N, hc, H]` (or flat `[N, hc*H]`) | `bfloat16` | persistent HC stream tensor (per-token), produced by previous `mhc_post` or the initial `embed_layer` + repeat (`model.py:804-806`) |
| `fn` | `[hc3, hc*H]` | `float32` | `Block.hc_attn_fn` / `Block.hc_ffn_fn` (`model.py:667-668`) |

**Outputs**

| Name | Shape | Dtype | Consumed by |
|---|---|---|---|
| `gemm_out_mul` | `[splits, N, hc3]` | `float32` | `mhc_pre` (§2) |
| `gemm_out_sqrsum` | `[splits, N]` | `float32` | `mhc_pre` (§2) |

**Weights**: only `fn`. `base` and `scale` are passed straight through
to `mhc_pre`, not consumed here.

For v1 with `splits=1`, the leading split dim is 1; tests should still
allocate shape `[1, N, hc3]` / `[1, N]` so the `mhc_pre` reduce-over-
splits loop (`mhc.py:89-95`) is exercised.

### 1.3 Source to migrate (cite + quote)

**Primary** (Blackwell):
`deps/vllm/vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh`,
function `sm100_tf32_hc_prenorm_gemm_impl` (lines 42–346). Key
compute — the squared-sum side of the kernel — lives at lines 268–340.
Quoted excerpt (lines 277–326):

```cpp
// Launch reductions
float2 sum[2] = {float2{0, 0}, float2{0, 0}};
#pragma unroll kNumStages
for (uint32_t s = 0; s < num_total_stages; ++ s) {
    // Wait TMA arrival
    const auto& stage_idx = s % kNumStages;
    full_barriers[stage_idx]->wait((s / kNumStages) & 1);

    // Load from shared memory into tensor memory using movement shape `.16x256b`
    constexpr uint32_t kNumBankGroupBytes = 16;
    constexpr uint32_t kNumElemsPerBankGroup = kNumBankGroupBytes / sizeof(nv_bfloat16);
    constexpr uint32_t kNumLoads = BLOCK_K / kNumElemsPerBankGroup;
    const auto& smem_base_ptr = reinterpret_cast<uint8_t*>(smem_a[stage_idx]) +
                                sub_warp_idx * BLOCK_M_PER_WARP * kSwizzleAMode;
    // ...
    // Cast, reduce and store into tensor memory
    float2 fp32x2_values[2][kNumLoads];
    #pragma unroll
    for (uint32_t i = 0; i < kNumLoads; ++ i) {
        #pragma unroll
        for (uint32_t u = 0; u < 2; ++ u) {
            fp32x2_values[u][i] = __bfloat1622float2(*reinterpret_cast<nv_bfloat162*>(&uint32_values[u][i]));
            sum[u] = __ffma2_rn(fp32x2_values[u][i], fp32x2_values[u][i], sum[u]);
        }
        // Store upper and lower part at the same time
        const auto idx_0 = i * 2, idx_1 = i * 2 + 1;
        cute::SM100_TMEM_STORE_16dp256b1x::copy(
            upper_view[idx_0], upper_view[idx_1],
            lower_view[idx_0], lower_view[idx_1],
            cast_stage_idx * BLOCK_K + i * 8);
    }
```

Key MMA-issue line (line 203):

```cpp
umma_t::fma(BLOCK_K * cast_stage_idx + k * UMMA_K, b_desc,
            BLOCK_K * kNumCastStages, s > 0 or k > 0, runtime_instr_desc);
```

**Secondary** (Hopper-style reference, for porters who prefer a
non-tcgen05 layout):
`deps/vllm/vllm/third_party/deep_gemm/include/deep_gemm/impls/sm90_tf32_hc_prenorm_gemm.cuh`
(294 lines).

**Tertiary** (TileLang launcher / Python contract):
`mhc.py:181-308` (function `mhc_pre`). Lines 261–283 show the exact
output-tensor allocations and the `tf32_hc_prenorm_gemm(...)` call
signature we must reproduce.

### 1.4 Generated CUDA reference

Path: `docs/mpk/deepseek_v4/_generated_cuda/mhc_prenorm_gemm.cu`. This
will be produced by the Wave-1 TileLang extraction harness (plan
§"Mandatory kernel-authoring requirements" §2). For the prenorm GEMM
specifically the canonical implementation is *already* CUDA, so the
generated file may simply be the deep-gemm impl copied with launch
parameters resolved for `(BLOCK_M=64, BLOCK_N=hc3=24, BLOCK_K=64,
kNumSplits=splits, SHAPE_N=hc3, SHAPE_K=hc*H=16384)`. The
implementer should consult the actual `.cuh` rather than the extracted
file. **Do not block on the file existing.**

### 1.5 Grid design (mandatory per plan §3)

**Natural grid dimensions** (vLLM): the deep-gemm impl uses a 1-D grid
of size `kNumSplits * ceil(num_tokens / BLOCK_M)` and partitions:

- `m_block_idx = blockIdx.x / kNumSplits` — the M-tile of tokens.
- `k_split_idx = blockIdx.x % kNumSplits` — the K-split assignment.

(see `sm100_tf32_hc_prenorm_gemm.cuh:129-134`).

**MPK partitioning policy**: each MPK task corresponds to a `(m_block,
k_split)` pair. The runtime emits `ceil(N/BLOCK_M) * splits` tasks per
GEMM. Each CTA reads its `(m_block, k_split)` from
`task_desc->task_metadata` (we extend the `TaskMetadata` union — see
§"Wiring", below). Inside the kernel, replace `__shfl_sync(0xffffffff,
blockIdx.x, 0)` (line 129) with the equivalent broadcast from the
task-metadata field. No other `blockIdx` usage exists in the deep-gemm
impl after that line (verified by `grep blockIdx` in lines 1-345 of the
.cuh: only line 129 and the assertion fallback line 343).

**Alignment constraints**:

- `BLOCK_M = 64` (line 222 enforces). `N` does **not** need to be a
  multiple of 64 — the kernel checks `m_idx < shape_m` (line 338) and
  the v1 fallback (decomposed) does not require alignment.
- `BLOCK_K = 64` and the `K`-dim (= `hc*H = 16384`) must be a multiple
  of `BLOCK_K * splits` for the split-K dispatch to evenly cover the
  reduction; the deep-gemm code handles a `kRemainKBlocks` remainder
  (line 128) so non-evenness is OK numerically.
- `kNumSplits` is bounded by `kNumStages` (line 175): `splits ≤ 32`.

**CTA count selection**: in the persistent runtime, the scheduler
budgets up to `MAX_NUM_WORKERS = 160` CTAs
(`runtime_header.h:99`). `ceil(4096/64) * 4 ≈ 256` tasks for a 4k-token
batch — many more tasks than workers, so the scheduler will pipeline.
For decode (`N ≤ 16`), `ceil(16/64) = 1` M-block × `splits = 1..16`
tasks; small.

### 1.6 `/add-mpk-task` conformance checklist

- [ ] `blockIdx`-agnostic: the deep-gemm `__shfl_sync(...,
      blockIdx.x, ...)` at line 129 is the only routing use of
      `blockIdx`. **Action**: replace with
      `task_desc->task_metadata.<m_block_field>` /
      `<k_split_field>`. Confirm no other `blockIdx` usage post-port
      (`grep -n blockIdx` on the ported `.cuh` must return only
      comments and the asserts).
- [ ] `TaskType` enum entry: add
      `TASK_MHC_PRENORM_GEMM_SM100 = <next unused id in 296..297>` in
      `include/mirage/persistent_kernel/runtime_header.h` (the
      placeholder `TASK_SM100_TASK_END = 298` constrains us; pick
      e.g. `296`).
- [ ] `register_<task>_task` codegen entry: add
      `register_mhc_prenorm_gemm_sm100_task` in
      `src/kernel/task_register.cc` modelled on
      `register_linear_fp8_sm100_task` (`task_register.cc:4083`).
- [ ] `graph.cc` dispatch entry: add `else if (name ==
      "mhc_prenorm_gemm_sm100") { ... }` near the FP8 cluster
      (`graph.cc:787`).
- [ ] Layer-method pattern: imitate `linear_fp8_layer`
      (`persistent_kernel.py:1990-2015`) — 2 inputs (`residual`,
      `fn`) + 2 outputs (`gemm_out_mul`, `gemm_out_sqrsum`); all
      partitioning `(-1,-1,-1), -1, True` because the CTA picks its
      tile from `task_metadata`.

### 1.7 Initial (naive) implementation strategy

**v1 fallback (recommended for first landing)**: do **not** port the
deep-gemm tcgen05 kernel. Instead **decompose** `mhc_prenorm_gemm` into
two existing MPK ops, leveraging Reuse-from-V3:

1. Allocate a fp32 view of `residual` flattened to `[N, hc*H]`.
2. Compute `sqrsum[1, N] = sum_k residual_f32[n, k]^2` — this is the
   first half of `rmsnorm` without the rsqrt and weight multiply.
   Implementable as a one-off custom task **or** by using
   `rmsnorm_layer` on `residual.flatten(2)` with `weight = ones` and a
   tee on the squared-sum (more work — prefer the custom task).
3. Compute `gemm_out_mul[1, N, hc3] = residual_f32 @ fn.T` — this is a
   straight bf16-input × fp32-weight → fp32-output matmul; **note this
   is *not* a standard `linear_layer` shape** because the output is
   fp32 and the input/weight dtypes are mixed. The cleanest v1 path is
   a single new `.cuh` that implements the squared-sum and the matmul
   together with FP32 accumulators in registers, no tcgen05, no
   warp-spec. Concretely:

   - Grid `(N,)` (one CTA per token; same as `mhc_pre`'s natural grid).
   - 256 threads, each strides over `k ∈ [0, hc*H)` with stride 256.
   - Per-thread accumulators: `float sqrsum_local; float
     mul_local[hc3];` (24 floats — fits easily in registers).
   - For each `k` slice, read `residual[n, k]_bf16`, convert to
     float, FMA `sqrsum_local += x*x` and `mul_local[j] += x *
     fn[j, k]` for `j ∈ [0, hc3)`.
   - Cross-thread reduce via warp shuffles + smem.
   - Store `gemm_out_sqrsum[0, n] = sqrsum_local` and
     `gemm_out_mul[0, n, :] = mul_local[:]`.
   - `splits=1` for v1. The reduce-across-splits loop in `mhc_pre`
     (`mhc.py:89-95`) still runs correctly with 1 iteration.

**v2** (post-correctness): port `sm100_tf32_hc_prenorm_gemm_impl` with
the blockIdx-substitution above.

### 1.8 Test-mode unit test plan

**File**: `tests/runtime_python/test_mode/test_mhc_prenorm_gemm_testmode.py`.

**PyTorch oracle snippet** (extracted from `model.py:677-679` —
the matmul and sqrsum half of `hc_pre`):

```python
def torch_mhc_prenorm_gemm_ref(residual_bf16, fn_fp32):
    # residual: [N, hc, H] bf16; fn: [hc3, hc*H] fp32
    x = residual_bf16.flatten(1).float()                       # [N, hc*H]
    gemm_out_mul = torch.nn.functional.linear(x, fn_fp32)      # [N, hc3]
    gemm_out_sqrsum = (x * x).sum(dim=-1)                      # [N]
    return gemm_out_mul, gemm_out_sqrsum
```

**Test dims**: `N=4`, `hc=4`, `H=128` (so `hc_dim=512`,
`hc3=24`). Small enough to debug; large enough to exercise the
`hc*H=512 / threads=128 = 4` per-thread strides used at full size.

**Tolerance**: `rtol=1e-3, atol=1e-3` for both outputs. Because the
GEMM is TF32, allow `atol=5e-3` if the v2 ported tcgen05 path is the
target. v1 (fp32-accumulate, scalar) should match tightly.

**Pattern**: clone `test_rmsnorm_testmode.py:23-87`.

### 1.9 Reuse from V3

- The v1 fallback uses no existing V3 task directly, but it follows the
  same TBGraph + register_task scaffolding as
  `rmsnorm_layer`/`linear_layer`. If v1 must be reduced further (e.g.
  the rsqrt path is broken), the **fully-decomposed** v0 is `rmsnorm_layer(residual.view(N,hc*H), weight=1)` followed by `linear_layer(residual_flat, fn)`, both already in `persistent_kernel.py`. This v0 would output bf16 `mixes` directly and skip the `gemm_out_sqrsum` channel entirely — see "Pre-v1 oracle build" in §5.

---

## 2. Task 2 — `mhc_pre`

### 2.1 Math formula

This kernel finishes `hc_pre`: applies rsqrt, sigmoid (pre & post),
Sinkhorn iteration on `comb`, and the weighted sum that produces
`layer_input`. It is "everything in `mhc_pre` other than the gemm + sqrsum"
(`mhc.py:66`).

For each token `n ∈ [0, N)`:

```
rms[n]   = rsqrt( (Σ_s gemm_out_sqrsum[s, n]) / (hc*H) + rms_eps )       # rmsnorm denom
mixes[n, j] = ( Σ_s gemm_out_mul[s, n, j] ) * rms[n]   for j ∈ [0, hc3)
```

Then split `mixes` into three parts (matches `kernel.py:391-396` and
`mhc.py:100-115`):

```
pre[n, j]  = sigmoid(mixes[n, j]                  * hc_scale[0] + hc_base[j])                  + hc_eps,       for j ∈ [0, hc)
post[n, j] = hc_post_mult_value * sigmoid(mixes[n, j+hc]  * hc_scale[1] + hc_base[j+hc]),                       for j ∈ [0, hc)
                                                                                                                # hc_post_mult_value = 2 (matches `mhc.py:109` and kernel.py:394)
cm[n, j, k] = mixes[n, j*hc + k + 2*hc] * hc_scale[2] + hc_base[j*hc + k + 2*hc],   for j,k ∈ [0, hc)
```

Then Sinkhorn-normalize `cm` (matches `mhc.py:117-145` and
`kernel.py:401-423`):

```
# Init: row-softmax + eps
cm[n] = softmax(cm[n], dim=-1) + hc_sinkhorn_eps           # row-softmax with eps offset
# Col-normalize
cm[n, :, k] /= (Σ_j cm[n, j, k] + hc_sinkhorn_eps)
# (sinkhorn_repeat - 1) more alternating row/col normalizations:
repeat (sinkhorn_repeat - 1) times:
    cm[n, j, :] /= (Σ_k cm[n, j, k] + hc_sinkhorn_eps)
    cm[n, :, k] /= (Σ_j cm[n, j, k] + hc_sinkhorn_eps)
comb_mix[n, j, k] = cm[n, j, k]
```

Finally, apply the `pre` weights to mix the HC copies down to 1
(matches `mhc.py:163-177` and `model.py:681`):

```
layer_input[n, h]_bf16 = Σ_{i_hc} pre[n, i_hc] * residual[n, i_hc, h]_f32, cast to bf16
                                  for h ∈ [0, H)
```

The constants: `hc_pre_eps = hc_sinkhorn_eps = hc_eps = 1e-6` (per plan
correction; `model.py:680` passes `self.hc_eps` to both call sites
inside `hc_split_sinkhorn`). `sinkhorn_repeat = hc_sinkhorn_iters =
20`. `hc_post_mult_value = 2.0`.

Note `pre` is **not written to global memory**; it is consumed by the
mix step within the same kernel. Only `post_mix`, `comb_mix`,
`layer_input` are outputs (verified `mhc.py:77-79, 105, 148-149,
163-177`).

### 2.2 I/O tables

**Inputs**

| Name | Shape | Dtype | Source |
|---|---|---|---|
| `gemm_out_mul` | `[splits, N, hc3]` | `float32` | output of `mhc_prenorm_gemm` (§1) |
| `gemm_out_sqrsum` | `[splits, N]` | `float32` | output of `mhc_prenorm_gemm` (§1) |
| `hc_scale` | `[3]` | `float32` | `Block.hc_attn_scale` / `hc_ffn_scale` (`model.py:671-672`) |
| `hc_base` | `[hc3]` | `float32` | `Block.hc_attn_base` / `hc_ffn_base` (`model.py:669-670`) |
| `residual` | `[N, hc, H]` | `bfloat16` | persistent HC stream tensor |

**Outputs**

| Name | Shape | Dtype | Consumer |
|---|---|---|---|
| `post_mix` | `[N, hc]` | `float32` | `mhc_post` (§3) |
| `comb_mix` | `[N, hc, hc]` | `float32` (TileLang stores flat as `[N, hc*hc]`; we keep the 3-D view in MPK metadata) | `mhc_post` (§3) |
| `layer_input` | `[N, H]` | `bfloat16` | attn-norm / ffn-norm (`model.py:692, 698`) |

### 2.3 Source to migrate (cite + quote)

**Primary**: `deps/vllm/vllm/model_executor/layers/mhc.py:48-178`
(`mhc_pre_big_fuse_tilelang`). Quoted excerpt of the key compute
(lines 81–149, ~30 lines edited slightly to fit budget — the porter
should read all 100 lines):

```python
with T.Kernel(num_tokens, threads=96) as i:
    T.pdl_sync()
    # _pre_norm_fn_fwd_norm
    rms = T.alloc_fragment(1, T.float32)
    mixes = T.alloc_fragment(hc_mult3, T.float32)
    T.clear(mixes)
    rms[0] = 0
    for i_split in T.serial(n_splits):
        rms[0] += gemm_out_sqrsum[i_split, i]
    rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
    for j in T.Parallel(hc_mult3):
        mixes[j] = 0
        for i_split in T.serial(n_splits):
            mixes[j] += gemm_out_mul[i_split, i, j]
        mixes[j] *= rms[0]
    mixes_shared = T.alloc_shared(hc_mult3, T.float32)
    T.copy(mixes, mixes_shared)

    if T.get_thread_binding() < 32:
        # _pre_split_mixes_fwd (post & comb) + _sinkhorn_fwd  [warp 0]
        for j in T.Parallel(hc_mult):
            post_mix[i, j] = (T.sigmoid(mixes_shared[j + hc_mult] * hc_scale[1]
                                         + hc_base[j + hc_mult]) * hc_post_mult_value)
        for j, k in T.Parallel(hc_mult, hc_mult):
            cm[j, k] = (mixes_shared[j*hc_mult + k + hc_mult*2] * hc_scale[2]
                        + hc_base[j*hc_mult + k + hc_mult*2])
        # row-softmax + eps
        T.reduce_max(cm, row_max, dim=1)
        for j, k in T.Parallel(hc_mult, hc_mult):
            cm[j, k] = T.exp(cm[j, k] - row_max[j])
        T.reduce_sum(cm, row_sum, dim=1)
        for j, k in T.Parallel(hc_mult, hc_mult):
            cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps
        # col-normalize
        T.reduce_sum(cm, col_sum, dim=0)
        for j, k in T.Parallel(hc_mult, hc_mult):
            cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)
        for _ in T.serial(sinkhorn_repeat - 1):
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)
            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)
        for j, k in T.Parallel(hc_mult, hc_mult):
            comb_mix[i, j * hc_mult + k] = cm[j, k]
```

The "warp ≥ 1" branch (lines 150-177) does the pre-mix sigmoid +
weighted-sum:

```python
    else:
        pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
        for j in T.Parallel(hc_mult):
            pre_mix_shared[j] = (T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j])
                                  + hc_pre_eps)
        for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
            xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
            xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
            T.copy(residual[i, 0, i0_h * hidden_block], xs)
            T.copy(xs, xl)
            ol = T.alloc_fragment(hidden_block, T.float32)
            T.clear(ol)
            for i_hc in T.serial(hc_mult):
                pre = pre_mix_shared[i_hc]
                for i1_h in T.Parallel(hidden_block):
                    ol[i1_h] += pre * xl[i_hc, i1_h]
            T.copy(ol, layer_input[i, i0_h * hidden_block])
    T.pdl_trigger()
```

**Secondary**: official PyTorch reference
`deps/deepseek_v4/DeepSeek-V4-Pro/inference/kernel.py:371-427` (the
`hc_split_sinkhorn_kernel`) is the per-token Sinkhorn logic in
isolation — useful because it removes the prenorm-gemm fusion and shows
the Sinkhorn math alone.

### 2.4 Generated CUDA reference

Path: `docs/mpk/deepseek_v4/_generated_cuda/mhc_pre.cu`. Produced by
JIT-dumping `mhc_pre_big_fuse_tilelang`. The implementer should diff
against this file when verifying their port. Do not block the spec on
the file existing.

### 2.5 Grid design (mandatory)

**Natural grid dimensions**: `(N,)` — one CTA per token (`mhc.py:81`
`with T.Kernel(num_tokens, threads=96) as i:`). Block dim is 96 threads
in vLLM; **MPK port uses 128 threads** (one warp-group) to align with
the existing Blackwell tasks' `WORKER_NUM_THREADS=256` convention while
keeping the 32-thread split warp-0-vs-warp≥1 logic intact. Concretely:
threads 0–31 do the post/comb/Sinkhorn branch; threads 32–127 do the
pre-mix + weighted-sum branch. (vLLM picks 96 to give 3 warps for the
"pre" branch; the MPK port can use 96 or 128 — see Open Questions.)

**How a CTA reads its slice from `task_metadata`**: the kernel needs
exactly one field — the token index `n`. We pack it as
`task_metadata.token_offset` (existing field — repurpose, since this
task does not use `expert_offset`). The CTA loads pointers from
`task_desc->input_ptrs[0..4]` (gemm_out_mul, gemm_out_sqrsum, hc_scale,
hc_base, residual) and `task_desc->output_ptrs[0..2]` (post_mix,
comb_mix, layer_input), then internally uses `n =
task_desc->task_metadata.token_offset` instead of `blockIdx.x`. There
is no other partitioning axis.

**MPK runtime partitioning policy**: the scheduler emits **N tasks per
`mhc_pre` invocation**, one per token, and dispatches them to free
workers. Decode (`N=1..16`) emits 1–16 tasks (small fraction of 160
workers). Prefill / batched-prefill (`N` up to ~4096) emits 4096 tasks
that pipeline through 160 workers (~26 waves). The per-CTA work is
small (one `hc_dim`-sized weighted sum + one 4×4 Sinkhorn), so wave
overhead is the dominant cost; no further partitioning is needed.

**Alignment constraints**:

- `hidden_block = gcd(512, H) = gcd(512, 4096) = 512` (`mhc.py:69`).
  For Flash-Base `H=4096` this gives 8 pipelined stages
  (`H/hidden_block = 8`). No alignment concerns at `H=4096`.
- For the unit test (`H=128`), `hidden_block = gcd(512,128) = 128`, so
  the pipelined loop runs once (single stage). v1 implementation must
  handle the `H < 512` case (just don't pipeline).
- `hc=4` is hard-coded by config. The kernel can be templated on `hc`
  but v1 may bake in `hc=4`.

### 2.6 `/add-mpk-task` conformance checklist

- [ ] `blockIdx`-agnostic: replace the implicit `T.Kernel(num_tokens,
      ...) as i:` → `i = blockIdx.x` lowering with `n =
      task_desc->task_metadata.token_offset`. Grep ported `.cuh` for
      `blockIdx`; must be 0 hits (no scheduler-routing use; the
      kernel may still use `blockIdx.x` purely for tcgen05 cluster
      semantics if any, but here there are none).
- [ ] `TaskType` enum entry: add `TASK_MHC_PRE_SM100 = 297` in
      `runtime_header.h`. (NB: `TASK_SM100_TASK_END = 298` limits us;
      pick the lowest unused free id, e.g. 261 is taken
      (`TASK_MOE_MUL_SUM_ADD_SM100`); confirm the actual gap when
      implementing.)
- [ ] `register_<task>_task` codegen entry: add
      `register_mhc_pre_sm100_task` in `task_register.cc`. Pattern
      after `register_rmsnorm_task` (`task_register.cc:91`) since the
      grid is 1-D per-token and there's no GEMM-tile codegen needed.
- [ ] `graph.cc` dispatch: add `else if (name == "mhc_pre_sm100") { ...
      }` near the other SM100 entries (after `mhc_prenorm_gemm_sm100`).
- [ ] Layer-method pattern: imitate `rmsnorm_layer`
      (`persistent_kernel.py:1015-1031`). 5 inputs + 3 outputs; all
      partitioning `(-1,-1,-1), -1, True` since token-routing is via
      `task_metadata`.

### 2.7 Initial (naive) implementation strategy

Plain CUDA, FP32 accumulators, no warp-spec, no TMA:

1. **Phase A — reduce across splits + rsqrt** (all 128 threads):
   - Thread `t` strides over `j ∈ [t, hc3)` step `blockDim.x` for the
     `mixes[j]` sum-over-splits; reduce-add into smem fragment
     `mixes_smem[hc3]`.
   - Thread 0 sums `gemm_out_sqrsum[s, n]` for `s ∈ [0, splits)` into
     `rms`. Or: all threads load one `s` each (since `splits ≤ 32`)
     and warp-reduce.
   - Thread 0 computes `rms = rsqrt(rms / (hc*H) + rms_eps)` and
     broadcasts via smem.
   - All threads multiply `mixes_smem[j] *= rms`.
   - `__syncthreads()`.
2. **Phase B — warp 0 path** (threads 0–31): post/comb + Sinkhorn.
   Since `hc=4`, `cm` is 4×4 = 16 entries. Allocate in registers per
   lane (4 lanes hold `cm[:, lane]` columns), use `__shfl` for
   cross-lane row/col sums. Run 20 Sinkhorn iterations entirely in
   registers. Store `post_mix[n, :]` (4 fp32 writes) and `comb_mix[n,
   :, :]` (16 fp32 writes).
3. **Phase C — warps 1–3 path** (threads 32–127): compute `pre_mix[hc]
   = sigmoid(mixes_smem[0..3] * hc_scale[0] + hc_base[0..3]) +
   hc_eps`. Store to smem `pre_smem[4]`.
   - Loop over hidden blocks of `hidden_block` elements (e.g. 512 for
     `H=4096`, single block for `H=128`).
   - Inside: load `residual[n, 0..hc-1, h_blk]` bf16, convert to
     float, do `out_h = sum_{i_hc=0..hc-1} pre_smem[i_hc] *
     residual_f32[i_hc, h_blk]`, cast back to bf16, store
     `layer_input[n, h_blk] = out_h`.

This is a faithful 1-to-1 port of the TileLang reference; no warp-spec
beyond the warp-0 / warp≥1 branch, no async copy, no TMA. Performance
is far below vLLM's TileLang version — fine for v1 (per plan §H.S.2
"Implementations may be naive at first").

### 2.8 Test-mode unit test plan

**File**: `tests/runtime_python/test_mode/test_mhc_pre_testmode.py`.

**PyTorch oracle snippet** (extracted from `model.py:674-682` + the
Sinkhorn from `kernel.py:386-425`, both reproduced in §0.3 and §2.1):

```python
def torch_mhc_pre_ref(residual_bf16, fn_fp32, hc_scale, hc_base,
                     rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20):
    N, hc, H = residual_bf16.shape
    hc3 = (2 + hc) * hc
    x = residual_bf16.flatten(1).float()                  # [N, hc*H]
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)  # [N, 1]
    mixes = torch.nn.functional.linear(x, fn_fp32) * rsqrt            # [N, hc3]
    pre  = torch.sigmoid(mixes[:, :hc]    * hc_scale[0] + hc_base[:hc])    + hc_eps
    post = 2 * torch.sigmoid(mixes[:, hc:2*hc] * hc_scale[1] + hc_base[hc:2*hc])
    cm   = (mixes[:, 2*hc:] * hc_scale[2] + hc_base[2*hc:]).view(N, hc, hc)
    # Sinkhorn
    cm = torch.softmax(cm, dim=-1) + hc_eps
    cm = cm / (cm.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        cm = cm / (cm.sum(dim=-1, keepdim=True) + hc_eps)
        cm = cm / (cm.sum(dim=-2, keepdim=True) + hc_eps)
    layer_input = (pre.unsqueeze(-1) * residual_bf16.float()).sum(dim=1).to(torch.bfloat16)
    return post, cm, layer_input
```

The test simulates the prenorm GEMM by invoking the oracle above
twice — once via PyTorch end-to-end (ground truth) and once via MPK
where the GEMM output is *prefilled* by a separate PyTorch matmul. The
test then exercises only the `mhc_pre` kernel under MPK.

**Test dims**: `N=4`, `hc=4`, `H=128`, `splits=1`. Allocate
`gemm_out_mul[1, 4, 24]`, `gemm_out_sqrsum[1, 4]` in PyTorch by hand
from the formulas, then run `mhc_pre_layer` via `pk.run_test_mode()`.

**Tolerance**: `torch.allclose(post_mix, ref_post, rtol=1e-3,
atol=1e-3)`, same for `comb_mix`. For `layer_input` (bf16), use `rtol=1e-2,
atol=1e-2` to absorb the bf16 cast.

### 2.9 Reuse from V3

- Build pattern: `rmsnorm_layer` (5-input variant doesn't exist;
  `linear_fp8_layer` shows how to pass 4 inputs + 1 output to
  `customized()`). The 5-in/3-out signature matches no existing V3
  layer exactly — this is a genuinely new layer method, but the
  scaffolding (TBGraph, partition tuples, register_task) is a direct
  copy.
- The pipelined-load loop in the kernel mirrors the `for win_idx`
  pattern in `rms_norm_sm100`
  (`norm_sm100.cuh:52-67`); v1 can use a simple `for h_blk_idx in
  range(H / hidden_block)` synchronous loop instead.

---

## 3. Task 3 — `mhc_post`

### 3.1 Math formula

This is the "residual update" half of HC. After the attention or FFN
sub-block produces `x[N, H] bf16`, `mhc_post` expands it back to
`[N, hc, H] bf16` using the pre-computed `post_mix` and `comb_mix`
(`model.py:684-687`):

```
out[n, hc_out, h] = post_mix[n, hc_out] * x[n, h]
                    + Σ_{hc_in ∈ [0, hc)} comb_mix[n, hc_in, hc_out] * residual[n, hc_in, h]
```

Note the inner sum is over `hc_in` (the first dim of `comb`), and
`hc_out` indexes the output HC copy. The vLLM TileLang code swaps the
argument convention (it calls `comb` "a" and treats it as `[N, hc,
hc]` directly — see `mhc.py:401-404`):

```python
for i_hco, i1_h in T.Parallel(hc, h_blk):
    x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
    for i_hci in T.serial(hc):
        x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
```

Here `a = comb_res_mix`, `b = residual`, `c = post_layer_mix`, `d = x`
(see `mhc.py:411-426`). So the contracted axis is the **first** `hc`
dim of `comb`, exactly matching the PyTorch reference
`model.py:686` (`torch.sum(comb.unsqueeze(-1) *
residual.unsqueeze(-2), dim=2)` where `dim=2` is the first `hc` of
`[b,s,hc,hc]`).

### 3.2 I/O tables

**Inputs**

| Name | Shape | Dtype | Source |
|---|---|---|---|
| `x` | `[N, H]` | `bfloat16` | output of attn-block or FFN-block (`model.py:693, 699`) |
| `residual` | `[N, hc, H]` | `bfloat16` | the HC stream snapshotted before `mhc_pre` (`model.py:690, 696`) |
| `post_mix` | `[N, hc]` | `float32` | output of `mhc_pre` (§2) |
| `comb_mix` | `[N, hc, hc]` | `float32` | output of `mhc_pre` (§2) |

**Outputs**

| Name | Shape | Dtype | Consumer |
|---|---|---|---|
| `output` | `[N, hc, H]` | `bfloat16` | becomes the new HC stream `residual` for the next layer |

### 3.3 Source to migrate (cite + quote)

**Primary**: `mhc.py:359-408`. Quoted compute (lines 380-407):

```python
with T.Kernel(n, threads=n_thr) as i_n:
    x_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
    b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
    d_shared = T.alloc_shared(h_blk, T.bfloat16)

    x_local = T.alloc_fragment((hc, h_blk), T.float32)
    b_local = T.alloc_fragment((hc, h_blk), T.float32)
    d_local = T.alloc_fragment(h_blk, T.float32)

    a_local = T.alloc_fragment((hc, hc), T.float32)
    c_local = T.alloc_fragment(hc, T.float32)
    T.pdl_sync()
    T.copy(a[i_n, 0, 0], a_local)                # comb_mix[n, :, :] -> regs
    T.copy(c[i_n, 0], c_local)                   # post_mix[n, :]    -> regs

    for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
        T.copy(b[i_n, 0, i0_h * h_blk], b_shared) # residual[n, :, h_blk]
        T.copy(d[i_n, i0_h * h_blk], d_shared)    # x[n, h_blk]

        T.copy(b_shared, b_local)
        T.copy(d_shared, d_local)
        for i_hco, i1_h in T.Parallel(hc, h_blk):
            x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
            for i_hci in T.serial(hc):
                x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
        T.copy(x_local, x_shared)
        T.copy(x_shared, x[i_n, 0, i0_h * h_blk])
    T.pdl_trigger()
```

`n_thr = 128`, `h_blk = gcd(hidden, 1024)` (= 1024 for `H=4096`).

### 3.4 Generated CUDA reference

Path: `docs/mpk/deepseek_v4/_generated_cuda/mhc_post.cu`.

### 3.5 Grid design (mandatory)

**Natural grid dimensions**: `(N,)` — one CTA per token (`mhc.py:380`
`with T.Kernel(n, threads=n_thr) as i_n:`). 128 threads/CTA.

**How a CTA reads its slice from `task_metadata`**: same pattern as
§2 — pack the token index into `task_metadata.token_offset`. Inside
the kernel, `n = task_desc->task_metadata.token_offset` replaces the
TileLang `i_n` (which would lower to `blockIdx.x`). Base pointers
from `task_desc->input_ptrs[0..3]` and `output_ptrs[0]`.

**MPK runtime partitioning policy**: `N` tasks per `mhc_post`
invocation. Identical wave behavior to `mhc_pre`.

**Alignment constraints**:

- `h_blk = gcd(H, 1024) = 1024` for `H=4096` → 4 pipelined stages.
- For tests with `H=128`, `h_blk = gcd(128, 1024) = 128` → 1 stage.
- `hc=4` baked in for v1 (template parameter, but only `hc=4` is
  instantiated initially).

### 3.6 `/add-mpk-task` conformance checklist

- [ ] `blockIdx`-agnostic: substitute the implicit `i_n` with
      `task_metadata.token_offset`. Verify 0 `blockIdx.*` references
      after port.
- [ ] `TaskType` enum entry: add `TASK_MHC_POST_SM100 = <next unused id>`
      in `runtime_header.h`.
- [ ] `register_<task>_task` codegen entry: add
      `register_mhc_post_sm100_task` in `task_register.cc`, modeled
      on `register_elementwise_add_sm100_task` (closest analog — 4
      inputs, 1 output, 1-D grid; if no such function exists, model
      after `register_rmsnorm_task`).
- [ ] `graph.cc` dispatch: add `else if (name == "mhc_post_sm100") { ...
      }`.
- [ ] Layer-method pattern: imitate `elementwise_add_layer`
      (`persistent_kernel.py:2566-2584`) for the partition tuples;
      4-input, 1-output signature.

### 3.7 Initial (naive) implementation strategy

Plain CUDA, 128 threads/CTA:

1. Load `post_mix[n, :]` (4 floats) and `comb_mix[n, :, :]` (16 floats)
   into registers per thread (via broadcast / smem load — they're tiny).
2. For each `h_blk` chunk of `H/h_blk_size` slices:
   - Load `x[n, h_blk]` bf16 → fp32 (h_blk_size threads).
   - Load `residual[n, 0..hc-1, h_blk]` bf16 → fp32.
   - Compute `out[i_hco, h] = post_mix[i_hco] * x[h] +
     sum_{i_hci} comb_mix[i_hci, i_hco] * residual[i_hci, h]` per
     output HC slot.
   - Cast back to bf16 and store `output[n, i_hco, h_blk]`.
3. No warp-spec needed at v1. Each thread handles `h_blk_size /
   blockDim.x` hidden elements per chunk.

### 3.8 Test-mode unit test plan

**File**: `tests/runtime_python/test_mode/test_mhc_post_testmode.py`.

**PyTorch oracle snippet** (extracted from `model.py:684-687`):

```python
def torch_mhc_post_ref(x_bf16, residual_bf16, post_mix_f32, comb_mix_f32):
    # x: [N, H] bf16, residual: [N, hc, H] bf16, post: [N, hc] f32, comb: [N, hc, hc] f32
    y = (post_mix_f32.unsqueeze(-1) * x_bf16.unsqueeze(-2).float()       # [N, hc, H]
         + torch.sum(comb_mix_f32.unsqueeze(-1) * residual_bf16.unsqueeze(-2).float(),
                     dim=2))                                              # contracts inner hc
    return y.to(torch.bfloat16)
```

**Test dims**: `N=4`, `hc=4`, `H=128`. Random `x`, `residual`,
`post_mix`, `comb_mix`. Tolerance: `rtol=1e-3, atol=1e-3` (output is
bf16; tighten to `atol=5e-3` if the v1 kernel rounds aggressively).

### 3.9 Reuse from V3

- `elementwise_add_layer` is the closest existing pattern (4 → 1
  shape; 1-D per-token). Reuse the partition tuples directly.
- If the v1 kernel is too slow to develop, a **decomposed v0**
  is possible:
  - `bmm = comb_mix.transpose(-2, -1) @ residual` (a small per-token
    bmm) — no existing MPK layer covers this exactly; would need a
    new tiny task.
  - `out = post_mix[..., None] * x[:, None, :] + bmm` — uses
    `elementwise_add_layer` plus a broadcast-mul (also no existing
    layer covers this exactly).
- Net: there is no free decomposition reusing only V3 layers; just
  write the v1 kernel.

---

## 4. Task 4 — `mhc_head`

### 4.1 Math formula

This is `ParallelHead.hc_head` from `model.py:729-736` and
`Transformer.forward` `model.py:809`. It collapses the `hc` HC copies
back to a single hidden vector after all blocks, just before the LM
head's norm + linear. The math is essentially `hc_pre` with a single
projection row per HC slot (vector instead of `hc3`-row matrix) and no
Sinkhorn / no `comb`. Specifically:

```
# Per token n
x[n, k]_f32  = residual[n, k // H, k % H]_f32                         for k ∈ [0, hc*H)
rms[n]       = rsqrt( (Σ_k x[n,k]^2) / (hc*H) + rms_eps )
mixes[n, m]  = ( Σ_k x[n, k] * fn[m, k] ) * rms[n]                    for m ∈ [0, hc)
pre[n, m]    = sigmoid( mixes[n, m] * hc_scale[0] + hc_base[m] ) + hc_eps     for m ∈ [0, hc)
out[n, h]_bf16 = Σ_{m ∈ [0, hc)} pre[n, m] * residual[n, m, h]_f32, cast to bf16
                                                                       for h ∈ [0, H)
```

The vLLM TileLang implementation
(`mhc.py:460-551`, `hc_head_fuse_tilelang`) is two-pass within a single
kernel:

- **Pass 1** (`mhc.py:495-520`): accumulate `sqrsum_r[0] = sum
  residual^2` and `mixes_r[m] = sum residual * fn[m, :]` via cross-
  thread reducers, while streaming `residual` block-by-block.
- **In-between** (`mhc.py:522-531`): compute `rsqrt` then
  `pre_mix_shared[m] = sigmoid(mixes_r[m] * rsqrt * hc_scale[0] +
  hc_base[m]) + hc_eps`.
- **Pass 2** (`mhc.py:535-549`): apply the weighted sum to produce
  `out[N, H]` bf16. This is the same compute as the warp≥1 branch of
  `mhc_pre` (§2.7, Phase C).

`hc_scale` is shape `[1]` here (just one scalar), not `[3]` — see
`mhc.py:488`. `hc_base` is `[hc]` not `[hc3]`. `fn` is `[hc, hc*H]`,
not `[hc3, hc*H]`. These are the model's *separate*
`hc_head_fn/scale/base` tensors (`model.py:751-753`, `798-800`), NOT
the per-block `hc_attn_fn` etc.

### 4.2 I/O tables

**Inputs**

| Name | Shape | Dtype | Source |
|---|---|---|---|
| `residual` | `[N, hc, H]` | `bfloat16` | output of the last Block (`model.py:807-809`) |
| `fn` | `[hc, hc*H]` | `float32` | `Transformer.hc_head_fn` (`model.py:798`) |
| `hc_scale` | `[1]` | `float32` | `Transformer.hc_head_scale` (`model.py:800`) |
| `hc_base` | `[hc]` | `float32` | `Transformer.hc_head_base` (`model.py:799`) |

**Outputs**

| Name | Shape | Dtype | Consumer |
|---|---|---|---|
| `output` | `[N, H]` | `bfloat16` | LM head's RMSNorm + `Linear` (`model.py:716, 722`) |

### 4.3 Source to migrate (cite + quote)

**Primary**: `mhc.py:460-551`. Quoted Pass 1 + transition + Pass 2
(lines 492-549):

```python
with T.Kernel(num_tokens, threads=n_thr) as i:
    T.pdl_sync()
    # Pass 1: per-token squared sum + hc_mult dot-products
    sqrsum_r = T.alloc_reducer((1,), T.float32, replication="all")
    mixes_r  = T.alloc_reducer((hc_mult,), T.float32, replication="all")
    T.fill(sqrsum_r, 0.0)
    T.fill(mixes_r, 0.0)
    for m_c in T.serial(hc_mult):
        for i_h in T.serial(n_h):
            x_local = T.alloc_fragment(h_block, T.float32)
            T.copy(residual[i, m_c, i_h * h_block], x_local)
            for k in T.Parallel(h_block):
                sqrsum_r[0] += x_local[k] * x_local[k]
            for m_m in T.unroll(hc_mult):
                fn_local = T.alloc_fragment(h_block, T.float32)
                T.copy(fn[m_m, m_c * hidden_size + i_h * h_block], fn_local)
                for k in T.Parallel(h_block):
                    mixes_r[m_m] += x_local[k] * fn_local[k]
    T.finalize_reducer(sqrsum_r)
    T.finalize_reducer(mixes_r)

    # Compute pre_mix
    pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
    rsqrt_val = T.alloc_fragment(1, T.float32)
    rsqrt_val[0] = T.rsqrt(sqrsum_r[0] / hc_dim + rms_eps)
    for m in T.Parallel(hc_mult):
        pre_mix_shared[m] = (
            T.sigmoid(mixes_r[m] * rsqrt_val[0] * hc_scale[0] + hc_base[m]) + hc_eps
        )

    # Pass 2: apply_mix
    for i0_h in T.Pipelined(n_h, num_stages=2):
        xs = T.alloc_shared((hc_mult, h_block), T.bfloat16)
        xl = T.alloc_fragment((hc_mult, h_block), T.float32)
        T.copy(residual[i, 0, i0_h * h_block], xs, disable_tma=True)
        T.copy(xs, xl)
        ol = T.alloc_fragment(h_block, T.float32)
        T.clear(ol)
        for i_hc in T.serial(hc_mult):
            pre = pre_mix_shared[i_hc]
            for i1_h in T.Parallel(h_block):
                ol[i1_h] += pre * xl[i_hc, i1_h]
        T.copy(ol, out[i, i0_h * h_block], disable_tma=True)
    T.pdl_trigger()
```

**Secondary**: `model.py:729-736` (the PyTorch reference, reproduced
in §0.3).

### 4.4 Generated CUDA reference

Path: `docs/mpk/deepseek_v4/_generated_cuda/mhc_head.cu`.

### 4.5 Grid design (mandatory)

**Natural grid dimensions**: `(N,)` — one CTA per token (`mhc.py:492`).
128 threads/CTA.

**How a CTA reads its slice from `task_metadata`**: token index
`n = task_desc->task_metadata.token_offset`. Same pattern as §2 / §3.
Input/output base pointers from `task_desc->input_ptrs[]` and
`output_ptrs[]`.

**MPK runtime partitioning policy**: `N` tasks per `mhc_head`
invocation (called once at the very end of the forward).

**Alignment constraints**:

- `h_block = gcd(1024, H) = 1024` for `H=4096`.
- `n_h = H / h_block = 4` for `H=4096`.
- For tests with `H=128`, `h_block = 128`, `n_h = 1`.

### 4.6 `/add-mpk-task` conformance checklist

- [ ] `blockIdx`-agnostic: substitute the TileLang `i` (lowered from
      `with T.Kernel(num_tokens, ...)`) with
      `task_metadata.token_offset`.
- [ ] `TaskType` enum entry: add `TASK_MHC_HEAD_SM100 = <next unused>`.
- [ ] `register_<task>_task` codegen entry: add
      `register_mhc_head_sm100_task`. Similar shape to `mhc_pre` but
      fewer outputs.
- [ ] `graph.cc` dispatch: add `else if (name == "mhc_head_sm100") { ...
      }`.
- [ ] Layer-method pattern: imitate `rmsnorm_layer`
      (`persistent_kernel.py:1015-1031`) — closest analog (token-major,
      1-D grid, 4 inputs / 1 output).

### 4.7 Initial (naive) implementation strategy

Almost identical to §2.7 Phase A + Phase C, but with `hc` rows of `fn`
instead of `hc3` rows, and no `comb` / Sinkhorn. Naive single-CTA-per-
token implementation with FP32 accumulators:

1. **Pass 1**: 128 threads stride through `residual[n, m_c, k]` for
   each `m_c ∈ [0, hc)`. Per-thread `sqrsum_local` and `mixes_local[hc]`
   accumulators. After the full stride, warp-reduce + smem-finalize to
   get `sqrsum_total` and `mixes_total[hc]`.
2. **rsqrt + sigmoid**: thread 0 computes `rsqrt = rsqrt(sqrsum_total /
   (hc*H) + rms_eps)`. All threads compute `pre_smem[m] = sigmoid(
   mixes_total[m] * rsqrt * hc_scale[0] + hc_base[m]) + hc_eps` and
   store to smem.
3. **Pass 2**: same as `mhc_pre`'s Phase C — chunk `H` into `h_block`
   slices, weighted-sum across `hc`, cast to bf16, store.

### 4.8 Test-mode unit test plan

**File**: `tests/runtime_python/test_mode/test_mhc_head_testmode.py`.

**PyTorch oracle snippet** (extracted from `model.py:729-736`):

```python
def torch_mhc_head_ref(residual_bf16, fn_fp32, hc_scale, hc_base,
                       rms_eps=1e-6, hc_eps=1e-6):
    # residual: [N, hc, H] bf16, fn: [hc, hc*H] fp32,
    # hc_scale: [1], hc_base: [hc]
    x = residual_bf16.flatten(1).float()                           # [N, hc*H]
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)
    mixes = torch.nn.functional.linear(x, fn_fp32) * rsqrt         # [N, hc]
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps
    out = (pre.unsqueeze(-1) * residual_bf16.float()).sum(dim=1)   # [N, H]
    return out.to(torch.bfloat16)
```

**Test dims**: `N=4`, `hc=4`, `H=128`. Tolerance: `rtol=1e-3,
atol=1e-3` for the bf16 output. (Tighten if necessary; the kernel
keeps fp32 throughout except the final cast.)

### 4.9 Reuse from V3

- Pass-2 (apply_mix) is the same compute as `mhc_pre` Phase C — share
  a `__device__` helper between `mhc_pre_sm100.cuh` and
  `mhc_head_sm100.cuh` (e.g. a templated `hc_apply_pre_mix<T, hc>`).
- Pass-1 sqrsum+dot resembles `rms_norm_sm100`'s squared-sum loop
  (`norm_sm100.cuh:60-67`) but accumulates an extra `hc`-dim
  dot-product. The helper can be cloned and extended.
- The `TBGraph` scaffolding (4 inputs, 1 output, 1-D grid) is closest
  to `linear_fp8_layer` minus the scales — no exact V3 layer matches;
  cleanly new layer method.

---

## 5. Module-level test plan for HC

Wave 3 adds a single end-to-end "HC module" test:
`tests/runtime_python/test_mode/test_hc_module_testmode.py`. It chains
the four kernels in the same order as a real layer would use them:

```
embed = torch.randn(N, hc, H, bf16)
fn_attn, base_attn, scale_attn = ... (loaded from a real Flash-Base layer
                                       or random for the v1 module test)
# Stage 1: mhc_pre on residual=embed
gemm_out_mul, gemm_out_sqrsum = pk.mhc_prenorm_gemm(embed, fn_attn)
post_mix, comb_mix, layer_input = pk.mhc_pre(gemm_out_mul, gemm_out_sqrsum,
                                              scale_attn, base_attn, embed)
# Stage 2: simulate attn sub-block as identity → x = layer_input
# Stage 3: mhc_post
out = pk.mhc_post(layer_input, embed, post_mix, comb_mix)
# Stage 4: mhc_head on the final HC state
logits_input = pk.mhc_head(out, fn_head, scale_head, base_head)

# Reference: run the PyTorch hc_pre, identity, hc_post, hc_head on the same
# inputs and assert torch.allclose(rtol=1e-3, atol=1e-3) on each intermediate.
```

This catches dtype/shape contract bugs at the boundaries (e.g. `[N, hc,
hc]` vs `[N, hc*hc]` for `comb_mix`).

Module gate: pass `torch.allclose(rtol=1e-3, atol=1e-3)` on `out` and
on `logits_input`.

---

## 6. Wiring summary (additive changes only)

All changes are additive — no V3 paths touched. Each item below is a
single line / block to add.

### 6.1 `include/mirage/persistent_kernel/runtime_header.h`

Add four `TaskType` enum entries in the SM100 range. Pick the next
unused IDs (current high-water mark in SM100 range = 295 per
`runtime_header.h:196`, with `TASK_SM100_TASK_END = 298`). Suggested:

```cpp
TASK_MHC_PRENORM_GEMM_SM100 = 296,
TASK_MHC_PRE_SM100          = 297,
// If we hit the 298 cap, bump TASK_SM100_TASK_END to 310 in the same diff.
TASK_MHC_POST_SM100         = 299,   // after extending the range
TASK_MHC_HEAD_SM100         = 300,
```

If extending the range, also extend `TASK_SM100_TASK_END` and any
range-checking macros. **Open Question 1** below.

Also: consider extending the `TaskMetadata` union with an explicit
`token_offset` field. The current union has `expert_offset`,
`request_id/kv_idx/merge_task_offset`, `task_offset`, and a raw
`raw_payload`. For the per-token mHC tasks the simplest thing is to
**alias `task_offset`** as `token_offset` (i.e. reuse the existing slot
since mHC tasks do not use nvshmem team-mapping). **Open Question 2**.

### 6.2 `src/kernel/graph.cc`

Add four `else if` branches in the SM100 cluster (near `graph.cc:787`,
the `linear_fp8_sm100` block). Pattern (sketch):

```cpp
else if (name == "mhc_prenorm_gemm_sm100") {
    int variant_id = task_register->register_mhc_prenorm_gemm_sm100_task(
        customized->bgraph, params);
    task_config[op] = std::make_tuple(2, 2, TASK_MHC_PRENORM_GEMM_SM100, variant_id);
} else if (name == "mhc_pre_sm100") {
    int variant_id = task_register->register_mhc_pre_sm100_task(
        customized->bgraph, params);
    task_config[op] = std::make_tuple(5, 3, TASK_MHC_PRE_SM100, variant_id);
} else if (name == "mhc_post_sm100") {
    int variant_id = task_register->register_mhc_post_sm100_task(
        customized->bgraph, params);
    task_config[op] = std::make_tuple(4, 1, TASK_MHC_POST_SM100, variant_id);
} else if (name == "mhc_head_sm100") {
    int variant_id = task_register->register_mhc_head_sm100_task(
        customized->bgraph, params);
    task_config[op] = std::make_tuple(4, 1, TASK_MHC_HEAD_SM100, variant_id);
}
```

The `std::make_tuple(in, out, task_type, variant_id)` matches the
existing FP8 entries (see `graph.cc:781-795`).

### 6.3 `src/kernel/task_register.cc`

Add four `TaskRegister::register_mhc_*_sm100_task` methods. Each one:

- Validates `bgraph.operators.size() == num_inputs + num_outputs`.
- Reads the relevant dims (`N`, `H`, `hc`, `hc3`, `splits`) off the
  input ops' DTensor metadata.
- Emits a `code.e(...)` block that lowers to a call into
  `kernel::mhc_*_task_impl<...>(task_desc->input_ptrs[i],
  task_desc->output_ptrs[j], <constants>)`.
- Returns `register_task_variant(TASK_MHC_*_SM100, code.to_string())`.

Closest existing patterns to copy:

- `register_quantize_fp8_sm100_task` (`task_register.cc:4020-4081`) —
  2-in / 2-out, picks up dims from DTensor, emits a templated
  `kernel::per_token_group_quantize_fp8_task_impl<...>` call.
- `register_linear_fp8_sm100_task` (`task_register.cc:4083+`) — 4-in /
  1-out.

### 6.4 `python/mirage/mpk/persistent_kernel.py`

Append four layer methods (additive). Sketch for `mhc_pre_layer`:

```python
def mhc_pre_layer(
    self,
    gemm_out_mul: DTensor,
    gemm_out_sqrsum: DTensor,
    hc_scale: DTensor,
    hc_base: DTensor,
    residual: DTensor,
    post_mix: DTensor,
    comb_mix: DTensor,
    layer_input: DTensor,
    grid_dim: tuple,
    block_dim: tuple,
):
    assert gemm_out_mul.num_dims == 3        # [splits, N, hc3]
    assert gemm_out_sqrsum.num_dims == 2     # [splits, N]
    assert hc_scale.num_dims == 1
    assert hc_base.num_dims == 1
    assert residual.num_dims == 3            # [N, hc, H]
    assert post_mix.num_dims == 2            # [N, hc]
    assert comb_mix.num_dims == 3            # [N, hc, hc]
    assert layer_input.num_dims == 2         # [N, H]
    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(gemm_out_mul,    (-1, -1, -1), -1, True)
    tb_graph.new_input(gemm_out_sqrsum, (-1, -1, -1), -1, True)
    tb_graph.new_input(hc_scale,        (-1, -1, -1), -1, True)
    tb_graph.new_input(hc_base,         (-1, -1, -1), -1, True)
    tb_graph.new_input(residual,        (-1, -1, -1), -1, True)
    tb_graph.new_input(post_mix,        (-1, -1, -1), -1, True)
    tb_graph.new_input(comb_mix,        (-1, -1, -1), -1, True)
    tb_graph.new_input(layer_input,     (-1, -1, -1), -1, True)
    self.kn_graph.customized(
        [gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual,
         post_mix, comb_mix, layer_input], tb_graph)
    self.kn_graph.register_task(tb_graph, "mhc_pre_sm100")
```

Notes:
- All partitions are `(-1,-1,-1), -1, True` because the kernel does
  per-token routing via `task_metadata`, not via grid partition.
- The signature has 5 inputs + 3 outputs = 8 operators, which is at
  `MAX_INPUTS_PER_TASK + MAX_OUTPUTS_PER_TASK = 7 + 3 = 10` — fits.
  See **Open Question 3**: do all 5 inputs need to be `new_input`? V3
  layers use `new_input` for outputs too (e.g. `rmsnorm_layer:1029`
  treats `output` as `new_input` — confusing but consistent). We
  follow that convention.

The other three layer methods follow the same skeleton; `mhc_prenorm_gemm_layer` has 2 inputs + 2 outputs; `mhc_post_layer` has 4 inputs + 1 output; `mhc_head_layer` has 4 inputs + 1 output.

### 6.5 No changes to the Cython binding

`register_task` and `customized` already accept arbitrary task names
and forward to the C++ dispatch in `graph.cc`. No `.pyx` edits needed.

### 6.6 Recompile

After every `src/` change (`graph.cc`, `task_register.cc`,
`runtime_header.h`):

```bash
pip install -e . -v --no-deps
```

(per `CLAUDE.md` §"Recompiling after C++ changes").

---

## 7. Open Questions

**OPEN: 1** — `TASK_SM100_TASK_END = 298` in `runtime_header.h:197`
leaves only `296, 297` free for new tasks before we hit the end
sentinel. We have 4 new mHC tasks **plus** the V4 attention pipeline
(5 tasks per `attention.md`), V4 sparse (3 tasks per `sparse.md`), V4
MoE (3 tasks per `moe.md`), and V4 MTP (1 task per `mtp.md`). That's
16 new SM100 tasks total against 2 free slots. We need to bump
`TASK_SM100_TASK_END` up to at least `320` to fit Wave 2. Owner: the
agent who lands the first V4 task (likely `mhc_prenorm_gemm` or the
attention RMSNorm) should bundle the range extension into their PR. No
existing references depend on the 230..298 range numerically
(`grep -n 'TASK_SM100_TASK_END' src/ include/` to confirm).

**OPEN: 2** — `TaskMetadata` union: should `token_offset` be a new
field, or alias the existing `task_offset`? Aliasing is zero-cost but
risks confusing readers ("why is my mHC token routed through an
nvshmem field?"). Recommendation: add a new struct branch
`struct { int token_offset; }` alongside the existing branches. The
union is 8 bytes (`unsigned long long raw_payload`), so an extra
`int32` branch costs nothing. The `static_assert` at
`runtime_header.h:273-275` only checks total size, so the change is
non-breaking. Owner: the first per-token V4 task to land (probably
`mhc_pre`) adds the field.

**OPEN: 3** — Thread count for `mhc_pre`: vLLM uses 96 threads (3
warps; 1 warp for post/comb/Sinkhorn + 2 warps for pre/mix). MPK's
Blackwell convention is `WORKER_NUM_THREADS = 256`. We can: (a) use
128 threads (1 warp for warp-0 branch + 3 warps for pre/mix branch),
(b) use 256 threads (warps 0 + 1..7), (c) match vLLM at 96. (a) is
cleanest. Recommendation: **128 threads**. Verify wave occupancy on
B200 during implementation.

**OPEN: 4** — Output dtype of `comb_mix`: vLLM stores it as flat
`[N, hc*hc] f32` (`mhc.py:78, 148-149`); PyTorch's `hc_pre` semantically
treats it as `[N, hc, hc]`. Pick one shape and document it in the
layer method's docstring. Recommendation: **store `[N, hc, hc]`** in
MPK (matches PyTorch reference and is what `mhc_post` expects per
`model.py:686`). The TileLang storage as flat `[N, hc*hc]` is purely a
linear-memory convention; `[N, hc, hc]` and `[N, hc*hc]` are the same
bytes with different DTensor metadata.

**OPEN: 5** — Should `mhc_pre` and `mhc_prenorm_gemm` be fused into a
single MPK task? vLLM keeps them separate (two distinct kernel calls
in `mhc.py:277-302`) because the deep-gemm kernel uses tcgen05 / TMA
and the post-gemm work is per-token serial. For MPK v1 with the
decomposed scalar-CUDA prenorm-GEMM fallback, fusion *is* possible and
would eliminate the `gemm_out_mul` / `gemm_out_sqrsum` global
materialization. Recommendation: **keep separate** to match vLLM
boundaries (per plan §H.S.2). Revisit in v2 if profiling shows
the global writeback is a bottleneck.

**OPEN: 6** — `hc_post_mult_value`: vLLM passes this as a parameter
(`mhc.py:61, 189, 204`), default not in the signature — caller must
provide. PyTorch reference (`kernel.py:394`) hardcodes `2`. Confirm
the `2.0` value by reading the Flash-Base config (already done:
`model.py:686` and `kernel.py:394` show `2 * sigmoid(...)`). Bake `2.0`
as a compile-time constant in the v1 MPK kernel; revisit if config
ever varies.

**OPEN: 7** — Test fixture for HC weights: the prenorm `fn` is fp32
and `hc3 × hc*H = 24 × 16384 = ~1.5MB` per layer. For unit tests we
generate random fp32. For the module test we should also use random
weights (not real Flash-Base) to keep the test self-contained — the
per-layer real-weight check happens in Wave 3 module tests against an
instantiated `Block`.

**OPEN: 8** — Should `mhc_prenorm_gemm` honor `splits > 1` in v1? The
scalar v1 fallback naturally has `splits=1` (one CTA reduces fully per
token). Supporting `splits > 1` requires multiple CTAs writing into
`gemm_out_mul[s, n, :]` and `gemm_out_sqrsum[s, n]` simultaneously,
plus the `mhc_pre` reduce-across-splits. Recommendation: **v1
`splits=1` only**; v2 add split-K for performance. The DTensor shape
should still allocate the leading `splits` dim so `mhc_pre` does not
need recompilation when v2 lands.

---

## 8. Cross-references

- Plan: `~/.claude/plans/i-want-to-add-dapper-pascal.md`, especially
  §"Mandatory kernel-authoring requirements" (lines 75-132) and the
  Wave-2 task list at lines 234-242.
- Sibling specs (Wave 1): `overview.md`, `attention.md`, `sparse.md`,
  `moe.md`, `mtp.md` (same directory).
- MPK conventions: `CLAUDE.md` §"Key Concepts" (Task,
  blockIdx-agnostic rule); `python/mirage/mpk/persistent_kernel.py`
  §`rmsnorm_layer` (line 1015), §`linear_fp8_layer` (line 1990),
  §`quantize_fp8_layer` (line 1967), §`silu_mul_layer` (line 2529),
  §`elementwise_add_layer` (line 2566).
- Existing CUDA tasks for style: `norm_sm100.cuh:1-100`
  (`rms_norm_sm100`), `linear_fp8_sm100.cuh` (FP8 GEMM scaffold).
- `/add-mpk-task` skill: reference for the full add-task workflow.
- `/test-mode` skill: reference for the test_mode harness pattern.
