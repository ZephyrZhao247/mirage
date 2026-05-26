# DeepSeek V4-Flash — MoE deltas (Wave-1 spec)

> Status: Wave-1 spec only. No code changes are made by this document.
> Scope: only the MoE-side deltas vs. V3. Attention / mHC / sparse / MTP are in
> sibling specs.

## Wave 1.5 corrections

MTP block (layer 43) uses the full MoE pipeline (256 routed + 1 shared experts), not a dedicated MLP. Verified via checkpoint keys `mtp.0.ffn.experts.{0..255}.*` (model.safetensors.index.json). The V4 MoE orchestration in §4 applies unchanged for layer 43, with `sqrtsoftplus` routing (layer 43 is NOT in `[0..2]` hash range).

## Why most of V3 MoE is reusable

The Flash-Base MoE block uses the same **expert-as-FP8-group-GEMM** layout that
V3 already implements:

* Routing -> per-token top-K weights + per-expert routing indices.
* Per-token FP8 quant of the routed hidden state.
* `moe_w13_fp8_layer` group GEMM (256 experts, 2048 intermediate dim).
* Per-position activation (V3 = `silu(gate) * up`; V4 = same but **clamped**).
* Per-expert-tok FP8 quant of the activation output.
* `moe_w2_fp8_layer` group GEMM.
* `moe_mul_sum_add_layer` combines K routed experts + (shared expert + residual).
* Shared expert = dense BF16/FP8 SwiGLU MLP that runs on every token.

The genuinely-new V4 deltas are limited to:

1. **`hash_route_lookup_layer`** (NEW). Used in `layer_idx < num_hash_layers = 3`
   in place of router GEMM + topk. Per-token expert ids are fetched from a
   precomputed table `tid2eid[vocab_size, num_experts_per_tok]` keyed by
   `input_ids[t]`. Routing weights and the mask are derived from a uniform
   weight (`1/K` after normalization, scaled by `routed_scaling_factor`).
   (`deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:556-584`,
   `deps/vllm/vllm/model_executor/models/deepseek_v4.py:754-769, 856-877`,
   `deps/vllm/csrc/moe/topk_softplus_sqrt_kernels.cu:96-260`.)

2. **`swiglu_clamped_layer`** — additive variant of the existing
   `silu_mul_layer`/`moe_silu_mul_layer`. Adds an L=10.0 clamp on gate
   (`gate ≤ L`) and on up (`-L ≤ up ≤ L`).
   (`deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:587-606`,
   `deps/vllm/vllm/model_executor/models/deepseek_v4.py:104-117`.)

3. **`sqrtsoftplus_topk_layer`** — additive variant of `moe_topk_sigmoid_routing_layer`.
   Replaces `sigmoid(logits)` with `sqrt(softplus(logits))`. Bias addition,
   top-K selection, `renormalize`, and `routed_scaling_factor` are unchanged
   semantically; the existing per-group selection used for V3 (8 groups, top
   4 groups) is dropped because V4 uses **flat** top-K over 256 experts
   (`topk_method = "noaux_tc"`, no group structure).
   (`deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:564-584`,
   `deps/vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py:60-228`.)

4. **DROPPED: `quantize_fp8_ue8m0_layer`**. UE8M0-scaled per-token-group FP8
   quantization is already in MPK as
   `quantize_fp8_layer(scale_ue8m0=True)` →
   `quantize_fp8_sm100`
   (`python/mirage/mpk/persistent_kernel.py:1967-1988`;
   kernel:
   `include/mirage/persistent_kernel/tasks/blackwell/per_token_group_quantize_fp8.cuh:33-113`;
   register: `src/kernel/task_register.cc:4020-4081`). **Reuse — do NOT
   propose a new task.**

## Key dimensions (Flash-Base, verified from
`/raid/catalyst/models/DeepSeek-V4-Flash-Base/config.json`)

| Symbol | Value | Source |
|---|---|---|
| `T` | runtime: tokens in current batch (≤ `max_num_batched_tokens`) | n/a |
| `E` (`n_routed_experts`) | **256** | config |
| `K` (`num_experts_per_tok`) | **6** | config |
| `D` (`hidden_size`) | 4096 | config |
| `I` (`moe_intermediate_size`) | 2048 | config |
| `num_hash_layers` | **3** (layers 0..2 use hash routing) | config |
| `scoring_func` | `"sqrtsoftplus"` (layers 3..42) | config |
| `topk_method` | `"noaux_tc"` (flat top-K + bias) | config |
| `norm_topk_prob` | `true` | config |
| `routed_scaling_factor` | `1.5` | config |
| `swiglu_limit` (L) | `10.0` | config |
| `expert_dtype` | `"fp8"` | config |
| FP8 group size | `128` (per-token-group quant) | MPK `per_token_group_quantize_fp8.cuh` |
| `n_shared_experts` | `1` | config |
| Shared expert intermediate size | `1 * I = 2048` | derived |
| `vocab_size` (V) | `129280` | config |
| Total MoE layers | 41 of 43 base layers (`first_k_dense_replace` ≥ 0; layer 0..2 also MoE with hash routing per `model.py:556`) | config |

> **Layer-0..2 dense vs MoE.** The official `Block` in `model.py`
> instantiates `MoE` for `layer_id ≥ first_k_dense_replace`. The Flash-Base
> config sets `first_k_dense_replace = 0`, so **every** non-MTP transformer
> block has the MoE FFN. Layers 0..2 select hash routing inside `Gate`
> (`layer_id < n_hash_layers`).
> (`deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:556`,
> `deps/vllm/vllm/model_executor/models/deepseek_v4.py:754`.)

## Layer-routing map

| `layer_idx` | Routing path | Reason |
|---|---|---|
| `0`, `1`, `2` | **hash** — `hash_route_lookup_layer` | `layer_idx < num_hash_layers=3` |
| `3..42` | **scored** — `linear_layer` (gate) + `sqrtsoftplus_topk_layer` | `scoring_func="sqrtsoftplus"` + `topk_method="noaux_tc"` |
| `43` (MTP) | (no MoE; MTP block uses its own FFN — see `mtp.md`) | per `compress_ratios` mapping and `mtp.md` |

## Reused-from-V3 tasks (no edits required)

| Task name | Python entry | C++ kernel | Used for |
|---|---|---|---|
| `quantize_fp8_sm100` (UE8M0) | `quantize_fp8_layer(scale_ue8m0=True)` (`persistent_kernel.py:1967-1988`) | `per_token_group_quantize_fp8.cuh:33-113` | (Not needed for MoE group-GEMM input; MoE wants float32 scale. UE8M0 is needed for the **dense** FP8 paths of the router GEMM input and the shared expert.) |
| `quantize_fp8_f32scale_sm100` | `quantize_fp8_layer(scale_ue8m0=False)` (`persistent_kernel.py:1967-1988`) | same kernel, `scale_ue8m0=false` template arg | MoE group-GEMM input quant (matches V3 use in `deepseek_v3/builder.py:946-952, 1025-1032`) |
| `moe_w13_fp8_sm100` | `moe_w13_fp8_layer` (`persistent_kernel.py:1881-1922`) | `fp8_group_gemm_sm100.cuh` | FP8 W1+W3 group GEMM, `E=256`, output `[T, K, 2*I]` |
| `moe_w2_fp8_sm100` | `moe_w2_fp8_layer` (`persistent_kernel.py:1924-1964`) | `fp8_group_gemm_sm100.cuh` | FP8 W2 group GEMM, output `[T, K, D]` |
| `moe_mul_sum_add_sm100` | `moe_mul_sum_add_layer` (`persistent_kernel.py:2386-2407`) | `moe_mul_sum_add_sm100.cuh` | combine `K` routed experts + `(residual + shared_expert)` |
| `rmsnorm_sm100` | `rmsnorm_layer` | `rmsnorm_sm100.cuh` | pre-MoE norm |
| `linear_sm100` | `linear_layer` | `linear_sm100.cuh` | router gate GEMM `x[T,D] @ gate.weight[E,D]^T -> [T,E]` |
| `linear_fp8_sm100` | `linear_fp8_layer` | `linear_fp8_sm100.cuh` | shared-expert gate/up + down (FP8 path) |
| `silu_mul` | `silu_mul_layer` | `ampere/silu_mul.cuh` (used on Blackwell too) | shared-expert SwiGLU **with clamp=10** (after Wave-2 extension; see §2 below) |

All checkpoints in Flash-Base ship FP8 expert weights with `weight_scale_inv`
(see `deepseek_v3/builder.py:923-933` for the exact rescale + repeat_interleave
pattern that MPK V4 must reuse verbatim).

---

# 1. `hash_route_lookup_layer` — NEW

## 1.1 Task name + file

* Python entry: `hash_route_lookup_layer` on
  `PersistentKernel` (`python/mirage/mpk/persistent_kernel.py`, additive).
* C++ kernel: **NEW** file
  `include/mirage/persistent_kernel/tasks/blackwell/hash_route_lookup_sm100.cuh`.
* Task name string (graph.cc dispatch + register_task): `hash_route_lookup_sm100`.
* `TaskType` enum entry: `TASK_HASH_ROUTE_LOOKUP_SM100 = 296`
  (next free slot after `TASK_MLA_PREFILL_TP8_SM100 = 295` and before
  the `TASK_SM100_TASK_END = 298` placeholder; verified
  `include/mirage/persistent_kernel/runtime_header.h:195-197`).

## 1.2 Math

For each token `t` in the batch:

```
indices[t, k]  = tid2eid[input_ids[t], k]                # k = 0..K-1
weights_raw[t, k] = 1.0                                  # uniform
weights[t, k]  = (weights_raw[t, k] / sum_k weights_raw) * routed_scaling_factor
              = (1/K) * routed_scaling_factor
```

That is, all K=6 indices come from the lookup; the routing weight is identical
across the K activated experts: `1/6 * 1.5 = 0.25`.

> **Why uniform weights?** The official `Gate.forward` in hash mode uses
> `weights = original_scores.gather(1, indices)` and `original_scores =
> sqrt(softplus(scores))`, then `weights /= weights.sum(...); weights *=
> route_scale`. But in the hash path the gate matmul + softplus + sqrt is
> *still computed*, only the `indices` come from the table.
> (`deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:564-584`.)
>
> **OPEN (BIG):** vLLM's `_compute_routing` in `fused_topk_bias_router.py:213-228`
> *also* recomputes scores (via `vllm_topk_softplus_sqrt` -> CUDA kernel) when
> `hash_indices_table is not None`. The CUDA kernel
> (`deps/vllm/csrc/moe/topk_softplus_sqrt_kernels.cu:240-260`) **gathers the
> scores at the hash-provided indices, then normalizes**. So routing weights
> are **NOT** uniform; they are the score-gather. This contradicts the naive
> "uniform weights" reading. We must either:
> (a) Make `hash_route_lookup_layer` produce only `indices`/`mask` and *still*
>     run the gate GEMM + `sqrtsoftplus_topk_layer` (with `hash_indices`
>     overriding the topk argmax). The vLLM kernel does exactly this.
> (b) Skip the gate GEMM in hash layers (faster) and use uniform weights;
>     this matches the *plain reading* of `model.py:580-583` but **diverges**
>     numerically from vLLM and from the checkpoint, because the gate weight
>     is shipped in the checkpoint shards for layers 0..2 even with hash
>     routing (verified by `model.safetensors.index.json:1570,3135,4711`
>     enumerating `layers.{0,1,2}.ffn.gate.tid2eid`, and by `Gate.__init__`
>     allocating `self.weight` unconditionally for both branches —
>     `model.py:557`).
>
> Resolution adopted by this spec for Wave-2: **option (a)** —
> `hash_route_lookup_layer` writes only `(routing_indices, mask)`, and the
> usual `moe_topk_weights` is filled by the *same* `sqrtsoftplus_topk_layer`
> running with a new kwarg `hash_indices_override` pointing at the lookup
> output. This mirrors the vLLM kernel and the official `original_scores.gather`
> behavior exactly. The implementer of `sqrtsoftplus_topk_layer` (§3) MUST
> support this override path. If profiling proves the gate GEMM unjustified,
> we can switch to option (b) as a perf optimization in a follow-up.

## 1.3 Inputs / Outputs / Weights

| Role | Tensor | Shape | dtype | Source |
|---|---|---|---|---|
| **Input** | `input_ids` | `[T]` | `int32` | `meta_tokens` runtime tensor (existing MPK plumbing) |
| **Weight** | `tid2eid` | `[V, K]` = `[129280, 6]` | `int32` | checkpoint key `layers.{layer_idx}.ffn.gate.tid2eid` (per `model.safetensors.index.json:1570,3135,4711`) |
| **Output** | `moe_routing_indices` | `[E, T]` = `[256, T]` | `int32` | same layout as V3 `moe_topk_sigmoid_routing_layer` (`persistent_kernel.py:1832, 1842`) — expert-major, `routing_indices[e][t] = k+1` if token `t` picks expert `e` at slot `k`, else 0 |
| **Output** | `moe_mask` | `[E+1]` = `[257]` | `int32` | same layout as V3 (`persistent_kernel.py:1833`) — compacted active-expert list with `mask[E]` as the count |

> **Why not write `moe_topk_weights` here?** Because under the adopted
> "option (a)" the weights are score-gathered, not uniform — see §1.2 OPEN
> resolution. The weight tensor is filled by `sqrtsoftplus_topk_layer` in
> §3 below.

## 1.4 Parameterization

No template parameters needed (`E`, `K`, `V` are deduced from tensor shapes by
the register function).

## 1.5 Source to migrate

* Reference impl (Triton-style fused kernel):
  `deps/vllm/csrc/moe/topk_softplus_sqrt_kernels.cu:240-260`
  (the per-token `expert_indices_for_token = tid2eid + token_id * k`
  followed by routing-table write — quote follows):

```cpp
// deps/vllm/csrc/moe/topk_softplus_sqrt_kernels.cu:240-260
if (use_hash) {
  const IndType token_id = input_ids[thread_row];
  const IndType* expert_indices_for_token = tid2eid + token_id * k;
#pragma unroll
  for (int ii = 0; ii < VPT; ++ii) {
    int local = expert_id_offset + ii;
    bool is_chosen = false;
    int chosen_k = -1;
    for (int kk = 0; kk < k; ++kk) {
      if (expert_indices_for_token[kk] == local) {
        is_chosen = true;
        chosen_k = kk;
        break;
      }
    }
    if (is_chosen) {
      // ... write weight at slot chosen_k, write index ...
    }
  }
}
```

The CUDA snippet above interleaves the hash lookup with the score-gather; for
MPK we split that into two kernels per §1.2 option (a).

* Python orchestration: `deps/vllm/vllm/model_executor/models/deepseek_v4.py:856-877`
  (forward path; `hash_indices_table=self.gate.tid2eid`).
* Official reference (PyTorch):
  `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:564-584`.

## 1.6 Generated CUDA reference

N/A — there is no TileLang kernel for this. The op is a small table lookup with
no GEMM and no FMA. Reference is the CUDA snippet above and the PyTorch:

```python
# deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:576-583
if self.hash:
    indices = self.tid2eid[input_ids]              # [T, K]
else:
    indices = scores.topk(self.topk, dim=-1)[1]
weights = original_scores.gather(1, indices)       # gather sqrtsoftplus scores
if self.score_func != "softmax":
    weights /= weights.sum(dim=-1, keepdim=True)
weights *= self.route_scale
```

## 1.7 Grid design (`/add-mpk-task` requirement #3)

* **Natural grid**: `(num_tokens,) = (T,)`. One CTA per token.
* **CTA work**: a CTA covers one token. It reads `tid2eid[input_ids[t], :]`
  (`K=6` int32s, well under a warp's worth of bandwidth), then atomically
  marks routing indices + mask entries for those `K` experts.
* **MPK scheduler partition policy**: `grid_dim = (max_num_batched_tokens, 1, 1)`.
  Each runtime CTA derives `t = task_desc->task_metadata.tile_id`. There is no
  `blockIdx` use. Concretely, the kernel reads
  `t = task_desc->task_metadata[0]` (token id within the batch). Threads in
  the CTA iterate `K=6` slots.
* **Alignment**: none. `K=6` is fine.
* **block_dim**: `(32, 1, 1)` (a single warp suffices). Use 128 for symmetry with
  other small layers if convenient — the loop bound is `K`.

## 1.8 `/add-mpk-task` conformance checklist

- [x] **`blockIdx`-agnostic**: yes; `t` comes from `task_desc->task_metadata[0]`,
      not `blockIdx.x`.
- [x] **TaskType enum entry**: `TASK_HASH_ROUTE_LOOKUP_SM100 = 296` in
      `include/mirage/persistent_kernel/runtime_header.h` (insert before
      `TASK_SM100_TASK_END = 298`).
- [x] **`register_<task>_task` in `src/kernel/task_register.cc`**: NEW
      `register_hash_route_lookup_sm100_task` — pattern modeled on
      `register_moe_topk_sigmoid_sm100_task` (`task_register.cc:2342-2407`),
      simpler because there is no template-parameter explosion: emit
      `kernel::hash_route_lookup_task_impl<E, K>(...)` and the input/output
      pointers (`input_ids`, `tid2eid`, `moe_routing_indices`, `moe_mask`).
- [x] **Dispatch in `src/kernel/graph.cc`**: NEW
      `else if (name == "hash_route_lookup_sm100") { variant_id =
      task_register->register_hash_route_lookup_sm100_task(customized->bgraph,
      params); }` — pattern modeled on `graph.cc:632-642`.
- [x] **Python `hash_route_lookup_layer` method** in `persistent_kernel.py`:
      structure same as `moe_topk_sigmoid_routing_layer` minus the bias
      input and the weights output (`persistent_kernel.py:1814-1848`).
- [x] **NOT included in the `_execute_task` allowlist branch in
      `persistent_kernel.cuh:691-700`** unless test mode needs the extra
      printf for debugging. (Optional; mirror the `280` entry for the
      sigmoid topk task if needed.)

## 1.9 Initial (naive) implementation

Naive single-warp-per-token version. Pseudocode for `hash_route_lookup_task_impl`:

```cpp
// hash_route_lookup_sm100.cuh, naive impl
template <int NUM_EXPERTS /*256*/, int TOPK /*6*/>
__device__ __forceinline__ void hash_route_lookup_task_impl(
    int const *input_ids,                              // [T]
    int const *tid2eid,                                // [V, TOPK]
    int *moe_routing_indices,                          // [NUM_EXPERTS, T]
    int *moe_active_expert_ids,                        // [NUM_EXPERTS + 1]
    int t,
    int T) {
  if (t >= T) return;
  int tok = input_ids[t];
  // Phase 0: zero routing_indices[*][t] for this token (cooperative)
  for (int e = threadIdx.x; e < NUM_EXPERTS; e += blockDim.x) {
    moe_routing_indices[e * T + t] = 0;
  }
  __syncthreads();
  // Phase 1: K experts for this token. K is tiny (6) -> single warp.
  if (threadIdx.x < TOPK) {
    int e = tid2eid[tok * TOPK + threadIdx.x];
    moe_routing_indices[e * T + t] = threadIdx.x + 1;   // 1-indexed slot, matches V3
    // mark active-expert list: atomic add a sentinel
    atomicMax(&moe_active_expert_ids[e], e);
  }
  __syncthreads();
  // Phase 2: compaction of active-expert list. The K experts for THIS token
  // may have already been marked by other tokens' kernels (different CTAs).
  // Use the same atomic compaction pattern as topk_sigmoid_sm100.cuh:351-361.
}
```

Notes:

* Phase 0 zeros the **whole column** `moe_routing_indices[:, t]`. This is the
  same per-token zeroing the sigmoid kernel does (`topk_sigmoid_sm100.cuh:104-114`).
  Cost: 256 writes per token.
* Phase 2 (compaction of active-expert list) reuses the atomic pattern from
  `topk_sigmoid_sm100.cuh:350-361`. We can't run it inside this CTA only
  (other tokens' kernels need to write), so it runs as a per-CTA finalize
  block atomically updating the shared `moe_active_expert_ids` array.
* For Wave-2, the implementer SHOULD inspect the
  `topk_softplus_sqrt_kernels.cu` CUDA reference for the exact write-pattern,
  since vLLM has battled the race conditions in this exact place.

> **OPEN:** the V3 `moe_topk_sigmoid_routing_layer` runs as a **single CTA**
> over all rows (`grid_dim=(1, 1, 1)`, `block_dim=(256, 1, 1)` —
> `deepseek_v3/builder.py:909-911`). That gives a single thread-block sole
> writer of `moe_active_expert_ids`, sidestepping cross-CTA races. We should
> do the same for `hash_route_lookup_layer`: `grid_dim=(1, 1, 1)`,
> `block_dim=(256, 1, 1)`, kernel iterates all `T` tokens internally. Adopt
> this pattern → Phase 0/2 inside the same CTA, no atomics needed across
> tokens. **Final decision**: single-CTA implementation.

Updated naive impl (single-CTA, mirroring V3 sigmoid kernel):

```cpp
// One CTA, threadIdx covers (NUM_EXPERTS, T) over outer loops.
__shared__ int active_count;
if (threadIdx.x == 0) active_count = 0;

// Zero routing_indices and active_expert_ids
for (int idx = threadIdx.x; idx < NUM_EXPERTS * T; idx += blockDim.x)
  moe_routing_indices[idx] = 0;
for (int e = threadIdx.x; e < NUM_EXPERTS; e += blockDim.x)
  moe_active_expert_ids[e] = -1;
if (threadIdx.x == 0) moe_active_expert_ids[NUM_EXPERTS] = 0;
__syncthreads();

// Each warp handles a chunk of tokens; lane k = expert slot
int const num_warps = blockDim.x / 32;
int warp = threadIdx.x / 32;
int lane = threadIdx.x % 32;
for (int t = warp; t < T; t += num_warps) {
  int tok = input_ids[t];
  if (lane < TOPK) {
    int e = tid2eid[tok * TOPK + lane];
    moe_routing_indices[e * T + t] = lane + 1;       // slot index, 1-based
    moe_active_expert_ids[e] = e;                     // mark active
  }
}
__syncthreads();

// Compaction (same as topk_sigmoid_sm100.cuh:351-361)
for (int e = threadIdx.x; e < NUM_EXPERTS; e += blockDim.x) {
  int mark = moe_active_expert_ids[e];
  if (mark >= 0) {
    int pos = atomicAdd(&moe_active_expert_ids[NUM_EXPERTS], 1);
    moe_active_expert_ids[pos] = e;
  }
}
```

## 1.10 Test-mode unit test plan

File: `tests/runtime_python/test_mode/test_hash_route_lookup_testmode.py`.

* Inputs:
  - `T = 8` tokens, `input_ids = torch.tensor([3,5,9,2,7,4,6,1], int32)`.
  - `V = 16, K = 6, E = 16` (small toy table for readability).
  - `tid2eid = torch.randint(0, E, (V, K), dtype=int32, manual_seed=0)`.
* PyTorch oracle:
  ```python
  indices = tid2eid[input_ids]          # [T, K]
  # convert to expert-major slot-1-indexed routing matrix
  routing = torch.zeros(E, T, dtype=int32)
  for t in range(T):
      for k in range(K):
          routing[indices[t, k].item(), t] = k + 1
  ```
* Compare `routing` against the MPK output via `torch.equal`.
* `moe_mask` correctness: `mask[E]` equals the unique-expert count, and
  `mask[:mask[E]]` is the (unsorted) list of active expert ids; check by
  `sorted(mpk_mask[:cnt]) == sorted({indices.flatten()})`.

Single-GPU rule: find a free GPU via `nvidia-smi --query-gpu=memory.used`,
set `CUDA_VISIBLE_DEVICES`. Activate `mirage_2` conda env.

---

# 2. `swiglu_clamped_layer` — parametric variant of `silu_mul_layer`

## 2.1 Task name + file (existing kernel **extended**)

* Kernel file (EXTENDED): `include/mirage/persistent_kernel/tasks/ampere/silu_mul.cuh`
  — add template parameter `bool WITH_CLAMP` and a `float L` runtime arg.
  (`silu_mul.cuh:19-43` shows the existing minimal kernel; the extension is
  one extra if-branch.)
* Also extend the moe variant if one exists. Note V3 uses
  `kernel::silu_mul_task_impl` for both dense (`silu_mul`) and MoE
  (`moe_silu_mul`) — register entries in `task_register.cc:361-403` for the
  dense path; the MoE variant has a similar register entry.
* Python entry: **extend**
  `silu_mul_layer(..., swiglu_limit: float | None = None)` and the MoE
  variant `moe_silu_mul_layer(..., swiglu_limit: float | None = None)` in
  `persistent_kernel.py:2339-2353, 2529-2543`. `None` (default) ⇒ legacy
  behavior; a positive float ⇒ clamp.
* Task name string (REUSED): `silu_mul` (dense) and `moe_silu_mul` (MoE).
  No new task-name registration in `graph.cc`.
* `TaskType` enum: REUSED — `TASK_SILU_MUL = 118`
  (`runtime_header.h:122`). The clamp is selected by a `params[0]` flag and
  `params[1]` carrying the float bit-pattern, like
  `register_moe_topk_sigmoid_sm100_task`'s `routed_scaling_factor` does
  (`task_register.cc:2347-2348`).

## 2.2 Math

For `i in [0, OUTPUT_SIZE)`, `t in [0, T)`:

```
gate = float(input[t, i])
up   = float(input[t, OUTPUT_SIZE + i])
if WITH_CLAMP:
    gate = min(gate, L)                  # clamp(max=L)  per model.py:602
    up   = clamp(up, -L, L)              # per model.py:601
output[t, i] = bf16( silu(gate) * up )   # silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
```

Note the **asymmetric** clamp on `gate`: only the upper bound is enforced
(`max=L`, no `min=-L`). This matches `model.py:602`:
`gate = torch.clamp(gate, max=self.swiglu_limit)`.

```python
# deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:596-606
def forward(self, x, weights=None):
    dtype = x.dtype
    gate = self.w1(x).float()
    up = self.w3(x).float()
    if self.swiglu_limit > 0:
        up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
        gate = torch.clamp(gate, max=self.swiglu_limit)
    x = F.silu(gate) * up
    if weights is not None:
        x = weights * x
    return self.w2(x.to(dtype))
```

## 2.3 Inputs / Outputs / Weights

| Role | Tensor | Shape | dtype |
|---|---|---|---|
| Input | `input` (gate \|\| up) | `[T, 2*I]` dense or `[T, K, 2*I]` MoE | bf16 |
| Output | `output` | `[T, I]` dense or `[T, K, I]` MoE | bf16 |
| Weight | none | — | — |

`I = 2048` for both routed and shared experts (Flash-Base shared expert has
`n_shared_experts=1`, intermediate `1*I=2048`).

## 2.4 Parameterization

* Template arg: `bool WITH_CLAMP`. Default `false` (V3 unaffected).
* Runtime arg: `float L`. Ignored unless `WITH_CLAMP`. Passed via
  `task_register.cc`'s `params` list:
  `params = [int(with_clamp), bit-cast<int>(L)]`. Kernel decoder is identical
  to the one in `register_moe_topk_sigmoid_sm100_task` at lines 2347-2348.
* New Python kwarg: `swiglu_limit: float | None = None` on both
  `silu_mul_layer` and `moe_silu_mul_layer`.

## 2.5 Source to migrate

Reference is in-place extension; quote the V3 kernel + the official
clamp:

```cpp
// EXISTING: include/mirage/persistent_kernel/tasks/ampere/silu_mul.cuh:24-41
template <typename T, int BATCH_SIZE, int OUTPUT_SIZE,
          int I_STRIDE, int O_STRIDE>
__device__ __forceinline__ void silu_mul_task_impl(void const *input_ptr,
                                                   void *output_ptr,
                                                   int num_active_tokens) {
  T const *__restrict__ d_input = static_cast<T const *>(input_ptr);
  T const *__restrict__ d_mul = static_cast<T const *>(input_ptr) + OUTPUT_SIZE;
  T *__restrict__ d_output = static_cast<T *>(output_ptr);
#pragma unroll
  for (int i = threadIdx.x; i < num_active_tokens * OUTPUT_SIZE;
       i += blockDim.x) {
    int batch_idx = i / OUTPUT_SIZE;
    int offset = i % OUTPUT_SIZE;
    float input_val = float(d_input[batch_idx * I_STRIDE + offset]);
    T mul_val = d_mul[batch_idx * I_STRIDE + offset];
    d_output[batch_idx * O_STRIDE + offset] =
        T(input_val / (1.0f + expf(-input_val))) * mul_val;
  }
}
```

Wave-2 extension diff:

```cpp
template <typename T, int BATCH_SIZE, int OUTPUT_SIZE,
          int I_STRIDE, int O_STRIDE, bool WITH_CLAMP>
__device__ __forceinline__ void silu_mul_task_impl(void const *input_ptr,
                                                   void *output_ptr,
                                                   int num_active_tokens,
                                                   float L /* unused if !WITH_CLAMP */) {
  // ... same as above ...
    float input_val = float(d_input[batch_idx * I_STRIDE + offset]);
    float mul_val   = float(d_mul[batch_idx * I_STRIDE + offset]);
    if (WITH_CLAMP) {
        input_val = fminf(input_val, L);                  // gate: max only
        mul_val   = fminf(fmaxf(mul_val, -L), L);         // up: full clamp
    }
    d_output[batch_idx * O_STRIDE + offset] =
        T(input_val / (1.0f + expf(-input_val))) * T(mul_val);
}
```

## 2.6 Generated CUDA reference

N/A — no TileLang version. The PyTorch reference is the snippet in §2.2.
vLLM's `SiluAndMulWithClamp`
(`deps/vllm/vllm/model_executor/layers/activation.py` — search this file if a
fused CUDA op exists, otherwise fall back to PyTorch).

## 2.7 Grid design

Unchanged from V3:

* **Natural grid**: dense — `(I/64,) = (32,)` blocks across the 2048-wide output;
  MoE — `(T, K, 1)`.
* CTA work: same compute pattern; the if-branch is element-wise, no grid
  change.
* **block_dim**: `(128, 1, 1)`.
* Alignment: 8 elements per LDG (16-byte alignment) — unchanged.

## 2.8 `/add-mpk-task` conformance checklist

- [x] **`blockIdx`-agnostic**: existing kernel reads `task_desc->input_ptrs[0]`
      and `task_desc->output_ptrs[0]` only; no `blockIdx` use.
- [x] **TaskType enum**: REUSED (`TASK_SILU_MUL = 118`,
      `TASK_MOE_SILU_MUL` — check the existing value).
- [x] **`register_<task>_task` extension**: modify
      `register_silu_mul_task` (`task_register.cc:361-403`) and
      `register_moe_silu_mul_task` to read
      `params = [with_clamp_flag, L_bits]` and switch on `with_clamp_flag`
      when emitting the template instantiation (`WITH_CLAMP=false|true`,
      `L=<float>`).
- [x] **`graph.cc` dispatch**: REUSED — same `name == "silu_mul"` / `name ==
      "moe_silu_mul"` strings; the only change is to pass the new `params`
      vector through (`graph.cc:479-483, 642-645`).

## 2.9 Initial (naive) implementation strategy

Element-wise if-branch as shown in §2.5. No restructuring; no perf impact in
`WITH_CLAMP=false` mode because the branch is template-dispatched.

## 2.10 Test-mode unit test plan

Two tests in `tests/runtime_python/test_mode/test_swiglu_clamped_testmode.py`:

* **Test A — legacy (`swiglu_limit=None`)**: identical to existing
  `test_silu_mul_testmode.py` (if present) — verify regression-free baseline
  on `T=2, I=2048`.
* **Test B — clamped (`swiglu_limit=10.0`)**: inputs hand-tuned to put both
  branches of the clamp into play: `gate = torch.linspace(-15, 15, 2048)`
  per token; `up = torch.linspace(-20, 20, 2048)`. PyTorch oracle:

  ```python
  L = 10.0
  gate = gate.clamp(max=L); up = up.clamp(min=-L, max=L)
  ref  = F.silu(gate) * up
  ```

  Compare with `torch.allclose(rtol=1e-3, atol=1e-3)` (bf16 path).

* **Both tests** for the MoE 3D variant (`moe_silu_mul`) too: shapes
  `[T, K, 2*I] -> [T, K, I]` with `K=6, I=2048`.

---

# 3. `sqrtsoftplus_topk_layer` — parametric variant of `moe_topk_sigmoid_routing_layer`

## 3.1 Task name + file (existing kernel **extended**)

* Kernel file (EXTENDED):
  `include/mirage/persistent_kernel/tasks/blackwell/topk_sigmoid_sm100.cuh`
  — add template parameter `enum class ScoreFunc { Sigmoid, SqrtSoftplus }`.
* Python entry: NEW
  `moe_sqrtsoftplus_topk_routing_layer` (the V4 callers should use this
  name; under the hood it can be the same code path as the V3 method with
  one extra kwarg). Or rename the V3 method to
  `moe_topk_routing_layer(..., score_func="sigmoid"|"sqrtsoftplus", ...)`
  — preferred to keep the API tidy. The V3 builder already takes a
  hardcoded `"sigmoid"` path; adding a new method is the additive choice.
* Task name string in `graph.cc`: REUSED `moe_topk_sigmoid_sm100` (extend the
  register function to switch on a `score_func` int in `params`).
  Alternatively add a parallel name `moe_topk_sqrtsoftplus_sm100` — but the
  underlying CUDA code is the same kernel parameterized differently. The
  spec adopts the **same name**, switched by `params[3]` (new entry, see §3.4).
* `TaskType` enum: REUSED — `TASK_MOE_TOPK_SIGMOID_SM100 = 280`
  (`runtime_header.h:176`). The variant is template-internal.

> **Group structure goes away.** V3 sets `num_groups=8, topk_group=4` in
> `deepseek_v3/builder.py:911-912` (V3 uses `topk_method="group_limited_greedy"`).
> V4 sets `topk_method="noaux_tc"` (flat top-K + e_score_correction_bias),
> i.e. `num_groups=1, topk_group=1` in our scheme.
> (`deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:564-584`,
> `deps/vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py:131-228`.)
>
> The existing kernel's `NUM_GROUPS=1, TOPK_GROUP=1` combination is already
> exercised by the `static_assert(NUM_GROUPS * EXPERTS_PER_GROUP == NUM_EXPERTS)`
> at `topk_sigmoid_sm100.cuh:146-147`. The "flat top-K" mode falls out for
> free: `EXPERTS_PER_GROUP = NUM_EXPERTS`, `THREADS_PER_GROUP = NUM_EXPERTS/VPT`
> — which constrains `THREADS_PER_GROUP <= WARP_SIZE`. With
> `NUM_EXPERTS=256, VPT=8`, `THREADS_PER_GROUP=32` which exactly fills a warp.
> **Verified ok.**

## 3.2 Math

For each token `t`:

```
scores_raw[t, e]    = logits[t, e]                                     # float32 from gate GEMM
scores[t, e]        = sqrt(softplus(scores_raw[t, e]))                  # element-wise
original_scores     = scores                                            # unbiased copy
scores_biased[t, e] = scores[t, e] + e_score_correction_bias[e]         # for selection

if hash_indices_override is not None:
    indices[t, :]   = hash_indices_override[t, :]                       # K experts from §1
else:
    indices[t, :]   = topk(scores_biased[t, :], K).indices              # flat top-K

weights[t, :]       = original_scores.gather(indices[t, :])             # K floats
if renormalize:
    weights[t, :]  /= weights[t, :].sum()
weights[t, :]      *= routed_scaling_factor
```

(All compute in float32.)

Reference:

```python
# deps/vllm/vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py:197-228
n_routed_experts = gating_output.shape[-1]
if scoring_func == "softmax":
    scores = gating_output.softmax(dim=-1)
elif scoring_func == "sigmoid":
    scores = gating_output.sigmoid()
elif scoring_func == "sqrtsoftplus":
    scores = F.softplus(gating_output).sqrt()
if e_score_correction_bias is not None:
    scores_for_choice = scores.view(-1, n_routed_experts) \
                        + e_score_correction_bias.unsqueeze(0)
else:
    scores_for_choice = scores.view(-1, n_routed_experts)
if hash_indices_table is not None:
    topk_indices = hash_indices_table[input_tokens]
else:
    topk_indices = torch.topk(scores_for_choice, k=topk, dim=-1)[1]
topk_weights = scores.gather(1, topk_indices)
if renormalize:
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
topk_weights = topk_weights.to(torch.float32)
if routed_scaling_factor != 1.0:
    topk_weights *= routed_scaling_factor
```

## 3.3 Inputs / Outputs / Weights

| Role | Tensor | Shape | dtype | Notes |
|---|---|---|---|---|
| Input | `logits` | `[T, E]` = `[T, 256]` | bf16 | gate-GEMM output (`linear_layer` writes bf16) |
| Input | `bias` | `[E]` | float32 | `e_score_correction_bias` from checkpoint (`layers.{l}.ffn.gate.e_score_correction_bias`); only for non-hash layers |
| Input (optional) | `hash_indices_override` | `[T, K]` | int32 | for hash layers; tells the kernel to skip the topk and use these indices instead |
| Output | `moe_topk_weights` | `[T, K]` = `[T, 6]` | float32 | the `K` weights, normalized + scaled |
| Output | `moe_routing_indices` | `[E, T]` | int32 | expert-major slot table, slot is 1-indexed |
| Output | `moe_mask` | `[E+1]` | int32 | compacted active-expert list |

## 3.4 Parameterization

`params` layout (extends V3's `[num_groups, topk_group, scaling_bits]`):

* `params[0] = num_groups` — for V4 set to **1**.
* `params[1] = topk_group` — for V4 set to **1**.
* `params[2] = bitcast<int>(routed_scaling_factor)` — **1.5f** = `0x3FC00000`.
* `params[3] = score_func` — **NEW**: `0 = Sigmoid`, `1 = SqrtSoftplus`,
  `2 = Softmax` (reserved; not used in V4 MoE).
* `params[4] = use_hash_override` — **NEW**: `0` (default) or `1`. Set to `1`
  when called from hash layers 0..2.
* `params[5] = renormalize` — **NEW**: `0` or `1`. V4 sets `1`
  (`norm_topk_prob=true`).

Backwards compatibility: when `params.size() == 3`, the register function
assumes `params[3]=0, params[4]=0, params[5]=1` (V3 behavior).
(`task_register.cc:2344` currently asserts `params.size() == 3`; relax to
`>= 3` and read additional entries optionally.)

## 3.5 Source to migrate

* PyTorch oracle: §3.2 above.
* CUDA reference for sqrtsoftplus + hash-override kernel:
  `deps/vllm/csrc/moe/topk_softplus_sqrt_kernels.cu:96-260` (the launch
  helper at line 467 plus the inner kernel at line 100 plus the hash branch
  at line 240). Quote (the scoring branch):

```cpp
// deps/vllm/csrc/moe/topk_softplus_sqrt_kernels.cu (sqrtsoftplus phase)
//   float logit = converter(row_chunk_temp[ii]);
//   float sp    = log1pf(expf(logit));        // softplus(x)
//   float score = sqrtf(sp);                  // sqrt(softplus(x))
//   row_chunk[ii] = score;
//   biased_chunk[ii] = score + bias[bias_offset + ii];
```

`softplus(x) = log(1 + exp(x))`. For numerical stability use
`log1pf(expf(x))` (matches vLLM's CUDA kernel). For very negative `x`,
`softplus(x) ≈ exp(x)`, `sqrt ≈ exp(x/2)` — both well within float32 range
given typical logit magnitudes; no special-case branch required.

## 3.6 Generated CUDA reference

N/A — kernel is in CUDA, not TileLang.

## 3.7 Grid design

Unchanged from V3 (`moe_topk_sigmoid_routing_layer`):

* **Natural grid**: `(1, 1, 1)` — single CTA processes all `T` tokens
  internally (rows-per-warp × warps-per-CTA covers up to 8 rows; the kernel
  iterates if `T > rows_per_CTA`). `block_dim = (256, 1, 1)` (8 warps).
* **CTA work**: each warp handles up to 1 row (since
  `ELTS_PER_WARP=32*8=256=E=ELTS_PER_ROW`, `ROWS_PER_WARP=1`). With 8 warps,
  one CTA does 8 rows; the kernel must loop over `T/8` row-groups.
  Verified: the existing kernel does *not* loop over `T` (it assumes
  `T ≤ ROWS_PER_WARP * WARPS = 8`). For V4 with `max_num_batched_tokens > 8`
  the **kernel must be extended** to loop over row-groups; this is part of
  the Wave-2 task and is **a bug already latent in V3** if MPK runs with
  large prefill batches.
  > **OPEN**: confirm whether V3 already handles `T > 8` by some other means,
  > or if a row-group outer loop must be added now.
* **`blockIdx`-agnostic**: existing kernel does not use `blockIdx`.

## 3.8 `/add-mpk-task` conformance checklist

- [x] **`blockIdx`-agnostic**: yes (existing).
- [x] **TaskType enum**: REUSED (`TASK_MOE_TOPK_SIGMOID_SM100 = 280`).
- [x] **Register function extension**: extend
      `register_moe_topk_sigmoid_sm100_task` to:
      1. accept `params.size() >= 3`,
      2. read optional `score_func`/`use_hash_override`/`renormalize`,
      3. emit the template instantiation with the new template args.
- [x] **Dispatch in graph.cc**: REUSED — same name string. No edit needed
      to `graph.cc:632-640` beyond pass-through of the longer `params` vec
      (already passed verbatim).
- [x] **Python method**: NEW
      `moe_sqrtsoftplus_topk_routing_layer(..., score_func="sqrtsoftplus",
      renormalize=True, hash_indices_override=None, ...)`. If
      `hash_indices_override is not None`, append it to the kernel's input
      list (becomes `input[2]`) and set `params[4]=1`.

## 3.9 Initial (naive) implementation strategy

* **Within `topk_sigmoid_task_impl`**, after `float logit = converter(...)` at
  `topk_sigmoid_sm100.cuh:196`, branch on the template arg:

  ```cpp
  if constexpr (SCORE_FUNC == ScoreFunc::Sigmoid) {
      sig = 1.0f / (1.0f + expf(-logit));
  } else if constexpr (SCORE_FUNC == ScoreFunc::SqrtSoftplus) {
      sig = sqrtf(log1pf(expf(logit)));
  }
  ```

* **Drop the group-top-2 reduction (Phase 2)** when `NUM_GROUPS == 1` —
  guard with `if constexpr (NUM_GROUPS > 1) { ... }`. With `NUM_GROUPS=1`,
  every expert is in the (single) group; the group-selection logic is a
  no-op and would actually break the per-thread reduction since
  `THREADS_PER_GROUP == THREADS_PER_ROW`.

* **Hash-override branch (Phase 5)**: if `USE_HASH_OVERRIDE == 1`, replace
  the iterative argmax with a direct read from `hash_indices_override` and
  the score-gather via `__shfl_sync`. This is exactly the pattern from
  `topk_softplus_sqrt_kernels.cu:240-260`.

* **Renormalize**: existing kernel already does `inv_sum *
  routed_scaling_factor` at line 344. Wrap that with
  `if (RENORMALIZE) inv_sum = 1.0f/(weight_sum+eps); else inv_sum = 1.0f;`.

## 3.10 Test-mode unit test plan

File: `tests/runtime_python/test_mode/test_sqrtsoftplus_topk_testmode.py`.

* **Test A — sigmoid regression** (`score_func="sigmoid"`,
  `num_groups=8, topk_group=4`, `T=2, E=256, K=8`): mirror existing
  `test_moe_topk_sigmoid_testmode.py`. Catches regression.
* **Test B — sqrtsoftplus flat top-K** (`score_func="sqrtsoftplus"`,
  `num_groups=1, topk_group=1`, `T=4, E=256, K=6`, random logits ∈ N(0, 1),
  random bias ∈ U(-0.1, 0.1), `renormalize=True`,
  `routed_scaling_factor=1.5`). PyTorch oracle = §3.2 snippet. Compare:
  - `mpk_topk_weights[t, :].sort().values == ref.sort().values` (set
    equality on float; the kernel may select tied experts in arbitrary
    order)
  - `set(mpk_indices[t, :].tolist()) == set(ref_indices[t, :].tolist())`
  - `mpk_mask[E] == len(set(ref_indices.flatten()))`
* **Test C — hash override**: feed
  `hash_indices_override = torch.randint(0, E, (T, K), int32)`, set
  `use_hash_override=1`. Oracle uses the same indices and gathers
  `original_scores` at them.

All three with bf16 logits → fp32 internal → bf16-tolerance compare.

---

# 4. V4 MoE forward orchestration (Python composition)

This is the per-layer composition that the V4 builder
(`python/mirage/mpk/models/deepseek_v4/builder.py`, **NEW**, Phase B of the
plan) emits. Pseudocode mirrors `deepseek_v3/builder.py:832-1153` with the
hash/sqrtsoftplus branches added.

```python
def _build_moe_v4(self, layer_idx, state_dict, x_in, residual):
    prefix = f"model.layers.{layer_idx}.ffn."
    T = self.max_num_batched_tokens
    E = 256; K = 6; D = self.hidden_size; I = self.moe_intermediate_size
    is_hash = (layer_idx < self.num_hash_layers)  # 3

    # 1) Pre-MoE RMSNorm (the same self.rmsnorm_out used by V3)
    self.mpk.rmsnorm_layer(input=x_in, weight=state_dict[f"...post_attn_norm"],
                           output=self.rmsnorm_out, ...)

    # 2) Allocate routing tensors (same layout as V3)
    moe_topk_weights    = self.mpk.new_tensor((T, K), dtype=float32, ...)
    moe_routing_indices = self.mpk.new_tensor((E, T), dtype=int32, ...)
    moe_mask            = self.mpk.new_tensor((E + 1,), dtype=int32, ...)

    if is_hash:
        # Layers 0..2: build hash routing table once at init from
        # state_dict[f"layers.{layer_idx}.ffn.gate.tid2eid"] (verified
        # present in model.safetensors.index.json:1570,3135,4711).
        w_tid2eid = self.mpk.attach_input(
            state_dict[f"{prefix}gate.tid2eid"],
            name=f"layer_{layer_idx}_tid2eid",
        )
        # 2.a Hash lookup writes (indices, mask)
        hash_indices_override = self.mpk.new_tensor(
            (T, K), dtype=int32, name=f"layer_{layer_idx}_hash_indices",
        )
        self.mpk.hash_route_lookup_layer(
            input_ids=self.meta_tokens,    # int32 [T]
            tid2eid=w_tid2eid,
            output_indices_TK=hash_indices_override,
            moe_routing_indices=moe_routing_indices,
            moe_mask=moe_mask,
            grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
        )
    else:
        hash_indices_override = None

    # 3) Gate GEMM (always — even on hash layers we still need scores)
    w_gate = self.mpk.attach_input(state_dict[f"{prefix}gate.weight"],
                                    name=f"layer_{layer_idx}_moe_gate")
    router_logits = self.mpk.new_tensor((T, E), dtype=bfloat16, ...)
    self.mpk.linear_layer(self.rmsnorm_out, w_gate, router_logits,
                          grid_dim=..., block_dim=(128, 1, 1))

    # 4) Top-K (sqrtsoftplus for non-hash; sqrtsoftplus + override for hash)
    w_bias = self.mpk.attach_input(
        state_dict[f"{prefix}gate.e_score_correction_bias"]
            if not is_hash else self._zero_bias_E,   # zeros if hash (no bias param exists)
        name=f"layer_{layer_idx}_moe_bias",
    )
    self.mpk.moe_sqrtsoftplus_topk_routing_layer(
        input=router_logits, bias=w_bias,
        hash_indices_override=hash_indices_override,    # None if not hash
        output=(moe_topk_weights, moe_routing_indices, moe_mask),
        num_groups=1, topk_group=1,
        routed_scaling_factor=self.routed_scaling_factor,   # 1.5
        renormalize=True,
        score_func="sqrtsoftplus",
        grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
    )

    # 5) Per-token group FP8 quant of MoE input (float32 scale, NOT UE8M0)
    moe_input_fp8   = self.mpk.new_tensor((T, D), dtype=float8_e4m3, ...)
    moe_input_scale = self.mpk.new_tensor((T, D // 128), dtype=float32, ...)
    self.mpk.quantize_fp8_layer(
        input=self.rmsnorm_out,
        output_fp8=moe_input_fp8, output_scale=moe_input_scale,
        grid_dim=(T, 1, 1), block_dim=(128, 1, 1),
        scale_ue8m0=False,     # MoE group GEMM wants float32 scale
    )

    # 6) W13 group GEMM (FP8) -> [T, K, 2*I]
    w13, w13_scale = self._attach_moe_fp8_weight(prefix + "experts.w13", ...)
    moe_mid = self.mpk.new_tensor((T, K, 2 * I), dtype=bfloat16, ...)
    self.mpk.moe_w13_fp8_layer(
        input_fp8=moe_input_fp8, input_scale=moe_input_scale,
        weight_fp8=w13, weight_scale=w13_scale,
        moe_routing_indices=moe_routing_indices, moe_mask=moe_mask,
        output=moe_mid,
        grid_dim=(E, _moe_fp8_m_split(2*I, preferred=2), 1),
        block_dim=(128, 1, 1),
    )

    # 7) Clamped SwiGLU (NEW: swiglu_limit=10.0)
    moe_act = self.mpk.new_tensor((T, K, I), dtype=bfloat16, ...)
    self.mpk.moe_silu_mul_layer(
        input=moe_mid, output=moe_act,
        swiglu_limit=self.swiglu_limit,                # 10.0
        grid_dim=(T, K, 1), block_dim=(128, 1, 1),
    )

    # 8) Per-token group FP8 quant of activation (float32 scale)
    moe_act_fp8   = self.mpk.new_tensor((T, K, I), dtype=float8_e4m3, ...)
    moe_act_scale = self.mpk.new_tensor((T, K, I // 128), dtype=float32, ...)
    self.mpk.quantize_fp8_layer(
        input=moe_act,
        output_fp8=moe_act_fp8, output_scale=moe_act_scale,
        grid_dim=(T * K, 1, 1), block_dim=(128, 1, 1),
        scale_ue8m0=False,
    )

    # 9) W2 group GEMM (FP8) -> [T, K, D]
    w2, w2_scale = self._attach_moe_fp8_weight(prefix + "experts.w2", ...)
    moe_down_out = self.mpk.new_tensor((T, K, D), dtype=bfloat16, ...)
    self.mpk.moe_w2_fp8_layer(
        input_fp8=moe_act_fp8, input_scale=moe_act_scale,
        weight_fp8=w2, weight_scale=w2_scale,
        moe_routing_indices=moe_routing_indices, moe_mask=moe_mask,
        output=moe_down_out,
        grid_dim=(E, _moe_fp8_m_split(D, preferred=2), 1),
        block_dim=(128, 1, 1),
    )

    # 10) Shared expert (1 expert, clamped SwiGLU, FP8 dense)
    shared_residual = self._build_shared_expert_v4(  # see §5
        prefix + "shared_experts.", state_dict, layer_idx, residual)

    # 11) Final combine: out = sum_k(weight_k * routed_k) + shared_residual
    moe_output = self.mpk.new_tensor((T, D), dtype=bfloat16, ...)
    self.mpk.moe_mul_sum_add_layer(
        input=moe_down_out,
        weight=moe_topk_weights,
        residual=shared_residual,
        output=moe_output,
        grid_dim=(T, 1, 1), block_dim=(128, 1, 1),
    )
    return moe_output
```

Pattern is **byte-identical to V3** except:

| Step | V3 path | V4 path |
|---|---|---|
| 2.a | n/a | `hash_route_lookup_layer` for layers 0..2 |
| 4 | `moe_topk_sigmoid_routing_layer(num_groups=8, topk_group=4, scaling=2.5)` | `moe_sqrtsoftplus_topk_routing_layer(num_groups=1, topk_group=1, scaling=1.5, score_func=sqrtsoftplus, renormalize=True, hash_indices_override=…)` |
| 7 | `moe_silu_mul_layer()` (no clamp) | `moe_silu_mul_layer(swiglu_limit=10.0)` |
| 8/9 | unchanged | unchanged |
| 10 | `silu_mul_layer()` inside shared expert | `silu_mul_layer(swiglu_limit=10.0)` (see §5) |
| 11 | unchanged | unchanged |

---

# 5. Shared expert details (§4 step 10)

The shared expert is a **single** dense MLP run on every token. Its
contribution is added to the residual *before* the per-expert weighted sum,
the same way V3 wires it.

Reference: `MoE.shared_experts = Expert(args.dim, args.moe_inter_dim)`
(`model.py:626-628`). `Expert.__init__` uses the *default*
`swiglu_limit=0`, so the **shared expert is unclamped**:

> "no swiglu_limit" — `model.py:627`.

That means the shared expert calls `silu_mul_layer(swiglu_limit=None)`
(i.e. legacy V3 path), while the routed experts call
`moe_silu_mul_layer(swiglu_limit=10.0)`. **Do not** uniformly apply the clamp.

Shapes (`D=4096, I=2048`):

| Tensor | Shape | dtype | Checkpoint key |
|---|---|---|---|
| `gate_proj.weight` (w1) | `[I, D] = [2048, 4096]` | fp8 (with `weight_scale_inv` `[I/128, D/128] = [16, 32]`) | `layers.{l}.ffn.shared_experts.gate_proj.{weight,weight_scale_inv}` |
| `up_proj.weight` (w3) | `[I, D]` | fp8 + scale | `layers.{l}.ffn.shared_experts.up_proj.{weight,weight_scale_inv}` |
| `down_proj.weight` (w2) | `[D, I] = [4096, 2048]` | fp8 + scale | `layers.{l}.ffn.shared_experts.down_proj.{weight,weight_scale_inv}` |

Composition (matches V3 `deepseek_v3/builder.py:1064-1153`):

```python
def _build_shared_expert_v4(self, prefix, sd, layer_idx, residual):
    # gate_up = fp8_linear(rmsnorm_out, fused_gate_up_weight)
    # shared_mid: (T, 2*I)
    self._fp8_linear(self.rmsnorm_out, w_gate_up, s_gate_up, shared_mid, ...)
    # silu_mul WITHOUT clamp (shared expert has no swiglu_limit)
    self.mpk.silu_mul_layer(shared_mid, shared_silu_out,
                            swiglu_limit=None, grid_dim=..., block_dim=...)
    # down_proj
    self._fp8_linear(shared_silu_out, w_down, s_down, shared_residual, ...)
    return shared_residual   # consumed by moe_mul_sum_add_layer as `residual` arg
```

> The `residual` arg of `moe_mul_sum_add_layer` is **`shared_expert_out`** —
> the attention-residual was already folded into `shared_residual` (or into
> the input `x_in` via the `moe_mul_sum_add_layer`'s second residual). The V3
> builder uses the *shared expert output alone* as the residual
> (`deepseek_v3/builder.py:1146-1153`), because the *previous* attention
> sub-block already wrote the post-attn residual into the layer input. V4
> matches.

---

# 6. Hash table source (`tid2eid`)

Verified:

* The `tid2eid` table is **shipped in the Flash-Base checkpoint** as
  `layers.{0,1,2}.ffn.gate.tid2eid`. (See
  `deps/deepseek_v4/DeepSeek-V4-Flash/model.safetensors.index.json:1570, 3135, 4711`.)
* dtype = `int32`, shape = `[vocab_size, num_experts_per_tok] = [129280, 6]`
  (per `Gate.__init__` at `model.py:559`).
* Loaded by the converter into MPK as a kernel-input tensor; no transformation.

In contrast, vLLM at init time sets `tid2eid` to `torch.randint(...)`
(`deepseek_v4.py:761-769`), but the weight loader then *overwrites* it from
the checkpoint (because `is_hash_moe` allocates a `nn.Parameter` which the
HF loader picks up by parameter name). MPK V4 should `attach_input` the
checkpoint tensor directly — no `randint` needed.

**Not deterministic**: it is **not** a deterministic hash of `token_id`. The
table is a learned permutation/assignment baked into the checkpoint.
(Confirmed by inspecting the checkpoint key listing — values are present and
are not equivalent to any closed-form `token_id mod n_routed_experts` rule.)

---

# 7. Module-level test plan

File: `tests/runtime_python/test_mode/test_moe_v4_module_testmode.py`.

This test instantiates the **official PyTorch `MoE` module** at a single
layer index, loads real shards from
`/raid/catalyst/models/DeepSeek-V4-Flash-Base/`, runs both PyTorch and MPK,
and compares.

Test matrix:

| Test | Layer | Path covered |
|---|---|---|
| `test_moe_layer0_hash` | layer 0 | hash routing + clamped SwiGLU + shared expert |
| `test_moe_layer3_scored` | layer 3 | sqrtsoftplus + e_score_correction_bias + clamped SwiGLU + shared expert |

For each:

```python
ref = MoE(layer_id=L, args=cfg).cuda().bfloat16()
ref.load_state_dict({k.removeprefix("layers.0.ffn."): v
                     for k, v in sd_layer.items()})

mpk = make_mpk_for_moe_v4(layer_idx=L, cfg=cfg)
mpk_out = mpk(x_bf16, input_ids)        # via test_mode
ref_out = ref(x_bf16, input_ids)
torch.testing.assert_close(mpk_out, ref_out, rtol=2e-3, atol=2e-3)
```

The bf16 tolerance is per-plan §"Wave-2 gate".

**Per `CLAUDE.md` local rule 1**, the test must auto-select a free GPU and
pin to it with `CUDA_VISIBLE_DEVICES`.

---

# 8. Wiring summary (purely additive)

Files to **add**:

* `include/mirage/persistent_kernel/tasks/blackwell/hash_route_lookup_sm100.cuh` (NEW)
* `tests/runtime_python/test_mode/test_hash_route_lookup_testmode.py` (NEW)
* `tests/runtime_python/test_mode/test_swiglu_clamped_testmode.py` (NEW)
* `tests/runtime_python/test_mode/test_sqrtsoftplus_topk_testmode.py` (NEW)
* `tests/runtime_python/test_mode/test_moe_v4_module_testmode.py` (NEW)

Files to **modify additively**:

* `include/mirage/persistent_kernel/runtime_header.h`
  — add `TASK_HASH_ROUTE_LOOKUP_SM100 = 296`
  (next free slot before `TASK_SM100_TASK_END = 298`).
* `include/mirage/persistent_kernel/tasks/blackwell/task_header.cuh`
  — `#include "hash_route_lookup_sm100.cuh"`.
* `include/mirage/persistent_kernel/tasks/ampere/silu_mul.cuh`
  — add `bool WITH_CLAMP` template parameter and the clamp branch.
  Also extend the (existing) MoE variant `moe_silu_mul`.
* `include/mirage/persistent_kernel/tasks/blackwell/topk_sigmoid_sm100.cuh`
  — add `ScoreFunc` template parameter (`Sigmoid` | `SqrtSoftplus`),
  `bool USE_HASH_OVERRIDE`, `bool RENORMALIZE` template parameters, and
  the `softplus + sqrt` branch + the hash-override branch.
  Guard the group-top-2 reduction with `if constexpr (NUM_GROUPS > 1)`.
* `src/kernel/task_register.cc`
  — NEW `register_hash_route_lookup_sm100_task`.
  — extend `register_silu_mul_task` (and the `moe_silu_mul` variant) to
    decode `params = [with_clamp, L_bits]` and emit `WITH_CLAMP=…, L=…`.
  — extend `register_moe_topk_sigmoid_sm100_task` to decode
    `params = [num_groups, topk_group, scaling_bits, score_func,
    use_hash_override, renormalize]` (`>= 3` for backwards compat) and
    emit the matching template instantiation. Add the optional
    `hash_indices_override` as `input_ptrs[2]`.
* `src/kernel/graph.cc`
  — NEW `else if (name == "hash_route_lookup_sm100") { … }` clause
    (modeled on `graph.cc:632-642`).
  — no edit needed for `silu_mul` / `moe_silu_mul` / `moe_topk_sigmoid_sm100`
    dispatch (params pass-through is generic).
* `python/mirage/mpk/persistent_kernel.py`
  — NEW `hash_route_lookup_layer` method.
  — extend `silu_mul_layer` / `moe_silu_mul_layer` with
    `swiglu_limit: float | None = None`.
  — NEW `moe_sqrtsoftplus_topk_routing_layer` method (or extend
    `moe_topk_sigmoid_routing_layer` with `score_func` /
    `hash_indices_override` / `renormalize` kwargs).
* `python/mirage/mpk/models/__init__.py`
  — register `deepseek_v4` builder (when Wave-3 builder lands; out of
    scope for this Wave-1 spec).

**Recompile** after `src/` changes per `CLAUDE.md`:
`pip install -e . -v --no-deps`.

Then run the Qwen3 sanity per local rule 4:

```bash
python demo/qwen3/demo.py --output-dir ./output/sanity_v4_moe \
    --use-mirage --model /raid/catalyst/models/Qwen3-8B/ \
    --max-num-batched-requests 1
```

---

# 9. Open questions

* **OPEN (resolved in §1.2):** in hash layers, are routing weights uniform
  (`1/K * route_scale`) or score-gather of `sqrt(softplus(logits))`?
  *Resolved*: vLLM CUDA kernel + checkpoint shipping
  `gate.weight` for layers 0..2 both indicate **score-gather**. Adopt
  option (a): keep the gate GEMM + `sqrtsoftplus_topk` with
  `hash_indices_override`.

* **OPEN (resolved in §6):** is `tid2eid` deterministic
  (e.g. `token_id mod E`) or shipped in the checkpoint?
  *Resolved*: shipped in the checkpoint as `layers.{0,1,2}.ffn.gate.tid2eid`
  (`model.safetensors.index.json:1570,3135,4711`). Loaded as-is.

* **OPEN:** is the existing `topk_sigmoid_sm100.cuh` kernel correct for
  `T > 8` (its compile-time `ROWS_PER_WARP=1, WARPS_PER_CTA=8` only covers
  8 rows)? If V3 has been passing tests, it must be running with `T ≤ 8`
  per kernel invocation, or the runtime is launching multiple CTAs each
  doing a row-subset. **Action**: instrument and confirm during Wave-2.
  This affects both V3 and V4 — if it's a latent V3 bug, fix it in the
  V4 kernel extension and backport.

* **OPEN:** the V4 `Block` orchestration adds two HC `mhc_post` calls per
  layer (one after attention, one after MoE). The `residual` argument of
  `moe_mul_sum_add_layer` therefore lives in *HC space* (`[T, hc, D]`),
  not flat `[T, D]`. This spec assumes the MoE output is reduced into the
  *flat* hidden state and `mhc_post` takes care of the HC reshuffle. Cross-
  check with `docs/mpk/deepseek_v4/hc.md` (sibling Wave-1 spec).

* **OPEN:** for hash layers, the `e_score_correction_bias` *does not exist*
  in the checkpoint (`Gate.__init__` at `model.py:560-562` skips it when
  `self.hash` is true). The `moe_sqrtsoftplus_topk_routing_layer` call in
  §4 step 4 currently passes `self._zero_bias_E` — confirm this is
  semantically equivalent to `bias=None` in the kernel, or add an
  `input[1] = nullptr`-allowed path in `register_moe_topk_sigmoid_sm100_task`.

* **OPEN:** when `T < max_num_batched_tokens` (partial batch), the routing
  matrix `[E, T]` has columns that are uninitialized beyond the live
  prefix. The V3 kernel does per-token zeroing via its own Phase 0
  (`topk_sigmoid_sm100.cuh:104-114`). Verify the V4 hash kernel does the
  same; **adopted in §1.9 already** (`for (int idx = … < E * T …)` covers
  all columns).

* **OPEN:** the V4 spec for `compress_ratios` says layer 43 (MTP) has
  `compress_ratio = 0`. Confirm the MTP block uses its own FFN (not MoE)
  via §4 of `mtp.md`. This affects whether MoE-builder runs for
  `layer_idx = 43`. Per current reading of `model.py:817+`, the answer is
  no: the MTP block uses its own dedicated FFN (single MLP, no routing).
