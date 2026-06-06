# topk_softplus_sqrt

## Identity
- Source file: `csrc/moe/topk_softplus_sqrt_kernels.cu:85-426` (CUDA `__global__` template `topkGatingSoftplusSqrt<VPT, NUM_EXPERTS, WARPS_PER_CTA, BYTES_PER_LDG, WARP_SIZE_PARAM, USE_HASH, IndType, InputType>`).
- Launcher helper: `topkGatingSoftplusSqrtLauncherHelper` at lines 458-502 (template; dispatches `USE_HASH` via `DISPATCH_HASH` macro at lines 446-456 — branches on the runtime `use_hash` flag and instantiates **both** `USE_HASH=true` and `USE_HASH=false` template specializations).
- Top-level launcher: `topkGatingSoftplusSqrtKernelLauncher` at lines 536-624 — switch on `num_experts` (must be one of 1/2/4/8/16/32/64/128/256/512, or multiples of 64: 192/320/384/448/576).
- Host entry: `topk_softplus_sqrt` at lines 690-727 (dtype dispatch on `gating_output` for {float32, fp16, bf16}, scalar-type dispatch on `topk_indices` for {int32, uint32, int64}).
- Op registration: `csrc/moe/torch_bindings.cpp:19-24` registers `torch.ops._moe_C.topk_softplus_sqrt` with schema `(Tensor! topk_weights, Tensor! topk_indices, Tensor! token_expert_indices, Tensor gating_output, bool renormalize, float routed_scaling_factor, Tensor? bias, Tensor? input_ids, Tensor? tid2eid) -> ()`.
- Python wrapper: `vllm/_custom_ops.py:2476-2497` (`topk_hash_softplus_sqrt(...)` — name disambiguates from `topk_softmax`/`topk_sigmoid`). Forwards to `torch.ops._moe_C.topk_softplus_sqrt`.
- Language/DSL: **CUDA C++** (hand-written warp-cooperative kernel). Adapted from TensorRT-LLM's MoE topk softmax kernel (file header lines 1-19).
- Third-party dep: none.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py:132` | `vllm_topk_softplus_sqrt` (within `fused_topk_bias` at line 217 of same file) | `gating_output: [M, E]`, `topk_weights: [M, k]`, `topk_indices: [M, k]`, `token_expert_indices: [M, k]`; optional `input_tokens: [M]`, `hash_indices_table: [vocab, k]` | `gating_output`: fp32 / bf16 / fp16; `topk_weights`: fp32; `topk_indices`: int32 / uint32 / int64; `e_score_correction_bias`: fp32; `input_tokens`/`hash_indices_table`: matches `topk_indices` dtype | `scoring_func == "sqrtsoftplus"` (V4-Flash default per `nvidia/model.py:425`); not XPU (line 119); not ROCm-AITER (line 172). |
| `vllm/models/deepseek_v4/nvidia/model.py:556` | `DeepseekV4MoE.forward` → `fused_topk_bias(...)` (mega-MoE path) | M = num tokens, E = 256, k = 6 (V4-Flash) | bf16/fp32 logits, fp32 weights, int32 indices | `use_mega_moe` True OR `_forward_fused_moe` line 598's `self.experts(router_logits=...)` |

V4-Flash dispatches both branches by layer index. `nvidia/model.py:447`: `is_hash_moe = extract_layer_index(prefix) < config.num_hash_layers` → layers `0..num_hash_layers-1` (V4-Flash: 0..2, i.e. 3 layers) populate `self.gate.tid2eid = nn.Parameter(torch.randint(0, n_routed_experts, (vocab, k), dtype=...))` and pass it as `hash_indices_table` (line 567). Layers `num_hash_layers..n_layers-1` (V4-Flash: 3..42, i.e. 40 layers) leave `tid2eid=None` and pass an `e_score_correction_bias` (line 463-466). The kernel reads `tid2eid` is-set ↔ `USE_HASH=true` at the host dispatcher line 643-646.

## Inputs

| Name | Shape | Dtype | Layout | Hash branch | Scored branch | Meaning |
| --- | --- | --- | --- | --- | --- | --- |
| `gating_output` (`input`) | `[M, E]` (V4-Flash: `[M, 256]`) | fp32 / bf16 / fp16 | row-major, contiguous on E | **read** (still scored for the weights even though indices are hash-driven) | **read** (this is the score tensor) | `gate(hidden_states)` output from `GateLinear.forward()` — see `dsv3_router_gemm.md` |
| `topk_weights` (`output`) | `[M, k]` (V4-Flash: `[M, 6]`) | fp32 | row-major, contiguous on k | **write** | **write** | Output: scaled scores for selected experts |
| `topk_indices` (`indices`) | `[M, k]` (V4-Flash: `[M, 6]`) | int32 (V4-Flash `nvidia/model.py:448`) / uint32 / int64 | row-major, contiguous on k | **write** (copy from `tid2eid[token_id]`) | **write** (top-k argmax of biased scores) | Output: expert id per (token, slot) |
| `token_expert_indices` (`source_rows`) | `[M, k]` | int32 | row-major | **not written** (kernel returns before this write on hash branch, lines 297-300) | **write**: `source_rows[idx] = k_idx * M + thread_row` (line 385) | Used by `FusedMoE` permutation when scattering tokens to experts |
| `renormalize` | scalar bool | — | — | controls `selected_sum` warp-reduce + scale denom (lines 268-279) | controls `selected_sum` accumulation + scale denom (lines 386-388, 411-416) | `topk_norm_prob = True` for V4-Flash (`config.norm_topk_prob`) |
| `routed_scaling_factor` | scalar float (passed as double) | — | — | applied (line 275, 292) | applied (line 412, 419) | V4-Flash: `route_scale = 1.5` (`deepseek_v4/DeepSeek-V4-Flash/inference/model.py:583`, `inference/config.json:12`) |
| `correction_bias` (`e_score_correction_bias`) | `[E]` (V4-Flash: `[256]`) or null | fp32 | contiguous | **null** (hash layers have `self.bias = None` at `inference/model.py:560`; vLLM mirrors this at `nvidia/model.py:445-447`) | **non-null** (`bias` param at `inference/model.py:562`) | Added to scores **after** `sqrt(softplus(.))` for expert SELECTION; the SAME bias is subtracted from `max_val` before the output write so the *weight* is the un-biased score (lines 379-381) |
| `input_ids` (`input_tokens`) | `[M]` | matches `topk_indices` dtype | contiguous | **non-null** (= `input_ids` flattened — see `nvidia/model.py:566`) | **null** (passed but unused) | Per-token vocab id used as index into `tid2eid` (line 238) |
| `tid2eid` (`hash_indices_table`) | `[vocab_size, k]` (V4-Flash: `[129280, 6]`) | matches `topk_indices` dtype | row-major | **non-null** (presence triggers `USE_HASH=true` at host dispatcher line 643-646) | **null** | Precomputed `token_id → [expert_0, ..., expert_{k-1}]` lookup table (`Gate.tid2eid` parameter from `inference/model.py:559`) |
| `start_expert`, `end_expert` | scalar int | — | — | (params passed but the hash branch does not gate experts) | `[start_expert, end_expert)` is the local expert range; experts outside this range get index `NUM_EXPERTS` (line 384) | Always `[0, num_experts)` at the caller (`csrc/moe/topk_softplus_sqrt_kernels.cu:511`) |
| `finished` | `[M]` bool or null | bool | — | (param plumbed but always `nullptr` from the caller line 510) | same | Early-exit mask; V4-Flash passes nullptr |

`NUM_EXPERTS`, `WARPS_PER_CTA = 4`, `BYTES_PER_LDG`, `WARP_SIZE_PARAM = 32` (CUDA), `USE_HASH`, `IndType`, `InputType` are template constants. `VPT` (values per thread) is derived via `detail::TopkConstants` (lines 433-443): `VPT = max(1, E / (ELTS_PER_LDG * WARP_SIZE))`, `THREADS_PER_ROW = E / VPT`, `ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW`.

For V4-Flash on CUDA (`E=256`, bf16 logits, `WARP_SIZE=32`): `BYTES_PER_LDG = MIN(BYTES_PER_LDG_POWER_OF_2=16, sizeof(bf16)*256=512) = 16` → `ELTS_PER_LDG = 8`, `VPT = max(1, 256/(8*32)) = 1` ... wait — actually `MAX(1, 256/(8*32)) = MAX(1, 1) = 1` then `VPT = VECs_PER_THREAD * ELTS_PER_LDG = 1 * 8 = 8`, `THREADS_PER_ROW = 256 / 8 = 32`, `ROWS_PER_WARP = 32 / 32 = 1`, `ROWS_PER_CTA = 4 * 1 = 4`. Grid = `ceil(M / 4)` CTAs.

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `topk_weights` | `[M, k]` (V4-Flash: `[M, 6]`) | fp32 | row-major, contiguous | Final routing weights. **Hash branch**: gathered `row_chunk[ii]` × scale (line 292). **Scored branch**: `max_val - correction_bias[expert]` then × scale (lines 380-382, 419). |
| `topk_indices` | `[M, k]` (V4-Flash: `[M, 6]`) | int32 (V4-Flash) | row-major, contiguous | Expert ids. **Hash branch**: copied directly from `tid2eid[token_id, k_idx]` (line 259) — these are NOT offset-adjusted by `start_expert` because the hash branch does not call `node_uses_expert`. **Scored branch**: `(expert - start_expert)` if expert is in this node's local range, else `NUM_EXPERTS` (sentinel; line 384). |
| `token_expert_indices` | `[M, k]` | int32 | row-major, contiguous | Scored branch only: `k_idx * num_rows + thread_row` (line 385). Used by FusedMoE to scatter tokens to expert queues. **Hash branch returns BEFORE writing this** (line 300 `return`). |

## Grid / Block

- `grid_dim.x = ceil(M / ROWS_PER_CTA)` where `ROWS_PER_CTA = WARPS_PER_TB * ROWS_PER_WARP`. For V4-Flash (E=256, bf16): `ROWS_PER_CTA = 4`, so `grid.x = ceil(M / 4)`.
- `block_dim = (WARP_SIZE_PARAM, WARPS_PER_TB) = (32, 4) = 128 threads/CTA` (line 475).
- `__launch_bounds__(WARPS_PER_CTA * WARP_SIZE_PARAM) = __launch_bounds__(128)` (line 85).
- **Per-thread work**: one thread covers `VPT` experts of one row. `THREADS_PER_ROW` cooperating threads cover all `NUM_EXPERTS` for one (token) row, communicating via butterfly `__shfl_xor` (lines 270-273, 353-366).
- **Vector loads** (lines 176-232): three dtype branches.
  - fp32: `AlignedArray<float, ELTS_PER_LDG>` direct vector load.
  - bf16: `AlignedArray<__nv_bfloat16, ELTS_PER_LDG>` load + `__bfloat1622float2` element-pair conversion to fp32 (lines 196-200). When `ELTS_PER_LDG == 1` falls back to scalar `__bfloat162float`.
  - fp16: parallel to bf16 with `__half22float2` (lines 218-224).
- **Per-CTA structure** (lines 141-154): `cta_base_row = blockIdx.x * ROWS_PER_CTA`; `warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP`; `thread_row_in_warp = threadIdx.x / THREADS_PER_ROW`; `thread_row = warp_base_row + thread_row_in_warp`. Out-of-bounds rows (`thread_row >= num_rows`) early-exit at line 153.
- **PDL**: `griddepcontrol.wait` at line 171, `griddepcontrol.launch_dependents` at lines 298 (hash branch) and 423 (scored branch). The kernel is launched with `cudaLaunchAttributeProgrammaticStreamSerialization` (line 487-489).
- **Autotune**: NONE. Block geometry is determined fully by `NUM_EXPERTS` and dtype via the constexpr `TopkConstants`. The launcher hard-codes `WARPS_PER_TB = 4` and `MAX_BYTES_PER_LDG = 16` (power-of-2 E) or `4` (bf16/fp16 with E multiple-of-64) (lines 543-565).

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:565-584` (`Gate.forward`). Both branches are fused into the same kernel; the V4-Flash dispatch chooses branch based on `layer_idx < num_hash_layers` (3 for V4-Flash).

```python
# PyTorch-operator equivalent of one row (token) of work.
#
# Common inputs:
#   gating_output:    [M, E]  fp32/bf16/fp16    (E.g. [M, 256] on V4-Flash)
#   routed_scaling:   scalar float              (V4-Flash: 1.5)
#   renormalize:      bool                      (V4-Flash: True)
#   k = topk:         int                       (V4-Flash: 6)
#
# Common preamble (lines 168-232): cast row to fp32 in registers row_chunk[VPT].

scores_row = gating_output[t, :].float()                            # [E] fp32

# softplus_sqrt with numerical stability:
beta = 1.0; threshold = 20.0
softplus = torch.where(scores_row * beta > threshold,
                       scores_row,
                       torch.log1p(torch.exp(scores_row * beta)) / beta)
scores_unbiased = torch.sqrt(softplus)                              # = sqrt(softplus(x))

if USE_HASH:
    # ===================== HASH BRANCH (lines 237-300) =====================
    # `tid2eid[token_id]` already holds the k chosen experts; no scoring/topk.
    token_id      = input_ids[t]                                    # int
    expert_ids    = tid2eid[token_id, :]                            # [k] int

    # Bias is NOT added in this branch (correction_bias is null for hash layers).
    # We still need the *unbiased* scores at those expert positions:
    selected_w    = scores_unbiased[expert_ids]                     # [k] fp32

    # Optional renormalize across the k chosen experts (lines 268-279):
    selected_sum  = selected_w.sum() if renormalize else 1.0
    denom         = selected_sum if (renormalize and selected_sum > 0) else 1.0
    scale         = routed_scaling / denom if renormalize else routed_scaling

    topk_indices[t, :]  = expert_ids                                # direct copy (line 259)
    topk_weights[t, :]  = selected_w * scale                        # (line 292)
    # token_expert_indices is NOT written on this branch (early return line 300).

else:
    # ===================== SCORED BRANCH (lines 301-425) ===================
    # Add bias for SELECTION, then top-k argmax. The bias is REMOVED before
    # the weight is written so the weight is the un-biased score.
    scores_for_choice = scores_unbiased + (correction_bias if correction_bias is not None
                                           else 0.0)                # [E] fp32

    # Iterative k-pass argmax with butterfly reduce (lines 328-408):
    selected_sum  = 0.0
    for k_idx in range(k):
        # Per-thread local argmax over its VPT experts (lines 330-346)
        # then butterfly across THREADS_PER_ROW lanes (lines 352-366);
        # ties broken by lower-expert-index winning (line 361-365).
        max_val, expert = topk_argmax_warp(scores_for_choice)

        # Gate by node range — out-of-range experts get the sentinel NUM_EXPERTS
        # (used at line 384). V4-Flash MegaMoE uses EP so start_expert/end_expert
        # carve the local slice.
        node_uses_expert = (expert >= start_expert) and (expert < end_expert)
        should_process   = (not finished[t] if finished is not None else True) and node_uses_expert

        # Strip bias from the WEIGHT (line 379-381).
        if correction_bias is not None:
            max_val = max_val - correction_bias[expert]

        topk_weights[t, k_idx]         = max_val
        topk_indices[t, k_idx]         = (expert - start_expert) if should_process else NUM_EXPERTS
        token_expert_indices[t, k_idx] = k_idx * M + t

        if renormalize:
            selected_sum += max_val

        # Blank out the winning slot to -inf for the next k iteration
        # (lines 393-407) so the same expert can't win twice.
        scores_for_choice[expert] = -10000.0

    if renormalize:
        denom = selected_sum if selected_sum > 0 else 1.0
        scale = routed_scaling / denom
    else:
        scale = routed_scaling
    topk_weights[t, :] = topk_weights[t, :] * scale                 # (lines 411-419)
```

Numerical / fusion notes:
- **softplus stability**: `beta = 1.0`, `threshold = 20.0` (lines 233-234). For `x > 20`, `softplus(x) ≈ x` (within fp32 ε). For `x ≤ 20`, the explicit `__logf(1.0f + __expf(val_b)) / beta` (line 244, line 308) is used. This matches the PyTorch reference `F.softplus(scores).sqrt()` at `inference/model.py:571` within 1 ULP for `|x| < 20`.
- **Bias semantics**: bias is added pre-selection (lines 316, 379) and subtracted post-selection (lines 380-381). The PyTorch reference in `inference/model.py:572-580` calls this out explicitly: "Bias shifts scores for expert selection (topk) but does not affect routing weights." The kernel preserves this contract — `topk_weights[t, k_idx]` is computed from the un-biased `sqrt(softplus(.))`.
- **Bias on hash branch**: zero. `Gate.__init__` at `inference/model.py:558-562` sets `self.bias = None` when `self.hash = True`. `nvidia/model.py:445-447` mirrors this. The kernel's hash branch never references `correction_bias`.
- **Renormalize semantics**: applied AFTER bias is stripped (line 387: `selected_sum += max_val` where `max_val` has already had bias subtracted). The renorm denominator is the sum of the un-biased weights. Reference: `inference/model.py:582` `weights /= weights.sum(dim=-1, keepdim=True)`.
- **Tie-breaking**: lower expert index wins (line 361-365). Critical for determinism with the PyTorch reference (`torch.topk` with `sorted=False` does NOT guarantee tie ordering; the kernel's tie break gives bit-exact reproducibility on equal scores).
- **Hash branch indices NOT offset by start_expert**: see lines 259, 292 — `indices[idx] = expert` writes the global expert id verbatim. This is because hash MoE layers in V4-Flash do not use EP slicing; the indices feed into the FusedMoE permutation which expects global ids.
- **Inner-loop expert blanking** (lines 393-407): only the lane owning the winning expert writes `-10000.f` into its `row_chunk` slot. The exact lane is computed from `expert / COLS_PER_GROUP_LDG` and `(expert / ELTS_PER_LDG) % THREADS_PER_ROW`.

## Config-dependent dispatch

- **Activation condition**: V4-Flash sets `score_func = "sqrtsoftplus"` (`inference/config.json:11`); `DeepseekV4MoE` at `nvidia/model.py:425` reads this and `fused_topk_bias` at `fused_topk_bias_router.py:216` matches the branch. Always-on for V4-Flash NVIDIA.
- **USE_HASH=true selection**: dispatched in the host helper at lines 642-646: `if (tid2eid.has_value()) use_hash = true;` This in turn depends on `nvidia/model.py:447-461` populating `self.gate.tid2eid` only when `extract_layer_index(prefix) < config.num_hash_layers`. V4-Flash: `num_hash_layers = 3`, so layers 0/1/2 → hash branch, layers 3..42 → scored branch.
- **Locked alternative pointer — `topk_softmax`**: `csrc/moe/topk_softmax_kernels.cu` + binding `csrc/moe/torch_bindings.cpp` `m.def("topk_softmax(...)")`. Selected by `scoring_func == "softmax"` in `fused_topk_bias_router.py:192-203`. **NOT used by V4-Flash** (its `score_func` is `sqrtsoftplus`). No separate spec.
- **Locked alternative pointer — `topk_sigmoid`**: `csrc/moe/torch_bindings.cpp:11-17` `topk_sigmoid` op. Selected by `scoring_func == "sigmoid"` in `fused_topk_bias_router.py:204-215`. **NOT used by V4-Flash**. No separate spec.
- **PyTorch fallback (XPU/CPU)**: `_topk_softplus_sqrt_torch` at `fused_topk_bias_router.py:60-103` — used when `current_platform.is_xpu()` (line 119). NVIDIA always uses the CUDA kernel. Not in scope for V4-Flash B200.
- **Downstream consumer constraints**:
  - `topk_indices.dtype` is consumed by `FusedMoE` / `DeepGEMM mega_moe` permutation routines. V4-Flash uses `int32` for `FusedMoE` and `int64` for `MegaMoE` (`nvidia/model.py:448`: `self.hash_indices_dtype = torch.int64 if self.use_mega_moe else torch.int32`). The kernel supports all three (int32/uint32/int64) via templated `IndType` (host dispatch at lines 647-687).
  - `topk_weights` is consumed by the expert permutation as fp32. Kernel always writes fp32.
- **Hard preconditions** (TORCH_CHECK at lines 621, 700, 705-725):
  - `num_experts ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 192, 320, 384, 448, 576}` (V4-Flash: 256 ✓)
  - `gating_output.scalar_type() ∈ {fp32, fp16, bf16}` (V4-Flash: fp32 from `GateLinear.out_dtype = torch.float32` at `nvidia/model.py:441`)
  - `topk_indices.scalar_type() ∈ {int32, uint32, int64}`
  - When `tid2eid.has_value()`, `input_ids.has_value()` MUST also be true (assert at line 644)
  - On CUDA, `WARP_SIZE == 32` (static_assert at line 506-507)
