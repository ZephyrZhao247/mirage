# fp8_fp4_mega_moe

## Identity
- Source file: `vllm/third_party/deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh:21-62` (kernel template signature `sm100_fp8_fp4_mega_moe_impl`); body continues through the file (~1500 lines covering dispatch warps, L1 GEMM + SwiGLU epilogue, L2 GEMM, combine reduce).
- Python wrapper: `vllm/third_party/deep_gemm/mega/__init__.py:108-128` (`fp8_fp4_mega_moe`); the symmetric buffer helper `transform_weights_for_mega_moe` at lines 96-105 and `_transpose_sf_for_utccp` at lines 87-93 pre-process the FP4 weights to match the kernel's UTCCP layout.
- Language/DSL: **CUDA** (CUTLASS / CuTe; TMA + tcgen05 PTX intrinsics). Built into the precompiled `deep_gemm._C` extension.
- Third-party dep: DeepGEMM (vendored at `vllm/third_party/deep_gemm/`), CUTLASS (transitively).
- Registered as: `torch.ops` not used — the Python wrapper calls `_C.fp8_fp4_mega_moe(...)` directly (mega/__init__.py:117).
- **SM target**: `sm_100a` (Blackwell). Body gated by `#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 1000)) or defined(__CLION_IDE__)` at line 63. Uses `tcgen05` instructions and 2-CTA cluster MMA exclusive to SM100a.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/nvidia/model.py:382` | `DeepseekV4MegaMoEExperts._run_mega_moe` | `y: [T, H]`, `l1_weights = (W13: [E, 2*I, H/2] fp4, W13_sf: [E, 2*I, H/32] int32)`, `l2_weights = (W2: [E, H, I/2] fp4, W2_sf: [E, H, I/32] int32)`, `symm_buffer` (FP8 acts + UE8M0 scales) | bf16 output; FP8 E4M3 acts; FP4 E2M1 weights; UE8M0 packed-int32 scales | `vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"`; requires `--enable-expert-parallel` |

Single call site in production. Runs once per MoE-layer forward, immediately after `prepare_megamoe_inputs` populates the FP8 symmetric-memory dispatch buffers and after `finalize_weights` has interleaved + UTCCP-transposed the FP4 weights.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `y` | `[num_tokens, hidden_size]` (V4-Flash: H=7168) | bf16 | row-major, contiguous | Output buffer the combine-reduce stage writes into |
| `l1_weights[0]` (`tensor_map_l1_weights`) | `[E_local, 2*I, H/2]` after `_interleave_l1_weights` reorders gate/up by groups of 8 (V4-Flash: I=2048) | float_e2m1_unpacksmem_t (FP4) | K-major, TMA-described; gate/up interleaved in groups of 8 along N (mega/__init__.py:75-84) | L1 fused gate+up weight (combined `w1 \| w3` in the reference Expert, see `model.py:591,593`) |
| `l1_weights[1]` (`tensor_map_l1_weights_sf`) | `[E_local, 2*I, H/32]` UTCCP-transposed via `_transpose_sf_for_utccp` (mega/__init__.py:87-93) | int32 (packed UE8M0 scales) | UTCCP 4x32 layout in MN: `idx → (idx & ~127) + (idx & 31)*4 + ((idx>>5) & 3)` (cuh:127-132) | UE8M0 group scales for L1 FP4 weights |
| `l2_weights[0]` (`tensor_map_l2_weights`) | `[E_local, H, I/2]` (not interleaved) | float_e2m1_unpacksmem_t (FP4) | K-major, TMA-described | L2 down-projection weight (`w2` in the reference Expert) |
| `l2_weights[1]` (`tensor_map_l2_weights_sf`) | `[E_local, H, I/32]` UTCCP-transposed | int32 (packed UE8M0 scales) | UTCCP 4x32 MN layout | UE8M0 group scales for L2 FP4 weights |
| `sym_buffer` (`symm_buffer.buffer` + descriptors via TMA) | bytes; views: `x: [T, H] fp8`, `x_sf: [T, H/32] int32`, `topk_idx: [T, top_k] int64`, `topk_weights: [T, top_k] fp32`, plus intermediate pools `l1_acts`, `l1_acts_sf`, `l2_acts`, `l2_acts_sf` | mixed; sym-mem backed | NVLink symmetric-memory tensor (`symm_mem.rendezvous`-allocated; mega/__init__.py:38-39) | All-to-all dispatch + combine staging area; each rank reads peers' slices via NVLink |
| `cumulative_local_expert_recv_stats` | `[E_local]` (or `None`) | int32 | optional | Telemetry counter for tokens received per local expert |
| `recipe` | `(1, 1, 32)` tuple | int | constexpr config | DeepGEMM scale-recipe ID — selects the 1x1x32 group-scale codepath |
| `activation` | `"swiglu"` (str) | — | constexpr config | Activation kind — SwiGLU is the only supported value here |
| `activation_clamp` | scalar float or `None` | fp32 (or None) | runtime arg | Pre-SwiGLU clamp magnitude. When set, gate is clamped to `[-c, c]` and up to `[-c, +c]` matching DeepSeek's `swiglu_limit` convention. `DeepseekV4MoE.forward` passes `float(self.swiglu_limit) if self.swiglu_limit is not None else None` (model.py:570-572). |
| `fast_math` | bool | — | runtime arg | Selects approximate SwiGLU/silu math path |
| Template params (compile-time, `__grid_constant__`) | `kNumMaxTokensPerRank, kHidden, kIntermediateHidden, kNumExperts, kNumTopk, kNumExpertsPerWave, BLOCK_M, BLOCK_N=128, BLOCK_K=128, STORE_BLOCK_M, SF_BLOCK_M, SF_BLOCK_N, kNumMaxPoolTokens, kNumPaddedSFPoolTokens, kNumStages, kNumDispatchThreads, kNumNonEpilogueThreads=128, kNumEpilogueThreads, kNumSMs, kNumRanks, kActivationClamp, kFastMath` | uint32_t / bool / float | template | Auto-tuned by DeepGEMM JIT layer; the in-Python `fp8_fp4_mega_moe` wrapper does not expose these. |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `y` (in-place) | `[num_tokens, hidden_size]` | bf16 | row-major | Combined MoE result `Σ_topk weights * Expert(x)`. Reductions happen inside `combine_token_buffer` (cuh:156-159) and a final TMA store writes to `y`. |
| `cumulative_local_expert_recv_stats` (optional) | `[E_local]` | int32 | — | Accumulated per-local-expert receive counts (optional telemetry; pass `None` to skip). |
| `sym_buffer.l1_acts*, l2_acts*` | various intermediates | fp8 / int32 | sym-mem | Scratch buffers re-used across calls (zeroed at construction). |

## Grid / Block

- Grid: `(kNumSMs, 1, 1)` — persistent grid sized to fully occupy the device. SM count is a template parameter; the kernel scheduler (`MegaMoEScheduler`, cuh:316-322) hands out `(expert_id, m_block_id, l1/l2)` work items inside each SM.
- Cluster: 2 CTAs per cluster (2-CTA MMA on Blackwell). `cute::block_rank_in_cluster()` is 0 for the leader CTA (cuh:74), 1 for the peer.
- Block: `kNumThreads = kNumDispatchThreads + kNumNonEpilogueThreads + kNumEpilogueThreads` threads/CTA. Warps are partitioned by role:
  - **Dispatch warps** (`warp_idx < kNumDispatchWarps`): count tokens per expert, atomically allocate slots in the symmetric send-buffers, issue NVLink `multimem` writes to peers' `l1_acts`/`l1_acts_sf` buffers (cuh:359-388 …). 48 registers each.
  - **MMA non-epilogue warps** (`kNumNonEpilogueThreads == 128 ⇒ 4 warps`): drive UMMA via `tcgen05`, issue TMA loads of `(A, B, SFA, SFB)`, manage the 2-CTA cluster barrier handshake. 40 registers each.
  - **Epilogue warps** (`kNumEpilogueThreads / 32` warps, in warpgroups of 4): pull MMA results from tensor memory (`TMEM`), apply SwiGLU (with `activation_clamp` clip), requantize to FP8 for L1 output / keep bf16 for L2 output, TMA-store to symmetric pool, finally combine-reduce per-topk fragments back to `y`. 208 registers each.
- 2-CTA MMA layout (cuh:166-176): `UMMA_M = 2 * LAYOUT_AD_M = 256`, `UMMA_N = BLOCK_M`, `UMMA_K = 32`. K-major matrices, A/B swapped (acts → "B", weights → "A"). `LOAD_BLOCK_M = BLOCK_M / 2` (multicast on A).
- Swizzle (cuh:178-182): A and B both use `BLOCK_K * sizeof(elem) = 128 * 1B = 128B` swizzle for FP8; CD uses 128B swizzle.
- Pipeline (cuh:325-332): `kNumStages` MMA stages with `phase ^= (stage_idx == 0)` flipping at wrap-around. `kNumEpilogueStages = 2`, `kNumTMAStoreStages = 2`.
- Tensor memory (cuh:217-223): `kNumAccumTmemCols = UMMA_N * kNumEpilogueStages`, plus `SF_BLOCK_M/32` SFA and `SF_BLOCK_N/32` SFB columns; rounded to a multiple of 32, ≤ 512.
- Per-CTA work granularity: one cluster cooperates on a `(BLOCK_M, BLOCK_N, BLOCK_K) = (BLOCK_M, 128, 128)` GEMM tile per stage, where `BLOCK_M` is a template-tuned multiple of 16. The L1 epilogue post-SwiGLU output tile is `BLOCK_N / 2 = 64` (`L1_OUT_BLOCK_N`, cuh:194) because SwiGLU halves N.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:596-606` (`Expert.forward` — SwiGLU FFN: `silu(clamp(w1·x)) * clamp(w3·x) → w2`) and `model.py:630-644` (`MoE.forward` — top-k routing + per-expert dispatch + weight-multiplied accumulation + shared-expert add + all-reduce). The reference iterates over experts in Python and does `y[idx] += expert(x[idx], weights[idx, top, None])`; this kernel fuses dispatch, both expert GEMMs, the activation, the requant, and the combine into a single persistent grid.

```python
# PyTorch-operator equivalent of one persistent-grid forward.
# Reference: DeepSeek-V4-Flash/inference/model.py:596-606 (Expert.forward) and
#            DeepSeek-V4-Flash/inference/model.py:630-644 (MoE.forward).
#
# Inputs (post prepare_megamoe_inputs, post finalize_weights):
#   x_fp8:     [T, H] float8_e4m3fn          (in symm_buffer.x)
#   x_sf:      [T, H//32] int32 (UE8M0)       (in symm_buffer.x_sf)
#   topk_idx:  [T, top_k] int64               (in symm_buffer.topk_idx)
#   topk_w:    [T, top_k] fp32                (in symm_buffer.topk_weights)
#   W13:       [E_local, 2*I, H//2] fp4-e2m1, gate/up interleaved by groups of 8
#   W13_sf:    [E_local, 2*I, H//32] int32 UE8M0 (UTCCP-transposed)
#   W2:        [E_local, H, I//2] fp4-e2m1
#   W2_sf:     [E_local, H, I//32] int32 UE8M0
#   activation_clamp: float | None  (V4-Flash: float(swiglu_limit) or None)

# 1) Dispatch: each token's (x_fp8[t], x_sf[t]) is multimem-pushed to the owners of
#    every expert in topk_idx[t]. Per-expert recv slots in symm_buffer.l1_acts
#    are atomically allocated by dispatch warps.
#
# 2) L1 GEMM (gate + up combined) on each rank's local experts:
for e in range(E_local):
    # M = number of tokens routed to local expert e (post dispatch).
    a_e = symm_buffer.l1_acts[e]      # [M_e, H] fp8 e4m3 (B-side after swap A/B)
    a_sf_e = symm_buffer.l1_acts_sf[e]  # [M_e, H//32] UE8M0 int32

    # FP4 × FP8 group-scaled MMA. K-major, BLOCK_K=128, group-scale period 32.
    # Effective math: per K-group of 32 elements:
    #   acc += dequant_fp4(W13[e]) * scale_W13[e] * dequant_fp8(a_e) * scale_a_e
    # accumulated in fp32 tensor-memory.
    l1_acc = matmul_fp4_fp8_grouped(
        W13[e], W13_sf[e],   # [2*I, H//2] fp4, [2*I, H//32] int32
        a_e,    a_sf_e,      # [M_e, H]    fp8, [M_e, H//32]  int32
        recipe=(1, 1, 32),
    )                                             # [M_e, 2*I] fp32

    # 2a) SwiGLU activation + clamp + requant to FP8 for L2.
    #     Gate/up interleaving means lane-pairs (g_i, u_i) are co-located.
    gate, up = l1_acc.chunk(2, dim=-1)            # each [M_e, I]
    if activation_clamp is not None:
        gate = gate.clamp(max=activation_clamp)
        up   = up.clamp(min=-activation_clamp, max=activation_clamp)
    h = F.silu(gate) * up                         # [M_e, I] fp32

    # Per-group UE8M0 requant (same recipe as activation prep):
    h_fp8, h_sf = quant_fp8_ue8m0_groups(h, group_size=32)   # [M_e, I] fp8, [M_e, I//32] int32
    symm_buffer.l2_acts[e]    = h_fp8
    symm_buffer.l2_acts_sf[e] = h_sf

    # 3) L2 GEMM (down projection).
    out_e = matmul_fp4_fp8_grouped(
        W2[e], W2_sf[e],   # [H, I//2] fp4, [H, I//32] int32
        h_fp8, h_sf,        # [M_e, I] fp8, [M_e, I//32] int32
        recipe=(1, 1, 32),
    )                                              # [M_e, H] fp32 → bf16

    # Stage out_e into the per-topk combine buffer (one row per (token, k) pair).
    for (t, k) in expert_to_tokens(e):
        symm_buffer.combine[t, k] = out_e[t_local(t, k)]  # bf16

# 4) Combine reduce: sum across topk + multimem reduce across ranks.
for t in range(T):
    y[t] = sum(topk_w[t, k] * symm_buffer.combine[t, k] for k in range(top_k))
# (multimem all-reduce across kNumRanks happens via NVLink in the combine warps;
#  shared_experts are added by the Python caller, model.py:580-582.)
```

Notes on fusion / quant:
- **FP4 weights × FP8 acts**: the kernel uses `cutlass::float_e4m3_t` for activations (A-after-swap) and `cutlass::detail::float_e2m1_unpacksmem_t` for weights (B-after-swap), per cuh:163-164. UE8M0 group scales (period 32) are applied via the `tcgen05.cp` (UTCCP) path, hence the SF-transposed `int32` layout pre-baked by `_transpose_sf_for_utccp`.
- **2-CTA MMA + multicast A**: cuh:170-172 sets `UMMA_M = 2 * LAYOUT_AD_M`, halves the per-CTA A load (`LOAD_BLOCK_M = BLOCK_M / 2`), and uses cluster-broadcast TMA so both CTAs share A. Weights are larger and remain per-CTA (B).
- **SwiGLU + gate/up interleave**: `_interleave_l1_weights` (mega/__init__.py:75-84) reorders the L1 weight so that `[gate_0..7, up_0..7, gate_8..15, up_8..15, …]` is contiguous along N. This makes the post-MMA SwiGLU read 8-wide gate and 8-wide up rows from adjacent tensor-memory columns, halving the epilogue's bank conflicts.
- **`activation_clamp` semantics**: the reference at `model.py:600-602` clamps `up` to `[-swiglu_limit, +swiglu_limit]` and `gate` to `(-∞, swiglu_limit]`. The kernel's `kActivationClamp` constexpr / runtime `activation_clamp` param mirrors this; passing `None` from the Python side disables the clamp (V4-Flash default has `swiglu_limit=0.0` per `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:56`, so the upper-layer caller passes `None` in that case).
- **`fast_math`**: when `True`, the epilogue uses an approximate `silu` (e.g. `__expf` and `silu = x / (1 + __expf(-x))` instead of fp32-exact `exp`).
- **Combine reduce as part of the persistent kernel**: instead of writing each expert's bf16 output back to HBM and running a separate reduction, the L2 epilogue stages each `(token, k)` slice into `combine_token_buffer` (sym-mem) and a final pass does the topk-weighted sum + cross-rank all-reduce inside this kernel. The `multimem` instructions targeted by `kBeforeCombineReduceBarrierTag` (cuh:342) coordinate the cross-rank reduce.

## Config-dependent dispatch

- Activation condition: **active iff `vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"`** (model.py:407-409).
- Hardware requirement: **`sm_100a` (Blackwell SM100) only**. Enforced at runtime by `_check_runtime_supported` (model.py:253-261): `torch.cuda.get_device_capability(device)[0] != 10` raises `NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")`. Compile-time gate at cuh:63 means the body is empty on `__CUDA_ARCH__ < 1000`. The kernel uses `tcgen05` + 2-CTA cluster MMA which are Blackwell-only — not just SM100 but specifically `sm_100a` (a-variant, with architecture-specific features).
- Config preconditions (model.py:410-435):
  - `--enable-expert-parallel` REQUIRED (raises at line 411 otherwise).
  - `scoring_func == "sqrtsoftplus"` REQUIRED (raises at line 427 otherwise).
  - `expert_dtype == "fp4"` REQUIRED (raises at line 431 otherwise).
- Shape preconditions (model.py:257-261):
  - `hidden_size % 128 == 0` and `intermediate_size % 128 == 0`.
  - `num_experts % num_ranks == 0` (asserted in template via `DG_STATIC_ASSERT` cuh:71).
  - `kNumTopk <= 32` (asserted in dispatch path, cuh:364).
- **V4-Flash-Base compatibility caveat**: V4-Flash-Base ships `expert_dtype="fp8"` (`deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:41`), which triggers the `expert_dtype != "fp4"` `NotImplementedError` at `model.py:430-434`. So MegaMoE is INCOMPATIBLE with the V4-Flash-Base checkpoint as shipped, but is documented here as the canonical vLLM fast path for any hypothetical FP4-export of DeepSeek V4. For V4-Flash-Base, the FusedMoE branch (see `fused_moe_kernel.md`) is the active path.
- Downstream consumers: `y` (bf16) is added to the shared-expert output (`shared_experts(hidden_states)`) at model.py:580-582 and returned. Layout: `[num_tokens, hidden_size]` bf16 row-major — matches the input `hidden_states` layout and feeds the post-MoE residual.
