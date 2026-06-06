# dsv3_router_gemm

## Identity

This spec covers the **MoE gate-projection GEMM**: `router_logits = gate_weight @ hidden_states.T` (computed as `hidden_states @ gate_weight.T`), the matmul that feeds `topk_softplus_sqrt`. There is no single kernel — `GateLinear.forward()` selects from four tiers by shape/dtype. All four tiers are Class B variants (selected by `(input_size, output_size, x.dtype, weight.dtype, num_tokens)`); V4-Flash falls to **Tier 4 (`F.linear`)** because its hidden dim is 4096, not 7168 or 3072.

- Sources:
  - **Tier 1** — `dsv3_router_gemm` (DSV3-specialized, H=7168, E∈{256,384}, M∈[1..16]):
    - Entry: `csrc/moe/dsv3_router_gemm_entry.cu:101-165`
    - fp32-out kernel body: `csrc/moe/dsv3_router_gemm_float_out.cu:54-171` (`router_gemm_kernel_float_output<T, kBlockSize=128, VPT, kNumTokens, kNumExperts, kHiddenDim>`); template instantiations at lines 196-292 (E=256/384, M=1..16, H=7168, bf16 input).
    - bf16-out kernel body: `csrc/moe/dsv3_router_gemm_bf16_out.cu` (parallel structure to float-out).
    - Utils: `csrc/moe/dsv3_router_gemm_utils.h:28-31` (`getSMVersion`).
    - Op registration: `csrc/moe/torch_bindings.cpp:139` schema `dsv3_router_gemm(Tensor! output, Tensor mat_a, Tensor mat_b) -> ()`; impl at `csrc/moe/dsv3_router_gemm_entry.cu:167-169` registers under `_moe_C`.
  - **Tier 2** — `fp32_router_gemm` (fp32-specialized, H=3072, E=256, M∈[1..32]):
    - Kernel: `csrc/libtorch_stable/fp32_router_gemm.cu:78-151` (`fp32_router_gemm_kernel<InputT, kBlockSize=128, kNumTokens, kNumExperts, kHiddenDim>`); template instantiations at lines 181-220 (E=256, H=3072, M=1..32, InputT ∈ {float, bf16}).
    - Entry: `csrc/libtorch_stable/fp32_router_gemm_entry.cu:63-123`.
    - Op registration: `csrc/libtorch_stable/torch_bindings.cpp:252` schema `fp32_router_gemm(Tensor! output, Tensor mat_a, Tensor mat_b) -> ()`; impl at `csrc/libtorch_stable/fp32_router_gemm_entry.cu:125-127`. Reachable as `torch.ops._C.fp32_router_gemm`.
  - **Tier 3** — cuBLAS bf16×bf16→fp32 (`torch.mm` with `out_dtype=torch.float32`): plain PyTorch, dispatches to cuBLASLt.
  - **Tier 4** — `F.linear` fallback: `vllm.model_executor.layers.linear.ReplicatedLinear.forward` (parent class' `super().forward(x)` at `gate_linear.py:141`).
- Python dispatch: `vllm/model_executor/layers/fused_moe/router/gate_linear.py:111-144` (`GateLinear.forward`). Also `gate_linear.py:150-175` wraps Tier 2 in a `direct_register_custom_op('fp32_router_gemm_dispatch')` to make `num_tokens` branching torch.compile-safe.
- Custom op wrappers: `vllm/_custom_ops.py:2400-2412` (`dsv3_router_gemm`, allocates output, calls `torch.ops._moe_C.dsv3_router_gemm`); `vllm/_custom_ops.py:2415-2426` (`fp32_router_gemm`, allocates fp32 output, calls `torch.ops._C.fp32_router_gemm`).
- Language/DSL: **CUDA C++** (all kernels). Hand-tuned per-shape; no CUTLASS/CUTE. Both Tier 1 and Tier 2 use the same algorithmic skeleton: warp-cooperative inner-product over the K dim with butterfly warp reduction + shared-memory cross-warp reduction; one CTA per expert (column of B).
- Third-party dep: none. Tier 1 was ported from SGLang's `sgl-kernel`, originally from TensorRT-LLM `dsv3RouterGemm` (file headers, lines 1-19).
- Registered as: `torch.ops._moe_C.dsv3_router_gemm` (Tier 1), `torch.ops._C.fp32_router_gemm` (Tier 2). Tiers 3 and 4 use core PyTorch ops.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate / tier |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/nvidia/model.py:555` | `DeepseekV4MoE.forward` (mega-MoE path) → `router_logits, _ = self.gate(hidden_states)` | `hidden_states: [M, H=4096]` (V4-Flash) | bf16 in, fp32 out | → `GateLinear.forward` → **Tier 4** (`H=4096` does not match Tier 1's 7168 nor Tier 2's 3072). |
| `vllm/models/deepseek_v4/nvidia/model.py:598` | `DeepseekV4MoE._forward_fused_moe` → `router_logits, _ = self.gate(hidden_states)` | same | same | same dispatch logic; **Tier 4** on V4-Flash. |
| `vllm/model_executor/layers/fused_moe/router/gate_linear.py:116` | `GateLinear.forward` (Tier 1 site) | `x: [M≤16, H=7168]`, `weight: [E∈{256,384}, 7168]` | bf16 in, fp32 or bf16 out | Selected if `allow_dsv3_router_gemm and x.shape[0] <= 16`. **NOT V4-Flash** (DeepSeek V3 / Kimi K2). |
| `vllm/model_executor/layers/fused_moe/router/gate_linear.py:130` | `GateLinear.forward` (Tier 2 site, wrapped in `fp32_router_gemm_dispatch` custom op) | `x: [M≤32, H=3072]`, `weight: [E=256, 3072]` | bf16/fp32 in, fp32 weight, fp32 out | Selected if `allow_fp32_router_gemm` and `x.dtype ∈ {fp32, bf16}`. The custom op falls back to `F.linear(x.float(), weight)` for `M > 32` (gate_linear.py:159-162). |
| `vllm/model_executor/layers/fused_moe/router/gate_linear.py:135` | `GateLinear.forward` (Tier 3 site) | bf16 weight + fp32 out_dtype | bf16 in, bf16 weight, fp32 out | `allow_cublas_router_gemm`. Issues `torch.mm(x, self.weight.T, out_dtype=torch.float32)` → cuBLASLt. |
| `vllm/model_executor/layers/fused_moe/router/gate_linear.py:141` | `GateLinear.forward` (Tier 4 site) | any | any | All Tier 1-3 checks failed (V4-Flash's `H=4096` lands here). |

V4-Flash's `GateLinear` is constructed at `nvidia/model.py:437-443` with `input_size = config.hidden_size = 4096`, `output_size = config.n_routed_experts = 256`, `bias = False`, `out_dtype = torch.float32`. With `H=4096`: `allow_dsv3_router_gemm = False` (H not in {7168}), `allow_fp32_router_gemm = False` (H not in {3072}), `allow_cublas_router_gemm` would be True if `weight.dtype == bf16` (set by default `params_dtype` for the `ReplicatedLinear` parent), so **Tier 3 (cuBLASLt bf16→fp32) is the most likely V4-Flash hot path**, not Tier 4. **DECISION REQUIRED**: confirm V4-Flash `GateLinear.weight.dtype` at runtime — if bf16, Tier 3 fires; if fp32 (e.g., `force_fp32_compute=True`), Tier 4 fires.

## Inputs

Each tier has its own contract. Common across all four: output is fp32 `[M, E]` (V4-Flash: `[M, 256]`) since `out_dtype = torch.float32` (set by `nvidia/model.py:441`).

### Tier 1: `dsv3_router_gemm` (`csrc/moe/dsv3_router_gemm_entry.cu:101-165`)

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `output` | `[M, E]` | fp32 OR bf16 (asserted at line 126) | row-major, contiguous on E | Output gate logits |
| `mat_a` (`hidden_states`) | `[M, H=7168]`, `M ∈ [1, 16]` | bf16 (asserted line 124) | row-major | Token activations |
| `mat_b` (`router_weight`) | `[E, H=7168]`, `E ∈ {256, 384}` (asserted lines 117-120) | bf16 (asserted line 125) | row-major (one row per expert; the kernel reads `b_col = mat_b + n_idx * kHiddenDim` at line 74 — so the underlying storage is **column-major in GEMM terms**: the kernel treats the leading dim as `kHiddenDim`, one column per expert) | Gate weight |
| SM gate | `sm >= 90 && sm <= 103` (lines 129-130) | — | — | H100 / Hopper / B200 (sm_90 .. sm_100). V4-Flash B200 ✓. |

### Tier 2: `fp32_router_gemm` (`csrc/libtorch_stable/fp32_router_gemm_entry.cu:63-123`)

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `output` | `[M, E=256]` | fp32 (asserted line 100) | row-major, contiguous (assert line 76) | Output gate logits |
| `mat_a` | `[M, H=3072]`, `M ∈ [1, 32]` | bf16 OR fp32 (asserted line 95-97) | row-major, contiguous | Token activations |
| `mat_b` | `[E=256, H=3072]` | fp32 (asserted line 98-99) | row-major, contiguous; same column-per-expert layout as Tier 1 | Gate weight (fp32 — Tier 2 is for models that store the gate in fp32) |
| SM gate | `sm >= 90` (line 107) | — | — | H100+ |

### Tier 3 (cuBLASLt): `torch.mm(x, weight.T, out_dtype=torch.float32)`

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `x` | `[M, H]` (V4-Flash: `[M, 4096]`) | bf16 | row-major contiguous | Activations |
| `weight` | `[E, H]` (V4-Flash: `[256, 4096]`) | bf16 (`allow_cublas_router_gemm` requires `weight.dtype == torch.bfloat16`, gate_linear.py:90-91) | row-major (transposed at call site via `.T`) | Gate weight |
| `out_dtype` | fp32 | — | — | Mixed-precision cuBLASLt configuration; output is fp32 |

### Tier 4 (F.linear): `ReplicatedLinear.forward(x)`

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `x` | `[M, H]` (V4-Flash: `[M, 4096]`) | bf16 or fp32 (auto-cast to `weight.dtype` at gate_linear.py:139-140) | contiguous | Activations |
| `weight` | `[E, H]` (V4-Flash: `[256, 4096]`) | typically bf16 (default `ReplicatedLinear` dtype) or fp32 (if `params_dtype=torch.float32` forced via `force_fp32_compute`) | contiguous | Gate weight |
| `out_dtype` | fp32 | — | — | Output cast to fp32 (gate_linear.py:142-143) to match `topk_softplus_sqrt`'s fp32-logits contract |

## Outputs

For all tiers: `output: [M, E]` fp32, row-major, contiguous on E. Consumed by `topk_softplus_sqrt` as `gating_output` (see `topk_softplus_sqrt.md`).

- **Tier 1** writes via `out[m * kNumExperts + n_idx] = final_sum` at `dsv3_router_gemm_float_out.cu:165` — one expert column per CTA writes M values in row-major order.
- **Tier 2** writes via `out[m * kNumExperts + n_idx] = final_sum` at `fp32_router_gemm.cu:144` — identical layout.
- **Tier 3** writes via cuBLASLt; layout is whatever `torch.mm` produces (row-major M × N).
- **Tier 4** writes via `F.linear` (cuBLASLt or aten backend).

Tier 1 bf16-out variant exists but V4-Flash and the DSV3 main checkpoint both use fp32-out (since fp32 logits feed `topk_softplus_sqrt`). The bf16-out branch (`output.dtype() == at::kBFloat16`, `dsv3_router_gemm_entry.cu:148-164`) is for callers that immediately consume the logits in bf16 (e.g., GroupedExpert routers); **not V4-Flash**.

## Grid / Block

### Tier 1 (`router_gemm_kernel_float_output`)

- `grid_dim.x = kNumExperts` (line 179) — one CTA per expert column. V4-Flash-equivalent shape: 256 or 384 CTAs.
- `block_dim = kBlockSize = 128` threads (line 177); 4 warps per CTA.
- `__launch_bounds__(128, 1)` (line 54).
- `VPT = 16 / sizeof(T) = 8` for bf16 (line 176). Each thread loads 8 bf16 values per K-iter via one `uint4` load (line 92). Block of 128 threads × 8 VPT = 1024 K-elements per iter; `k_iterations = H / 1024 = 7`.
- **Per-CTA work**:
  - Each CTA owns one expert column. `b_col = mat_b + n_idx * kHiddenDim` (line 74) — column-major B storage.
  - Inner loop over `k_iterations` (line 88-117): load 8 bf16 values from B (`uint4`), convert to fp32. For each of `kNumTokens` (template constant ∈ [1, 16]) M rows, load 8 bf16 from A, convert, FMA into per-thread `acc[m_idx]` (lines 100-116).
  - Warp-level butterfly reduce: `__shfl_xor_sync(0xffffffff, sum, {16,8,4,2,1})` (lines 137-142).
  - Shared-memory cross-warp reduce: lane 0 of each warp writes to `sm_reduction[m][warpId]` (line 146); `__syncthreads()` (line 150); thread 0 sums `kNumWarps = 4` warps' partials (lines 156-161) and writes to `out` (line 165).
- **PDL hooks**: `griddepcontrol.wait` at line 84, `griddepcontrol.launch_dependents` at line 169. Launched with `cudaLaunchAttributeProgrammaticStreamSerialization` (`dsv3_router_gemm_float_out.cu:184-188`).
- **Autotune**: NONE. `M` is a template constant; the `LoopUnroller` at `dsv3_router_gemm_entry.cu:43-99` dispatches `num_tokens ∈ [1, 16]` to the right instantiation; invalid `num_tokens` throws.

### Tier 2 (`fp32_router_gemm_kernel`)

- `grid_dim.x = kNumExperts = 256` (line 162) — one CTA per expert.
- `block_dim = 128` (line 163); `__launch_bounds__(128, 1)` (line 78).
- `VPT = 16 / sizeof(InputT)`: 4 for fp32, 8 for bf16 (line 80). For V4-Flash-equivalent bf16 with H=3072: `k_elems_per_k_iteration = 8 * 128 = 1024`; `k_iterations = 3072 / 1024 = 3`.
- Same warp-cooperative inner-product + butterfly + shared-mem reduce structure as Tier 1 (lines 86-150). Differences:
  - Weight is fp32 (loaded via `float4` for VPT=4 or two `float4`s for VPT=8, see `load_weight<4>` at lines 22-28 / `load_weight<8>` at lines 31-42).
  - Activations are templated on `InputT` (fp32 or bf16); the `load_activation` overloads at lines 50-67 convert bf16 → fp32 inline.
- PDL hooks at lines 102-104, 148-150.

### Tier 3 (cuBLASLt)

- Grid/block determined by cuBLASLt heuristics (opaque). Typical tile is 128×128 or 64×128 for bf16×bf16→fp32 GEMM on B200.

### Tier 4 (F.linear)

- Grid/block determined by ATen / cuBLAS. Same as Tier 3 in practice for bf16 × bf16 → bf16, then a separate cast for the fp32 downcast at gate_linear.py:142-143.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:565` (`scores = linear(x.float(), self.weight.float())` in `Gate.forward`). The reference computes the matmul in **fp32 explicitly** — both activation and weight are cast to fp32 before `linear`. All four tiers produce the same logits up to numerical drift.

```python
# PyTorch-operator equivalent (all four tiers compute this):
#
# Inputs:
#   x:      [M, H]  (V4-Flash: M variable, H=4096)
#   weight: [E, H]  (V4-Flash: E=256, H=4096)
#
# Output:
router_logits = (x.float() @ weight.float().T)                # [M, E] fp32

# Per the reference at inference/model.py:565, math precision is fp32
# (both operands cast to fp32 before linear). The four tiers approximate this
# at varying precision levels:
```

### Tier 1 — `dsv3_router_gemm`

```python
# x: [M≤16, H=7168] bf16; weight: [E∈{256,384}, H=7168] bf16; out: [M, E] fp32 (or bf16).
# Per-CTA (one CTA = one expert column n_idx), warp-cooperative:
#
# For each k-iteration (i in 0..k_iterations):
#   bf16 → fp32 conversion in registers (bf16_uint4_to_float8, lines 41-49)
#   acc[m_idx] += sum_{k in tile} a_float[k] * b_float[k]   # fp32 FMA
#
# After all k-iterations, butterfly warp-reduce + 4-warp sum (via shared mem).
# Output is fp32 (or bf16-cast after final sum).
router_logits = (x.float() @ weight.float().T).to(out.dtype)  # fp32 or bf16
```

Numerical drift vs reference: identical to a "bf16 → fp32 cast, fp32 multiply-add" decomposition. The ref already uses `x.float()` and `weight.float()` so the cast points match.

### Tier 2 — `fp32_router_gemm`

```python
# x: [M≤32, H=3072] bf16 OR fp32; weight: [E=256, H=3072] fp32; out: [M, E] fp32.
# Same warp-cooperative inner product as Tier 1 but weight already in fp32.
router_logits = x.float() @ weight.T          # weight is already fp32
```

Numerical drift: bit-equivalent to the reference when `x` is bf16 (the bf16 → fp32 cast at load is exact) and `weight` is fp32.

### Tier 3 — cuBLASLt bf16→fp32

```python
# x: [M, H=4096] bf16; weight: [E=256, H=4096] bf16.
# cuBLASLt computes:
#   intermediate = bf16(x) @ bf16(weight).T     # likely bf16-tf32 accumulator on B200
#   router_logits = intermediate.float()        # fp32 output via out_dtype=torch.float32
router_logits = torch.mm(x, weight.T, out_dtype=torch.float32)
```

Numerical drift vs reference: small. cuBLASLt on B200 uses bf16 MMA with fp32 accumulator by default, so the rounding error is bounded by the bf16 mantissa (7 bits) per FMA. Reference's `x.float() @ weight.float()` uses an fp32 multiplier (23-bit mantissa), which is more precise but more expensive.

### Tier 4 — F.linear

```python
# x: [M, H=4096]; weight: [E=256, H=4096].
# ReplicatedLinear.forward:
output = F.linear(x.to(weight.dtype), weight)  # bf16×bf16→bf16 or fp32×fp32→fp32
if out_dtype is not None:
    output = output.to(out_dtype)              # cast to fp32 (gate_linear.py:142-143)
```

Numerical drift: identical to a standard PyTorch GEMM. If `weight.dtype = bf16`, this is the slowest of the four (separate bf16 GEMM + fp32 cast) but the most portable.

## Config-dependent dispatch

This is a **Class B** kernel family — the four tiers are selected by `(input_size, output_size, x.dtype, weight.dtype, num_tokens)` and there is no single "active" path across models. **DECISION REQUIRED** for V4-Flash:

- V4-Flash's `GateLinear` is built with `H=4096, E=256`. Neither Tier 1's (`H=7168`) nor Tier 2's (`H=3072`) shape gate matches.
- `allow_cublas_router_gemm = allow_specialized_router_gemm AND weight.dtype == bf16 AND out_dtype == fp32` (`gate_linear.py:88-92`). `allow_specialized_router_gemm` is True on Hopper/Blackwell + CUDA + no-bias (`gate_linear.py:50-52, 70`). V4-Flash satisfies all: SM100, CUDA, `bias=False`. So if the weight is bf16 (default), Tier 3 fires; if fp32, Tier 4 fires.
- Reasonable default: **Tier 3** (cuBLASLt bf16→fp32). The user (or any caller) can force Tier 4 via `force_fp32_compute=True` (which sets `params_dtype = torch.float32` at `gate_linear.py:56-57`, but only when no specialized kernel is available — here a specialized kernel IS available via Tier 3, so `force_fp32_compute` does NOT trigger; this path is for non-Hopper/Blackwell devices).

### Per-tier conditions (verbatim from `gate_linear.py`):

| Tier | Active when | V4-Flash fires? |
| --- | --- | --- |
| 1 (`dsv3_router_gemm`) | `allow_dsv3_router_gemm and x.shape[0] <= 16`. `allow_dsv3_router_gemm = allow_specialized_router_gemm AND output_size ∈ {256, 384} AND input_size ∈ {7168}`. | No (V4-Flash H=4096, not 7168) |
| 2 (`fp32_router_gemm` via custom op) | `allow_fp32_router_gemm and x.dtype ∈ {fp32, bf16}`. `allow_fp32_router_gemm = not bias AND weight.dtype == fp32 AND cuda AND H100+ AND output_size ∈ {256} AND input_size ∈ {3072}`. Custom op then dispatches Tier 2 kernel iff `M <= 32`, else `F.linear(x.float(), weight)`. | No (V4-Flash H=4096, not 3072) |
| 3 (cuBLASLt bf16→fp32) | `allow_cublas_router_gemm and x.dtype == bf16`. `allow_cublas_router_gemm = allow_specialized_router_gemm AND weight.dtype == bf16 AND out_dtype == fp32`. | **Yes** if V4-Flash weight is bf16 (the default for `ReplicatedLinear`) |
| 4 (F.linear) | Otherwise. | Yes if V4-Flash weight is fp32 (forced via `force_fp32_compute`); otherwise No (Tier 3 takes it). |

### Downstream consumer constraints

- Output dtype MUST be fp32: `topk_softplus_sqrt`'s host dispatcher (`csrc/moe/topk_softplus_sqrt_kernels.cu:705-725`) supports fp32/fp16/bf16, but V4-Flash's MegaMoE path expects fp32 (set via `out_dtype=torch.float32` at `nvidia/model.py:441`).
- Output layout MUST be `[M, E]` row-major contiguous: all four tiers produce this; no transpose needed.

### Hard preconditions per tier

- **Tier 1**: `TORCH_CHECK`s at `dsv3_router_gemm_entry.cu:105-130`:
  - `output.dim() == 2 && mat_a.dim() == 2 && mat_b.dim() == 2`
  - `mat_a.size(1) == mat_b.size(1)`
  - `hidden_dim == 7168`
  - `num_experts ∈ {256, 384}`
  - `num_tokens ∈ [1, 16]`
  - `mat_a.dtype == bf16`, `mat_b.dtype == bf16`, `output.dtype ∈ {fp32, bf16}`
  - SM ∈ [90, 103]
- **Tier 2**: `STD_TORCH_CHECK`s at `fp32_router_gemm_entry.cu:68-107`:
  - All tensors CUDA, same device, contiguous, 2D.
  - `hidden_dim == 3072`, `num_experts == 256`
  - `num_tokens ∈ [0, 32]` (zero is a no-op early return at line 103)
  - `mat_a.dtype ∈ {fp32, bf16}`, `mat_b.dtype == fp32`, `output.dtype == fp32`
  - SM ≥ 90
- **Tier 3**: `torch.mm` requirements — `x` and `weight.T` 2D, contiguous on the inner dim.
- **Tier 4**: `F.linear` requirements (none beyond shape match).

### Locked alternative pointers

- The Tier 1 bf16-output variant (`dsv3_router_gemm_bf16_out.cu`) is used by DSV3-base / Kimi-K2 for in-place quant pipelines; V4-Flash uses fp32-out.
- `dsv3_router_gemm_dispatch` does NOT exist (Tier 1's custom op is called directly); only Tier 2 is wrapped in a Python-side custom op so torch.compile can freeze the `M <= 32` branch.
