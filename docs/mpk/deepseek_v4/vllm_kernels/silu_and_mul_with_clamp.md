# silu_and_mul_with_clamp

## Identity
- Source file: wrapper `silu_and_mul_clamp` at `csrc/libtorch_stable/activation_kernels.cu:266-271` (note: C++ symbol is `silu_and_mul_clamp`; the op-registry name and Python op are `silu_and_mul_with_clamp`).
- Templated kernel body: `act_and_mul_kernel<scalar_t, packed_t, silu_kernel, packed_silu_kernel, act_first=true, use_vec, HAS_CLAMP=true, use_256b>` at lines 78-125 (vectorized + scalar fallback paths). `compute<.., HAS_CLAMP=true>` at lines 13-35 handles the per-element clamp + silu + mul; `packed_compute<.., HAS_CLAMP=true>` at lines 37-71 handles the packed bf16/fp16 path.
- Launch macro: `LAUNCH_ACTIVATION_GATE_KERNEL(vllm::silu_kernel, vllm::packed_silu_kernel, ACT_FIRST=true, HAS_CLAMP=true, LIMIT=(float)limit)` at lines 204-257. Picks `use_256b = (CUDA_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128)` (i.e. SM100+ B200 with CUDA 12.9+, large enough batch) and `use_vec = (d % vec_size == 0)`.
- Op registration: `csrc/libtorch_stable/torch_bindings.cpp:368` schema `silu_and_mul_with_clamp(Tensor! result, Tensor input, float limit) -> ()`; `csrc/libtorch_stable/torch_bindings.cpp:592` impl `ops.impl("silu_and_mul_with_clamp", TORCH_BOX(&silu_and_mul_clamp))`. Reachable as `torch.ops._C.silu_and_mul_with_clamp`.
- Language/DSL: **CUDA C++** (vectorized, templated). Uses 128-bit (`ld128`/`st128`) or 256-bit (`ld256`/`st256`) loads on SM100+; falls back to scalar `VLLM_LDG` for unaligned `d`.
- Third-party dep: none.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/model_executor/layers/activation.py:189` | `SiluAndMulWithClamp.forward_cuda` (`self.op(out, x, self.swiglu_limit)`) | `x: [..., 2*d]`, `out: [..., d]` | bf16 / fp16 / fp32 (V4-Flash: bf16 from `gate_up_proj`) | `current_platform.is_cuda_alike()` AND NOT ROCm/XPU (lines 172-175). V4-Flash B200 → CUDA path. |
| `vllm/models/deepseek_v4/nvidia/model.py:110` | `DeepseekV4MLP.forward` (`self.act_fn(gate_up)`) — both shared experts and dense MLP | `gate_up: [T, 2 * intermediate_size]`, `out: [T, intermediate_size]` | bf16 | `swiglu_limit is not None` (V4-Flash: 10.0 from `config.swiglu_limit` per `inference/config.json:13`) — checked at `nvidia/model.py:103`. When None, falls through to plain `SiluAndMul`. |

On V4-Flash:
- **Shared experts** (`nvidia/model.py:473-481`): the always-on shared expert in MoE blocks. `intermediate_size = moe_intermediate_size * n_shared_experts = 2048 * 1 = 2048` → `d = 2048`, `2*d = 4096`.
- **Routed experts**: the experts inside MegaMoE use a different path (`fused_silu_mul_block_quant`); this kernel is NOT in the routed-expert hot path. V4-Flash routed experts are fp4 with a different fused activation (out of scope here).

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `out` | `[..., d]` (V4-Flash shared expert: `[T, 2048]`) | bf16 (matches `input.dtype`) | row-major, contiguous on `d`; one row per CTA (`blockIdx.x`) | Pre-allocated output buffer (`torch.empty(output_shape, dtype=x.dtype, device=x.device)` at `activation.py:188`) |
| `input` | `[..., 2*d]` (V4-Flash shared expert: `[T, 4096]`) | bf16 | row-major, contiguous on `2*d`; first half `[..., :d]` is **gate**, second half `[..., d:]` is **up** (matches PyTorch convention) | Output of `gate_up_proj` (a `MergedColumnParallelLinear` packing `w1 || w3` per expert) |
| `limit` (`swiglu_limit`) | scalar | double at the Python boundary, cast to `(float)limit` at line 270 then passed as `float LIMIT` to the kernel | — | V4-Flash: **10.0** (`config.swiglu_limit` from `inference/config.json:13`); zero/None would route to `SiluAndMul` instead |

Compile-time template constants picked by `LAUNCH_ACTIVATION_GATE_KERNEL`:
| Template param | Value | Selection |
| --- | --- | --- |
| `act_first` | `true` | gate is silu'd, up is the pure-mul multiplicand (per the `silu_and_mul_clamp` wrapper) |
| `HAS_CLAMP` | `true` | hard-wired for this entry point |
| `use_vec` | `(d % vec_size == 0)` | True for V4-Flash (`d=2048` divisible by 8 for bf16-128b and 16 for bf16-256b) |
| `use_256b` | `(CUDA_VERSION >= 12090 && cc_major >= 10 && num_tokens > 128)` | True on B200 with sufficient batch; False on H100/A100 |
| `scalar_t` | `bf16` / `fp16` / `fp32` | dispatched by `VLLM_STABLE_DISPATCH_FLOATING_TYPES` |
| `packed_t` | corresponding 2-wide vector type (bf16: `nv_bfloat162`; fp16: `half2`; fp32: `float2`) | via `PackedTypeConverter` |

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `out` | `[..., d]` (V4-Flash shared: `[T, 2048]`) | bf16 (= `input.dtype`) | row-major, contiguous | `silu(clamp(gate, max=L)) * clamp(up, -L, L)` per element; feeds `down_proj` (the `RowParallelLinear`) |

Early-exit shape contract: when `num_tokens == 0` the launcher returns without launching (lines 209-211).

## Grid / Block

- `grid_dim = (num_tokens,)` where `num_tokens = input.numel() / input.size(-1)`. **One CTA per row.**
- `block_dim`:
  - vectorized: `min(d / vec_size, 1024)` (line 224)
  - scalar fallback: `min(d, 1024)` (line 247)
  V4-Flash bf16 `d=2048`: vec_size = `support_vec / 2` = `16/2 = 8` (128b) or `32/2 = 16` (256b on SM100+). Block = `min(256, 1024) = 256` (128b) or `min(128, 1024) = 128` (256b).
- Autotune: NONE. Block size is chosen at launch from `d` and `vec_size`; no occupancy autotuner.
- Per-CTA work (vectorized path, lines 86-115):
  - Compute `x_ptr = input + blockIdx.x * 2 * d`, `y_ptr = x_ptr + d`, `out_ptr = out + blockIdx.x * d`.
  - Reinterpret as `PackedVec<cuda_t, use_256b>*`.
  - Strided loop: `for (int i = threadIdx.x; i < num_vecs; i += blockDim.x)` issues one `ld128` or `ld256` for `x` (gate) and one for `y` (up), then per `pvec_t::NUM_ELTS` packed elements calls `packed_compute<packed_t, packed_silu_kernel, act_first=true, HAS_CLAMP=true>` and writes back with `st128`/`st256`.
- Per-CTA work (scalar fallback, lines 117-124): `for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x)` reads one `scalar_t` from gate, one from up, calls `compute<scalar_t, silu_kernel, true, true>(x, y, limit)`, writes one scalar.
- No PDL / griddepcontrol hints in this kernel — fired as a standalone launch.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:596-606` (`Expert.forward`):
```
gate = w1(x).float()
up   = w3(x).float()
if swiglu_limit > 0:
    up   = torch.clamp(up,   min=-swiglu_limit, max=swiglu_limit)
    gate = torch.clamp(gate, max=swiglu_limit)
out  = F.silu(gate) * up        # cast back to input dtype handled at .w2 boundary
```

The vLLM kernel fuses the **bf16** version of these four ops (`gate.clamp(max=L)` → `silu` → `up.clamp(-L, L)` → `*`) into a single per-element pipeline so the bf16 row of `gate_up` is loaded exactly once. Reference does the math in fp32 and downcasts at the `w2` input; the kernel does it in fp32 only for the activation arithmetic (via `(float)x` casts inside `silu_kernel` / `packed_silu_kernel`) and stores back as bf16, matching the `Expert` semantics (the bf16 downcast is moved inside the kernel rather than at the `w2` input).

```python
# PyTorch-operator equivalent of one CTA's work for token-row t (vectorized path):
#
# Inputs:
#   input: [T, 2*d]  bf16     (V4-Flash shared expert: [T, 4096])
#   out:   [T,   d]  bf16     (V4-Flash shared expert: [T, 2048])
#   limit: float = 10.0       (V4-Flash swiglu_limit)
#
# Per-pair (gate, up) of bf16 elements (e.g. (nv_bfloat162, nv_bfloat162)):
gate_bf16, up_bf16 = input[t, :d], input[t, d:]            # contiguous halves
gate_f32 = gate_bf16.to(torch.float32)
up_f32   = up_bf16.to(torch.float32)

# packed_compute<...,act_first=true,HAS_CLAMP=true>:  (activation_kernels.cu:42-55)
gate_clamped = torch.clamp(gate_f32, max=limit)            # one-sided clamp
up_clamped   = torch.clamp(up_f32,   min=-limit, max=limit)  # two-sided clamp

# packed_silu_kernel  (activation_kernels.cu:133-140):
gate_silu = gate_clamped / (1.0 + torch.exp(-gate_clamped))  # x * sigmoid(x), via 1/(1+exp(-x)) form

# packed_mul:
result_f32 = gate_silu * up_clamped

# Single store at row t (bf16 narrow cast inside cast_to_packed):
out[t, :d] = result_f32.to(torch.bfloat16)
```

Fusion / numeric notes:
- **One round-trip through SMEM/regs**: the bf16 `gate || up` row is touched exactly once; the four PyTorch-level ops (`clamp(max=L)`, `silu`, `clamp(-L, L)`, `*`) are folded into a single per-element body inside `packed_compute`.
- **fp32 internal math**: the silu computes `((float)x) / (1.0f + expf((float)-x))` (line 130) — NOT `x * sigmoid(x)`; the implementation uses the algebraically equivalent `x / (1 + exp(-x))` so the `expf` argument is the positive-clamped gate value (no overflow for `gate ≤ 10.0`). The packed variant (lines 134-140) does the same per-lane.
- **Clamp asymmetry** matches reference: gate uses `fminf(x, limit)` only (line 22 / lines 47-49), up uses `fmaxf(fminf(x, limit), -limit)` (line 23 / lines 50-51). The reference at `model.py:601-602` is identical: `up = torch.clamp(up, min=-L, max=L); gate = torch.clamp(gate, max=L)`.
- **No fp8 quant fused**: this kernel does NOT quantize for the next FP8 GEMM. The shared expert's `down_proj` consumes bf16 directly. (The routed-expert path uses `fused_silu_mul_block_quant.cu` for fp4/fp8 quant — different kernel, not specced here.)
- **256-bit dispatch on SM100**: when CUDA ≥ 12.9, B200, and `num_tokens > 128`, the kernel uses `ld256`/`st256` for 32-byte vectorized loads (lines 97-103, 110-114). This halves the issued instruction count for `d=2048` bf16. Below the threshold (e.g., small decode batches), it uses 128-bit loads.
- **Scalar fallback is NOT for V4-Flash**: `use_vec = (d % vec_size == 0)` — V4-Flash's `d=2048` is divisible by 8 (128b) and 16 (256b), so always vectorized.

## Config-dependent dispatch

- **Activation condition**: V4-Flash, NVIDIA, SM100. Selected when `DeepseekV4MLP` is built with `swiglu_limit=10.0` from `config.swiglu_limit` (`nvidia/model.py:103-106`). On NVIDIA & not XPU (`activation.py:172-175`), uses the CUDA op; on XPU/ROCm uses `forward_native` (PyTorch `F.silu(torch.clamp(gate, max=L)) * torch.clamp(up, -L, L)` at `activation.py:179-183`).
- **Locked alternative pointer — `swiglustep_and_mul_triton` / `_swiglustep_and_mul_kernel`**: `vllm/model_executor/layers/activation.py:27-74`. A Triton-based clamped-SwiGLU variant computing `(silu(gate)).clamp(max=L) * up.clamp(-L, L)`. NOTE the semantic difference: that Triton kernel clamps `silu(gate)` AFTER the silu, while THIS kernel clamps `gate` BEFORE. **NOT used by V4-Flash MLP path** — no caller in `vllm/models/deepseek_v4/nvidia/`. Locked alternative; no separate spec.
- **Locked alternative pointer — `silu_and_mul` (no clamp)**: `activation_kernels.cu:259-264`; same templated kernel with `HAS_CLAMP=false`, `limit=0.0f`. Selected when `swiglu_limit is None` at `nvidia/model.py:106`. V4-Flash always has `swiglu_limit=10.0`, so unused on V4-Flash but used by other models with the same MLP class.
- **Locked alternative pointer — `fused_silu_mul_block_quant`**: `csrc/quantization/fused_kernels/fused_silu_mul_block_quant.cu`. Fuses silu+mul AND fp4/fp8 block quant for the routed expert path. Different I/O contract; not a drop-in alternative.
- **Downstream consumer constraints**:
  - `out.dtype == input.dtype` is enforced by the allocation at `activation.py:188`. The kernel's `VLLM_STABLE_DISPATCH_FLOATING_TYPES` resolves `scalar_t` from `dtype = input.scalar_type()` (line 206) and writes the same dtype. **Cannot mix dtypes** between input and output.
  - `out.size(-1) == input.size(-1) / 2 == d` (kernel splits `input` at the midpoint via `x_ptr = input + blockIdx.x * 2 * d`, `y_ptr = x_ptr + d`).
  - For the vectorized path: `d * sizeof(scalar_t)` must be divisible by `vec_size * sizeof(scalar_t)`. For V4-Flash bf16 `d=2048`: ✓ for both 128b (8-elt) and 256b (16-elt).
- **Hard preconditions**:
  - `input.size(-1) % 2 == 0` (implicit; `d = input.size(-1) / 2`).
  - `input` and `out` on the same device.
  - `input.dtype` must be in the `VLLM_STABLE_DISPATCH_FLOATING_TYPES` set (bf16/fp16/fp32). V4-Flash uses bf16 from `gate_up_proj`.
  - `limit > 0` is NOT enforced inside the kernel; passing `limit ≤ 0` would clamp everything to that value (the gate's `fminf(x, limit)` and the up's `fmaxf(fminf(x, limit), -limit)` are both well-defined). V4-Flash always passes 10.0.
