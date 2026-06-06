# prepare_megamoe_inputs

## Identity
- Source file: `vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py:15-115` (Triton JIT body `_prepare_megamoe_inputs_kernel`); user-facing wrapper `prepare_megamoe_inputs` at lines 118-173.
- Language/DSL: **Triton** (`@triton.jit`).
- Third-party dep: none (pure Triton + PyTorch).
- Registered as: Python-level function (no `torch.ops` op); module-level export from `vllm.models.deepseek_v4.nvidia.ops.prepare_megamoe`.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/nvidia/model.py:366` | `DeepseekV4MegaMoEExperts._run_mega_moe` | `hidden_states: [T, H=hidden_size]`, `topk_weights: [T, top_k]`, `topk_ids: [T, top_k]` (int32 or int64) | bf16 hidden, fp32 weights, int32/int64 ids | `vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"` (model.py:408); requires `--enable-expert-parallel` (model.py:410-415) |

Single call site in production. Runs once per MoE-layer forward, immediately after `fused_topk_bias` produces `(topk_weights, topk_ids)` from the gate output, and before `deep_gemm.fp8_fp4_mega_moe` consumes the symmetric-memory dispatch buffers it populated.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `hidden_states` | `[num_tokens, hidden_size]` (V4-Flash: `hidden_size=7168`) | bf16 | row-major; arbitrary strides `(hidden_stride_m, hidden_stride_k)` | Per-token activation entering the MoE layer (post attention residual + RMSNorm) |
| `topk_ids` | `[num_tokens, top_k]` (V4-Flash: `top_k=8` for routed experts) | int32 (FusedMoE path) or int64 (MegaMoE path; `hash_indices_dtype` at model.py:448) | row-major | Per-token routed expert ids from `fused_topk_bias` (`-1` denotes "no-route") |
| `topk_weights` | `[num_tokens, top_k]` | fp32 | row-major | Per-token routed expert weights (already renormalized & scaled by `routed_scaling_factor`) |
| `x_fp8` (out alias) | `[num_tokens, hidden_size]` | float8_e4m3fn | row-major slice of `symm_buffer.x[:num_tokens]` | FP8 quantized hidden states for dispatch |
| `x_sf` (out alias) | `[num_tokens, hidden_size // BLOCK_K]` (V4-Flash: `7168 // 128 = 56`) | int32 (packed UE8M0) | row-major slice of `symm_buffer.x_sf[:num_tokens]` | Packed UE8M0 group scales for `x_fp8` |
| `topk_idx_out` (out alias) | `[num_tokens, top_k]` | int64 | row-major slice of `symm_buffer.topk_idx[:num_tokens]` | Repacked top-k expert ids in the int64 layout DeepGEMM consumes |
| `topk_weights_out` (out alias) | `[num_tokens, top_k]` | fp32 | row-major slice of `symm_buffer.topk_weights[:num_tokens]` | Repacked top-k weights for DeepGEMM |
| `hidden_size` | scalar | `tl.constexpr` int | — | `hidden_states.shape[1]`; wrapper asserts `% 128 == 0` (line 130) |
| `top_k` | scalar | `tl.constexpr` int | — | `topk_ids.shape[1]` baked at JIT time |
| `BLOCK_K` | scalar | `tl.constexpr` int | — | Hard-coded to 128 by wrapper (line 142); one CTA per 128-element K-chunk |
| `GROUP_K` | scalar | `tl.constexpr` int | — | Hard-coded to 32 by wrapper (line 170); per-group UE8M0 scale span within `BLOCK_K` (so 4 groups per CTA) |
| `BLOCK_TOPK` | scalar | `tl.constexpr` int | — | `triton.next_power_of_2(top_k)` (line 144); single-tile load width for the topk repack |

## Outputs

All outputs are written in-place into pre-allocated slices of the DeepGEMM `SymmBuffer` (see `vllm/third_party/deep_gemm/mega/__init__.py:16-48`); the wrapper takes them as input tensors and returns `None`.

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `x_fp8` (`symm_buffer.x[:num_tokens]`) | `[num_tokens, hidden_size]` | float8_e4m3fn | row-major; symmetric-memory backed | FP8 dispatch buffer consumed by `deep_gemm.fp8_fp4_mega_moe`'s L1 activations TMA load |
| `x_sf` (`symm_buffer.x_sf[:num_tokens]`) | `[num_tokens, hidden_size // 128]` | int32 | row-major; symmetric-memory backed | Packed UE8M0 scales (4 group exponents per int32 — one int32 per 128-element K chunk) |
| `topk_idx_out` (`symm_buffer.topk_idx[:num_tokens]`) | `[num_tokens, top_k]` | int64 | row-major; symmetric-memory backed | Top-k expert ids in the int64 layout expected by `sm100_fp8_fp4_mega_moe_impl`'s dispatch warps (the `__ldg(input_topk_idx_buffer.get_base_ptr<int64_t>() + ...)` load at impls/sm100_fp8_fp4_mega_moe.cuh:377) |
| `topk_weights_out` (`symm_buffer.topk_weights[:num_tokens]`) | `[num_tokens, top_k]` | fp32 | row-major; symmetric-memory backed | Top-k weights for the combine epilogue |

Early-return contract: when `num_tokens == 0` the wrapper returns immediately (lines 128-129) without launching.

## Grid / Block

- `grid_dim = (num_tokens, triton.cdiv(hidden_size, BLOCK_K))` — one CTA per (token, K-chunk). V4-Flash: `(T, 56)`.
- `block_dim` (threads/CTA): `num_warps=4` → 128 threads/CTA (line 172).
- Autotune configs: **none** — `BLOCK_K=128`, `GROUP_K=32`, `BLOCK_TOPK=next_power_of_2(top_k)` baked at launch. No `num_stages` override.
- Per-CTA work: loads a `BLOCK_K=128` slab of the token's hidden vector, computes 4 group-wise absmax → UE8M0 scales (`num_groups = BLOCK_K // GROUP_K = 4`), quantizes to FP8 E4M3, writes the FP8 slab and a single int32 packed-scale word. The CTA whose `k_block_id == 0` additionally copies that token's top-k ids (cast int→int64) and top-k weights into the symmetric buffer.
- No int64 index trick at the per-CTA level: `token_id = tl.program_id(0)` stays in int32. Stride math is done at fp8 width (1 byte) and int32 width (4 bytes), so for V4-Flash sizes (T ≤ tens of thousands, hidden = 7168) the strided pointer arithmetic does not overflow.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:630-633` (`MoE.forward` calls `weights, indices = self.gate(x, ...)` and then dispatches `x` to per-expert FFNs). The reference passes `x` in bf16 directly to `Expert.forward`; the FP8 group-quant + UE8M0 packed-scale step is NOT in the reference — it is a vLLM-side optimization to feed DeepGEMM's FP8×FP4 MMA, with semantics matching `vllm/models/deepseek_v4/common/ops/fused_inv_rope_fp8_quant.py`'s UE8M0 convention (extract the IEEE-754 exponent of `absmax / fp8_max` and round up if the mantissa is nonzero).

```python
# PyTorch-operator equivalent of one CTA's work for token t, k-block kb:
#
# Inputs:
#   hidden_states: [T, H] bf16  (H = 7168 for V4-Flash)
#   topk_ids:      [T, top_k] int32/int64
#   topk_weights:  [T, top_k] fp32
#
# Constants: BLOCK_K=128, GROUP_K=32, num_groups = BLOCK_K // GROUP_K = 4
hidden = hidden_states[t, kb*BLOCK_K : (kb+1)*BLOCK_K].to(torch.float32)   # [BLOCK_K]
groups = hidden.view(num_groups, GROUP_K)                                  # [4, 32]

# Per-group UE8M0 scale (with mantissa-nonzero round-up):
amax = groups.abs().amax(dim=-1).clamp_min(1e-4)                           # [4]
scale = amax / 448.0                                                       # fp8_e4m3_max = 448
scale_bits = scale.view(torch.int32)                                       # bitcast
scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(torch.int32)
scale_exp = scale_exp.clamp(1, 254)                                        # avoid subnorm/inf
rounded_scale = (scale_exp << 23).view(torch.float32)                      # 2^k power of two

# FP8 quantize:
scaled = groups * (1.0 / rounded_scale).unsqueeze(-1)                      # [4, 32]
fp8 = scaled.view(BLOCK_K).to(torch.float8_e4m3fn)
x_fp8[t, kb*BLOCK_K : (kb+1)*BLOCK_K] = fp8

# Pack 4 UE8M0 exponent bytes into one int32 (one CTA → one int32 word):
#   packed = scale_exp[0] | (scale_exp[1] << 8) | (scale_exp[2] << 16) | (scale_exp[3] << 24)
packed = (scale_exp << (torch.arange(num_groups) * 8)).sum().to(torch.int32)
x_sf[t, kb] = packed

# Once per token (CTA with kb == 0): repack topk into int64 + fp32 in dispatch layout.
if kb == 0:
    topk_idx_out[t, :top_k]     = topk_ids[t, :top_k].to(torch.int64)
    topk_weights_out[t, :top_k] = topk_weights[t, :top_k]
```

Notes on fusion / quant:
- **UE8M0 packing**: DeepGEMM's `sm100_fp8_fp4_mega_moe.cuh:201` reads `SMEM_SFA_SIZE_PER_STAGE = SF_BLOCK_M * sizeof(uint32_t)` and the SF-transpose helper at `vllm/third_party/deep_gemm/mega/__init__.py:87-93` (`_transpose_sf_for_utccp`) operates on `dtype == torch.int`. The packed-int32 layout produced here is precisely the layout the UTCCP scale path inside the cutlass kernel consumes — no further reshuffling on the activations side.
- **Round-up rule**: this kernel uses `((scale_bits & 0x7FFFFF) != 0)` to bump the exponent when the mantissa is nonzero — i.e. it rounds `scale` *up* to the next power of two. Contrast `fused_inv_rope_fp8_quant`, which uses `exp2(ceil(log2(x)))` for the same effect; both reach a UE8M0-representable scale ≥ the true `absmax/fp8_max`. The `clamp(1, 254)` guards against the subnormal/`inf` exponent codes UE8M0 cannot encode.
- **Topk repack as a side-channel**: the topk write is gated on `k_block_id == 0` so exactly one CTA per token issues it. This avoids a second launch just to memcpy topk into the symmetric buffer. The `int → int64` cast at line 93 is the only typed transform; weights are byte-copies.
- **Shape contract for `top_k`**: `BLOCK_TOPK = next_power_of_2(top_k)` and a `topk_mask = offsets < top_k` mask gates both load and store, so `top_k` need not be a power of two.

## Config-dependent dispatch

- Activation condition: **active iff `moe_backend == "deep_gemm_mega_moe"`** (`vllm_config.kernel_config.moe_backend`, checked at `vllm/models/deepseek_v4/nvidia/model.py:407-409`).
- Hard constraints enforced upstream (model.py:410-435):
  - `--enable-expert-parallel` REQUIRED (`NotImplementedError` at line 411 otherwise).
  - `scoring_func == "sqrtsoftplus"` required (`NotImplementedError` at line 427 otherwise).
  - `config.expert_dtype == "fp4"` required (`NotImplementedError` at line 431 otherwise).
- **V4-Flash-Base compatibility caveat**: V4-Flash-Base ships `expert_dtype="fp8"` (see `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:41`), so the MegaMoE path would error at MoE init with `expert_dtype='fp8'` against the constraint above. The MegaMoE path is the canonical vLLM fast path for a hypothetical FP4 checkpoint; on the shipped V4-Flash-Base the FusedMoE branch (see `fused_moe_kernel.md`) is what runs.
- Wrapper preconditions (lines 130-140):
  - `hidden_size % 128 == 0` (`ValueError` at line 131 otherwise).
  - `topk_weights.shape == topk_ids.shape` (`ValueError` at line 137 otherwise).
- Downstream consumer constraints: outputs land in DeepGEMM's `SymmBuffer.x / .x_sf / .topk_idx / .topk_weights` and are immediately consumed by `deep_gemm.fp8_fp4_mega_moe` (model.py:382). Specifically the int32 packed-scale layout written here matches `sm100_fp8_fp4_mega_moe_impl`'s `fp8_sf_layout = layout::Data(kHidden / 32)` view (impls/sm100_fp8_fp4_mega_moe.cuh:101) and the int64 topk layout matches `input_topk_idx_layout = layout::Data(kNumTopk * sizeof(int64_t), false)` (impls/sm100_fp8_fp4_mega_moe.cuh:103). Any spec change to the packing order or scale rounding rule would silently corrupt the GEMM input.
