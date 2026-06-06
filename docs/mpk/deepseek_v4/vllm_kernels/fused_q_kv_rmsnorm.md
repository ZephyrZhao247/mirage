# fused_q_kv_rmsnorm

## Identity
- Source file: `vllm/models/deepseek_v4/common/ops/fused_qk_rmsnorm.py:8-54` (Triton JIT body `_fused_q_kv_rmsnorm_kernel`); user-facing wrapper `fused_q_kv_rmsnorm` at lines 57-96.
- Language/DSL: **Triton** (`@triton.jit`).
- Third-party dep: none (pure Triton + PyTorch).
- Registered as: Python-level function (no `torch.ops` op); imported from `vllm.models.deepseek_v4.common.ops` via `__init__.py:13,24`.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/attention.py:422` | `DeepseekV4MultiHeadLatentAttentionWrapper.attention_impl` | `qr: [T, q_lora_rank=1024]`, `kv: [T, head_dim=512]`, weights `[1024]` and `[512]` | bf16 activations; fp32 weights | always on; runs immediately after `qr_kv.split([q_lora_rank, head_dim], dim=-1)` (attention.py:421) |
| `tests/kernels/core/test_fused_q_kv_rmsnorm.py:44,67` | Correctness + large-`num_tokens` launch tests | random `[T, q_size]`, `[T, kv_size]` | bf16 (param) | test-only |

Single call site in production. Runs once per attention-layer forward, downstream of the fused W_qa+W_kv GEMM and upstream of `wq_b` (Q-LoRA projection) / `_fused_qnorm_rope_kv_insert` (KV path).

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `qr` (`q_ptr`) | `[num_tokens, Q_SIZE = q_lora_rank]` (V4-Flash: `Q_SIZE=1024`) | bf16 (DeepSeek V4-Flash) | row-major; `stride(-1) == 1`; arbitrary `stride(0)` (passed as `q_in_stride`) | Q-LoRA hidden state from `fused_wqa_wkv` slice |
| `q_weight` | `[Q_SIZE]` | fp32 | contiguous | RMSNorm gain for Q stream (`self.q_norm.weight.data`) |
| `kv` (`kv_ptr`) | `[num_tokens, KV_SIZE = head_dim]` (V4-Flash: `KV_SIZE=512`) | bf16 | row-major; `stride(-1) == 1`; arbitrary `stride(0)` | KV latent from `fused_wqa_wkv` slice |
| `kv_weight` | `[KV_SIZE]` | fp32 | contiguous | RMSNorm gain for KV stream (`self.kv_norm.weight.data`) |
| `eps` | scalar | float | — | `config.rms_norm_eps` (passed as `self.eps`) |
| `Q_SIZE` | scalar | int (`tl.constexpr`) | — | `qr.shape[1]` baked at JIT time |
| `KV_SIZE` | scalar | int (`tl.constexpr`) | — | `kv.shape[1]` baked at JIT time |
| `BLOCK_SIZE` | scalar | int (`tl.constexpr`) | — | `triton.next_power_of_2(max(Q_SIZE, KV_SIZE))` (V4-Flash: 1024); single-tile load |

Strides `q_in_stride`, `q_out_stride`, `kv_in_stride`, `kv_out_stride` are passed as runtime ints. `qr_out`/`kv_out` are allocated via `torch.empty_like` so their `stride(0)` equals `Q_SIZE`/`KV_SIZE`.

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `qr_out` (`q_out_ptr`) | `[num_tokens, Q_SIZE]` | bf16 (matches `qr.dtype`) | `torch.empty_like(qr)` — same strides as `qr` | Q-LoRA hidden state after RMSNorm, ready for `wq_b` |
| `kv_out` (`kv_out_ptr`) | `[num_tokens, KV_SIZE]` | bf16 (matches `kv.dtype`) | `torch.empty_like(kv)` | KV latent after RMSNorm, ready for `_fused_qnorm_rope_kv_insert` |

Early-return shape contract: when `num_tokens == 0` the wrapper returns the empty allocations without launching (lines 76-77).

## Grid / Block

- `grid_dim = (num_tokens, 2)` — outer dim is the token, inner dim selects Q (`pid_task==0`) vs KV (`pid_task==1`).
- `block_dim`: default Triton `num_warps` (= 4 for `BLOCK_SIZE=1024`); no explicit `num_warps`/`num_stages` overrides in the launch.
- Autotune configs: **none** — `BLOCK_SIZE` is computed from inputs at launch and baked as `tl.constexpr`. Single tile per CTA covers `max(Q_SIZE, KV_SIZE)` lanes with a boolean mask `block < SIZE` excluding lanes past the per-task `SIZE`.
- Per-CTA work: one CTA handles a single (token, task) pair end-to-end. Loads `SIZE` bf16 values, computes fp32 variance via `tl.sum(x*x)`, applies `rsqrt(var + eps) * w`, and stores back as `row_out.dtype.element_ty` (bf16).
- Int64 index trick: `token_idx = tl.program_id(0).to(tl.int64)` (line 30). The comment at lines 25-29 explains: (a) grid-y/z cap at 65535 forces `num_tokens` onto grid-x so chunked-prefill `T ≥ 65536` doesn't crash launch; (b) `q_in_stride` can be ~24K (128 heads × 192) and `token_idx * q_in_stride` overflows int32 once `T ≳ 87K`. Test `tests/kernels/core/test_fused_q_kv_rmsnorm.py:55-67` exercises this at `T = 65536`.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:496` (`qr = q = self.q_norm(self.wq_a(x))`) and `model.py:503` (`kv = self.kv_norm(kv)`). The kernel jointly executes BOTH RMSNorms (same per-token grid row, dispatched by `pid_task`); the reference applies them sequentially. The fused fp32 reduction → bf16 store pattern matches `csrc/layernorm_kernels.cu`'s `(scalar_t)(x * s_variance * w)` convention (kernel comment lines 44-46).

```python
# PyTorch-operator equivalent of one CTA's work for token t, task ∈ {0, 1}:
#
# Inputs:
#   qr:        [T, Q_SIZE]  bf16    (Q_SIZE = q_lora_rank, e.g. 1024)
#   kv:        [T, KV_SIZE] bf16    (KV_SIZE = head_dim,  e.g. 512)
#   q_weight:  [Q_SIZE]     fp32
#   kv_weight: [KV_SIZE]    fp32
#
# pid_task == 0 → Q branch; pid_task == 1 → KV branch.
if pid_task == 0:
    SIZE = Q_SIZE
    x_in  = qr[t, :SIZE].to(torch.float32)
    w     = q_weight.to(torch.float32)
else:
    SIZE = KV_SIZE
    x_in  = kv[t, :SIZE].to(torch.float32)
    w     = kv_weight.to(torch.float32)

# Mask lanes beyond SIZE: the kernel uses a single BLOCK_SIZE-wide tile and
# masks both load and store with `block < SIZE`. Lanes loaded as 0 do not
# contribute to the variance sum.
variance = (x_in * x_in).sum() / SIZE              # scalar fp32
rrms     = torch.rsqrt(variance + eps)             # scalar fp32
y        = x_in * rrms * w                         # [SIZE] fp32

# Single cast at store — matches DeepseekV4 compressor convention.
if pid_task == 0:
    qr_out[t, :SIZE] = y.to(qr_out.dtype)          # bf16
else:
    kv_out[t, :SIZE] = y.to(kv_out.dtype)          # bf16
```

Notes on fusion / quant:
- The "joint" fusion is at the **grid** level, not the **block** level: a single launch covers both norms with one PTX module, halving launch overhead vs two separate norms. The two tasks read from disjoint pointers, write to disjoint pointers, and share only `eps`, `t`, and `BLOCK_SIZE`.
- `BLOCK_SIZE = next_power_of_2(max(Q_SIZE, KV_SIZE))`. When `Q_SIZE > KV_SIZE` (V4-Flash: 1024 > 512), the KV branch wastes half its lanes — but a single shared `BLOCK_SIZE` is required because Triton bakes the constexpr at JIT time. This is intentional (one kernel, one PTX, two specializations would defeat the launch-fusion goal).
- No FP8 quant; bf16 in, bf16 out. The accumulator is fp32 throughout (comment lines 44-46). The cast happens exactly once at the store via `y.to(row_out.dtype.element_ty)`.

## Config-dependent dispatch

- Activation condition: always on (DeepseekV4 MLA wrapper, NVIDIA and ROCm).
- Variants: none. The kernel is platform-agnostic Triton; no SM90/SM100 split, no Class A/B branch.
- Downstream consumer constraints:
  - `qr_out` feeds `self.wq_b(qr)` at attention.py:442/476/489 (column-parallel linear, FP8-weighted) and the indexer / compressor at attention.py:453,461. Layout requirement: `[T, q_lora_rank]` bf16 row-major contiguous on dim 1 (matches `torch.empty_like(qr)`).
  - `kv_out` feeds `torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert` at attention.py:531-541 as the second positional arg (`kv`). The CUDA op asserts `kv.is_contiguous()` and `kv.size(1) == 512` (kernel `TORCH_CHECK` at `csrc/fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu:673`), so `KV_SIZE` here must equal 512 for V4-Flash.
- Hard preconditions (wrapper asserts at lines 64-69):
  - `qr.ndim == 2 and kv.ndim == 2`
  - `qr.shape[0] == kv.shape[0]` (token count must match)
  - `qr.stride(-1) == 1 and kv.stride(-1) == 1` (contiguous last dim)
  - `q_weight.is_contiguous() and kv_weight.is_contiguous()`
