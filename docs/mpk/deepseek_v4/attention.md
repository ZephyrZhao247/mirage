# DeepSeek V4-Flash — MPK Attention Pipeline Spec

**Scope.** This document specifies the 5 MPK tasks required to port the
DeepSeek V4-Flash attention pipeline (excluding the Compressor and Indexer,
which are covered in `docs/mpk/deepseek_v4/sparse.md`). It is the Wave-1
spec; **no code is written** by this document. All claims cite the official
PyTorch reference at
`deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py` and the vLLM
production path at
`deps/vllm/vllm/model_executor/layers/deepseek_v4_attention.py`. The plan
governing this spec is `/home/zepengz/.claude/plans/i-want-to-add-dapper-pascal.md`
(see "Mandatory kernel-authoring requirements", binding).

The five tasks specified here are:

| # | Task name | Layer file (`.cuh`) | When it runs |
|---|---|---|---|
| 1 | `mla_v4_q_kv_rmsnorm_layer`     | `mla_v4_q_kv_rmsnorm_sm100.cuh`     | Every layer, prefill + decode |
| 2 | `mla_v4_decode_layer`           | `mla_v4_decode_sm100.cuh`           | Decode step (Q_LEN==1) for ratio ∈ {0, 4, 128} |
| 3 | `mla_v4_prefill_layer`          | `mla_v4_prefill_sm100.cuh`          | Prefill step over gathered-KV workspace |
| 4 | `mla_v4_prefill_gather_layer`   | `mla_v4_prefill_gather_sm100.cuh`   | Prefill, before #3 |
| 5 | `inv_rope_fp8_quant_o_layer`    | `inv_rope_fp8_quant_o_sm100.cuh`    | Every layer, after attention; feeds wo_a FP8 GEMM |

The four sparse/cache tasks (`compressor`, `indexer_q_transform`,
`indexer_score_topk`, plus FP8 paged-cache insert) live in `sparse.md`.
This spec assumes those produce the `compressed_kv_cache` and
`topk_indices_buffer` consumed by tasks 2 and 3 below.

---

## 0. Verified dimensions (Flash-Base config.json)

Pulled from `/raid/catalyst/models/DeepSeek-V4-Flash-Base/config.json` and
cross-checked against `Attention.__init__` in
`deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:436-482`.

| Symbol | Value | Source |
|---|---|---|
| `H`           = `num_attention_heads`    | **64**   | config.json `num_attention_heads` |
| `D`           = `head_dim`                | **512**  | config.json `head_dim` |
| `D_rope`      = `qk_rope_head_dim`        | **64**   | config.json `qk_rope_head_dim` |
| `D_nope`      = `head_dim - rope`         | **448**  | derived; see `model.py:449` `self.nope_head_dim = args.head_dim - args.rope_head_dim` |
| `q_lora_rank`                              | **1024** | config.json `q_lora_rank`, `model.py:457` |
| **`kv_lora_rank`**                         | **512 = `head_dim`** | NOTE: no `kv_lora_rank` field in V4-Flash config.json. V4 uses a **single 512-wide KV latent** that doubles as the K/V cache row. `model.py:460` `self.wkv = Linear(self.dim, self.head_dim)` and `model.py:474` `kv_cache_size×head_dim`. vLLM confirms this by passing `kv_lora_rank=self.head_dim` (`deps/vllm/vllm/model_executor/models/deepseek_v4.py:1069`). |
| `o_lora_rank`                              | **1024** | config.json `o_lora_rank`, `model.py:446` |
| `o_groups`                                 | **8**    | config.json `o_groups`, `model.py:450` |
| `num_key_value_heads`                      | **1**    | MLA (single shared latent K/V row per token) |
| `sliding_window` (W)                       | **128**  | config.json `sliding_window`, `model.py:452` |
| `rope_theta` (base)                        | **10000** | config.json `rope_theta` |
| `compress_rope_theta`                      | **160000** | config.json (used by Compressor, NOT this spec; included for completeness — see `model.py:476`) |
| `max_position_embeddings`                  | **1048576** | config.json `max_position_embeddings` |
| `eps`                                      | **rms_norm_eps** from config | `model.py:454` `self.eps = args.norm_eps` |

Per `model.py:464` `softmax_scale = head_dim ** -0.5 = 1/sqrt(512)`.

Symbolic shorthand used in this spec:
- `T` — number of tokens in the current step (= 1 per request during decode,
  = prompt length during prefill).
- `B` — number of active requests.
- `S_total` — gathered KV-cache length for a request (≤ W + N where
  `N = ceil(seq_len / compress_ratio)`).

---

## 1. `mla_v4_q_kv_rmsnorm_layer` — joint Q-lora + KV-lora RMSNorm

### 1.1 When it runs

Every attention layer, both prefill and decode. It is the first task on the
attention critical path inside a Block, called **after** the fused
`wqa+wkv` GEMM splits its output into the Q-lora stream (`qr[T,
q_lora_rank=1024]`) and the KV-lora stream (`kv[T, head_dim=512]`).

### 1.2 Math

For each token row `t` and for each of the two streams independently
(stream `S` ∈ {q, kv} with width `W_S` ∈ {1024, 512}):

```
x_f32     = x.float()                                # cast bf16→f32 once
var       = mean(x_f32^2, axis=last)                 # scalar per row
rrms      = rsqrt(var + eps)
y_f32     = (x_f32 * rrms) * weight_f32              # weight is fp32 per channel
y_bf16    = bf16_cast(y_f32)
```

Both streams are normalized with **independent learnable per-channel
weights** (`q_norm.weight[1024]`, `kv_norm.weight[512]`) and a shared `eps`.
This is the standard RMSNorm, but vLLM and we **fuse the two streams into
one kernel** so the per-token global reduction can be issued once per token
with grid-y = 2 selecting stream. Quote from vLLM's Triton implementation
(`deps/vllm/vllm/v1/attention/ops/deepseek_v4_ops/fused_qk_rmsnorm.py:25-54`):

```python
# num_tokens goes on grid-x (max 2**31 - 1); task goes on grid-y.
# CUDA's grid-y/z are capped at 65535, so putting num_tokens there crashes
# the launch at max-num-batched-tokens >= 65536 with "invalid argument".
token_idx = tl.program_id(0).to(tl.int64)
pid_task  = tl.program_id(1)
if pid_task == 0:
    SIZE = Q_SIZE
    row_in = q_ptr + token_idx * q_in_stride
    weight_ptr = q_weight_ptr
    row_out = q_out_ptr + token_idx * q_out_stride
else:
    SIZE = KV_SIZE
    row_in = kv_ptr + token_idx * kv_in_stride
    weight_ptr = kv_weight_ptr
    row_out = kv_out_ptr + token_idx * kv_out_stride
# RMSNorm in fp32 throughout — matches csrc/layernorm_kernels.cu's
# `(scalar_t)(x * s_variance * w)` and DeepseekV4's compressor kernel
block = tl.arange(0, BLOCK_SIZE)
mask  = block < SIZE
x     = tl.load(row_in + block, mask=mask, other=0.0).to(tl.float32)
variance = tl.sum(x * x, axis=0) / SIZE
rrms     = tl.rsqrt(variance + eps)
w        = tl.load(weight_ptr + block, mask=mask, other=0.0).to(tl.float32)
y        = x * rrms * w
tl.store(row_out + block, y.to(row_out.dtype.element_ty), mask=mask)
```

This is identical to applying `RMSNorm.forward` (`model.py:191-196`)
separately to `q_lora` and `kv_lora`. Reference call site is
`deps/vllm/vllm/model_executor/layers/deepseek_v4_attention.py:409-415`:

```python
qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
qr, kv = fused_q_kv_rmsnorm(
    qr,
    kv,
    self.q_norm.weight.data,
    self.kv_norm.weight.data,
    self.eps,
)
```

### 1.3 Inputs / Outputs / State

| Tensor | Shape | Dtype | Source |
|---|---|---|---|
| **inputs** | | | |
| `qr_in`         | `[T, q_lora_rank=1024]` | bf16 | wqa output, dense linear (V3 `linear_fp8_layer`) |
| `kv_in`         | `[T, head_dim=512]`     | bf16 | wkv output, dense linear |
| `q_weight`      | `[q_lora_rank=1024]`    | fp32 | persistent param `q_norm.weight` (`model.py:458`) |
| `kv_weight`     | `[head_dim=512]`        | fp32 | persistent param `kv_norm.weight` (`model.py:461`) |
| **outputs** | | | |
| `qr_out`        | `[T, 1024]`             | bf16 | feeds `wq_b` (next dense linear) |
| `kv_out`        | `[T, 512]`              | bf16 | feeds GPT-J RoPE on last `D_rope=64` dims + FP8 cache-insert (sparse.md) and is the KV row used by `mla_v4_decode` / `mla_v4_prefill` (last 64 are RoPE-applied) |

Per-channel `q_weight` and `kv_weight` are read-only persistent parameters
loaded once at model init. No paged-cache reads or writes; this kernel
performs no rope, no quant, no cache insert (those happen in
`_fused_qnorm_rope_kv_insert`, in sparse.md / decode side).

`eps` is a compile-time scalar param (configurable per registration).

### 1.4 Source to migrate

Primary: `deps/vllm/vllm/v1/attention/ops/deepseek_v4_ops/fused_qk_rmsnorm.py:8-96`
(Triton). Quoted in full above (lines 25-54).

PyTorch reference for the per-stream math: `model.py:191-196` (`RMSNorm.forward`).

### 1.5 Generated CUDA reference

`docs/mpk/deepseek_v4/_generated_cuda/mla_v4_q_kv_rmsnorm.cu` — produced by
Wave-1 TileLang/Triton dump harness. Since the source is Triton (not
TileLang), the harness invokes Triton AOT on a representative shape
(`T=128`, `Q_SIZE=1024`, `KV_SIZE=512`) and dumps the generated PTX +
constant CUDA scaffolding. The reader of this spec uses it as a hand-port
target; the kernel is simple enough that a from-scratch port keyed off the
Triton DSL above is equivalent.

### 1.6 Grid design

**Natural grid.** `(T, 2)` — one CTA per (token, stream). Each CTA owns a
single row of one stream (either 1024 or 512 elements), reduces it in
shared memory, and writes it back. The Triton kernel already takes exactly
this shape (`_fused_q_kv_rmsnorm_kernel[(num_tokens, 2)]`,
`fused_qk_rmsnorm.py:80`).

**Block dim.** 128 threads is sufficient: a single warp of 32 threads can
already cover a 1024-wide row in 32-element strides; 128 threads gives 4×
unrolled coverage at 8-element stride and lets the per-row variance
reduction fit one warp-shuffle pass. (The V3 `rmsnorm_layer` uses 128-thread
blocks for similar row widths — see `persistent_kernel.py:1015-1031`.)

**MPK partitioning.** The MPK runtime dispatches each grid slot to an
arbitrary worker thread block. We follow V3's `rmsnorm_hopper` /
`rmsnorm` pattern (`persistent_kernel.py:1026-1031`) where:

- The CTA reads its `(token_idx, stream_idx)` slice from
  `task_desc->task_metadata` (see §1.7) — **NOT** from `blockIdx`.
- Input maps in `tb_graph.new_input`:
  - `qr_in`, `kv_in`: split on dim 0 by `token_idx`; full row per CTA.
  - `q_weight`, `kv_weight`: replicated.
  - `qr_out`, `kv_out`: split on dim 0 by `token_idx`.

We extend V3's single-stream `rmsnorm_layer` only by making grid-y select
the stream (i.e., the CTA's metadata carries a `stream_idx ∈ {0, 1}` field;
`stream_idx == 0` ⇒ Q path with width 1024 and weight `q_weight`;
`stream_idx == 1` ⇒ KV path with width 512 and weight `kv_weight`).

**Alignment constraints.** `T` need not be aligned (each CTA handles
exactly one token). Row widths 1024 and 512 are both multiples of 128, so
vectorized loads at 16B granularity (8 bf16) work without tail handling.

### 1.7 `/add-mpk-task` conformance checklist

- [ ] `blockIdx`-agnostic. The kernel reads `task_desc->task_metadata.token_id`
  and `task_desc->task_metadata.stream_idx`, **not** `blockIdx.{x,y,z}`. The
  enclosing `TaskType` will be `MLA_V4_Q_KV_RMSNORM_SM100`.
- [ ] `TaskType` enum entry added in
  `include/mirage/persistent_kernel/runtime_header.h`. New value
  `TASK_MLA_V4_Q_KV_RMSNORM_SM100`. Numbered after the last V3 enum value;
  see `runtime_header.h` near the existing `TASK_RMSNORM_*` entries.
- [ ] `register_<task>_task` codegen entry added to
  `src/kernel/task_register.cc`. Pattern: `register_rmsnorm_task` at
  `task_register.cc:91`. Signature accepts the 6 inputs (`qr_in`, `kv_in`,
  `q_weight`, `kv_weight`, `qr_out`, `kv_out`) and the `eps` param.
- [ ] Task-name → register-function dispatch added in `src/kernel/graph.cc`.
  Pattern: search for `"rmsnorm_hopper"` in graph.cc; add
  `"mla_v4_q_kv_rmsnorm_sm100"` next to it.
- [ ] Layer-method pattern in `python/mirage/mpk/persistent_kernel.py`
  mirrors V3 `rmsnorm_layer` (`persistent_kernel.py:1015-1031`) with the
  extra inputs and grid-y=2:

  ```python
  def mla_v4_q_kv_rmsnorm_layer(
      self,
      qr_in: DTensor, kv_in: DTensor,
      q_weight: DTensor, kv_weight: DTensor,
      qr_out: DTensor, kv_out: DTensor,
      eps: float,
      grid_dim: tuple, block_dim: tuple,
  ):
      tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
      tb_graph.new_input(qr_in,      (0, -1, -1), 1, True)
      tb_graph.new_input(kv_in,      (0, -1, -1), 1, True)
      tb_graph.new_input(q_weight,   (-1, -1, -1), 0, True)
      tb_graph.new_input(kv_weight,  (-1, -1, -1), 0, True)
      tb_graph.new_input(qr_out,     (0, -1, -1), 1, True)
      tb_graph.new_input(kv_out,     (0, -1, -1), 1, True)
      self.kn_graph.customized(
          [qr_in, kv_in, q_weight, kv_weight, qr_out, kv_out], tb_graph)
      self.kn_graph.register_task(tb_graph, "mla_v4_q_kv_rmsnorm_sm100", [eps])
  ```

### 1.8 Initial (naive) implementation strategy

Plain CUDA, no TMA, no warp specialization. One CTA per `(token,
stream)`. Algorithm (128 threads / CTA):

1. Read `stream_idx` from `task_desc->task_metadata`. Select
   `(row_in_ptr, row_out_ptr, weight_ptr, SIZE)` triple — Q or KV.
2. Each thread loops the row at stride `blockDim.x`, accumulating
   `sum_sq_local += x*x` in FP32.
3. Block-reduce `sum_sq` via warp shuffles + shmem.
4. Thread 0 computes `rrms = rsqrtf(sum_sq / SIZE + eps)`, broadcasts via shmem.
5. Each thread re-loops, computes `y = x * rrms * w`, casts to bf16, stores.

Vectorized `ldg.v8.b16` and `st.v8.b16` are an easy optimization later but
not required for v1.

### 1.9 Test-mode unit test plan

File: `tests/runtime_python/test_mode/test_mla_v4_q_kv_rmsnorm_testmode.py`.

PyTorch oracle:

```python
def rmsnorm_ref(x, w, eps):
    dtype = x.dtype
    x = x.float()
    var = x.square().mean(-1, keepdim=True)
    return ((x * torch.rsqrt(var + eps)) * w).to(dtype)
```

Shapes for unit test: `T=4`, `q_lora_rank=128`, `kv_lora_rank=64`,
`eps=1e-6`. (Smaller than production so the test runs fast; production
shapes are also exercised in a parametrized sweep marked `slow`.)

Tolerances: `torch.allclose(rtol=1e-3, atol=1e-3)` matches the Wave-2 gate
in the plan ("§Wave-2 gate: torch.allclose(rtol=1e-3, atol=1e-3) for bf16
paths").

Driver:

```python
pk = PersistentKernel(...)
params = pk.get_default_init_parameters()
params["test_mode"] = True
qr_in  = pk.new_input(...);  kv_in  = pk.new_input(...)
qrW    = pk.new_input(...);  kvW    = pk.new_input(...)
qr_out = pk.new_output(...); kv_out = pk.new_output(...)
pk.mla_v4_q_kv_rmsnorm_layer(qr_in, kv_in, qrW, kvW, qr_out, kv_out,
                             eps=1e-6,
                             grid_dim=(T, 2), block_dim=(128, 1, 1))
pk(params)
ref_qr = rmsnorm_ref(qr_in_t, qrW_t, 1e-6)
ref_kv = rmsnorm_ref(kv_in_t, kvW_t, 1e-6)
assert torch.allclose(qr_out_t, ref_qr, rtol=1e-3, atol=1e-3)
assert torch.allclose(kv_out_t, ref_kv, rtol=1e-3, atol=1e-3)
```

### 1.10 Reuse from V3

Clone the codegen structure of `register_rmsnorm_task`
(`src/kernel/task_register.cc:91-125`) and the Python layer-method
structure of `rmsnorm_layer` (`persistent_kernel.py:1015-1031`). The only
material delta is **two input streams instead of one** and **two output
streams instead of one**. The intra-CTA reduction code is byte-identical.

---

## 2. `mla_v4_decode_layer` — dual-cache decode

### 2.1 When it runs

Decode step (`Q_LEN == 1` per request). One launch per attention layer.
Per-layer `compress_ratio ∈ {0, 4, 128}` selects the cache mix:

| ratio | SWA cache | Compressed cache | Topk indices |
|---|---|---|---|
| 0   | yes | NO  | NO  |
| 4   | yes | yes | yes (from Indexer, per token) |
| 128 | yes | yes | yes (deterministic, precomputed in metadata) |

### 2.2 Math

After per-head Q normalization + RoPE (done in
`_fused_qnorm_rope_kv_insert`, see sparse.md / vLLM lines 500-534) and KV
RoPE + FP8 quant + paged-cache insert, the decode attention is:

```
scale     = D ** -0.5   = 1/sqrt(512)
S         = concat( sliding_window_indices,         # ≤ W=128 valid SWA slots
                    topk_indices_into_compressed )  # for ratio > 0
o_h       = softmax_scaled( scale * Q_h @ K[S]ᵀ + (-inf @ pad) + attn_sink_h,
                            axis=last ) @ V[S]      # MLA: K == V == 512-d row
```

Where:
- `Q[T, H, D]` — per-token RMSNorm'd + RoPE'd query, **64 heads** of 512 dims
  (last 64 are RoPE).
- `K`/`V` are the **same 512-d row** (MLA). `K` = the 512-d KV-lora row with
  the last 64 dims RoPE'd; `V` = the same row (the head_dim is the latent).
  See `model.py:528` `o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)`
  — kv is single tensor passed for both K and V.
- `attn_sink[H]` is an additive logit sink that always participates in the
  softmax (`model.py:456` `self.attn_sink = nn.Parameter(... n_local_heads ...)`).
- Invalid (-1) indices in `topk` are masked to `-inf` before softmax.

Reference call site (vLLM, `deepseek_v4_attention.py:855-871`):

```python
out, _ = flash_mla_with_kvcache(
    q=q,
    k_cache=swa_cache,
    block_table=None,
    head_dim_v=512,
    tile_scheduler_metadata=tile_metadata,
    cache_seqlens=None,
    is_fp8_kvcache=True,
    indices=swa_indices,
    topk_length=swa_lens,
    softmax_scale=self.scale,
    attn_sink=self.attn_sink,
    extra_k_cache=kv_cache if not swa_only else None,
    extra_indices_in_kvcache=topk_indices,
    extra_topk_length=topk_lens,
    out=output.unsqueeze(1),
)
```

Reference math (PyTorch, `model.py:530-534`):

```python
self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)        # SWA write at slot start_pos%W
if self.compress_ratio:
    self.compressor(x, start_pos)                           # may write a compressed row
o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink,
                topk_idxs, self.softmax_scale)              # decode reads BOTH segments
apply_rotary_emb(o[..., -rd:], freqs_cis, True)             # inverse RoPE → separate task §5
```

Note `self.kv_cache` is sized `W + max_seq_len/ratio` (`model.py:473-474`):
the first `W=128` slots are SWA ring-buffer, the rest is the compressed
pool. This spec treats the two segments as **two physical paged caches**
(matching vLLM), bridged by index lists.

### 2.3 Inputs / Outputs / State

| Tensor | Shape | Dtype | Source |
|---|---|---|---|
| **inputs** | | | |
| `q`                      | `[B, 1, H=64, D=512]` (Q_LEN=1) | bf16 | output of `wq_b` + per-head-norm + GPT-J RoPE |
| `swa_paged_cache`        | `[num_pages, page_size, 1, head_bytes]` where `head_bytes` = 448 fp8 + 64 bf16 RoPE + 7 scale-bytes + 1 pad (vLLM `deepseek_v4_attention.py:217-223`) | uint8-packed | written by KV-cache-insert task (sparse.md) |
| `compressed_paged_cache` | same layout, ratio>0 only | uint8-packed | written by Compressor task (sparse.md) |
| `swa_indices`            | `[B, top_k_swa]` int32 (token positions into SWA cache; -1 = pad) | int32 | from `swa_metadata.decode_swa_indices` (`deepseek_v4_attention.py:816`) |
| `swa_lens`               | `[B]` int32 | int32 | `swa_metadata.decode_swa_lens` |
| `topk_indices`           | `[B, 1, top_k_comp]` int32, **absent when ratio==0** | int32 | ratio=4 → Indexer; ratio=128 → `attn_metadata.c128a_global_decode_topk_indices` (line 813) |
| `topk_lens`              | `[B]` int32, absent when ratio==0 | int32 | matching length tensor |
| `attn_sink`              | `[H=64]` | fp32 | `model.py:456`; persistent |
| **outputs** | | | |
| `o`                      | `[B, 1, H_padded, D=512]` where `H_padded=64` (vLLM `deepseek_v4_attention.py:153-156`) | bf16 | feeds inverse RoPE + FP8 quant (§5) |
| **state read** | | | |
| `paged_kv_indptr_buffer`, `paged_kv_indices_buffer`, `paged_kv_last_page_len_buffer` | from MPK runtime meta-tensors (`persistent_kernel.py:48-50, 3312-3314`) | int32 | reused from V3 (no extension needed for v1) |

`scale = 1.0 / sqrtf(512)` is a register constant from `softmax_scale` param.

### 2.4 Source to migrate

Primary: `flash_mla_with_kvcache` call site at
`deps/vllm/vllm/model_executor/layers/deepseek_v4_attention.py:781-871`. The
**kernel body itself** (FlashMLA-Sparse FP8) lives in
`deps/vllm/vllm/attention/ops/flash_mla.py` and beneath that in CUTLASS C++
under FlashMLA. For v1 we **do not port FlashMLA-Sparse**; we write a naive
fused kernel and only match its semantics. Quote of the critical 15 lines
above (decode-path `flash_mla_with_kvcache` invocation, with the two
extra-cache slots set when ratio > 0).

V3 analog: `mla_decode_layer` (`persistent_kernel.py:1451-1481`) and its
kernel `include/mirage/persistent_kernel/tasks/blackwell/mla_decode_sm100.cuh`.
The V4 kernel differs in that it must accept **two cache pointers and two
index lists** and merge their contributions in the same softmax.

### 2.5 Generated CUDA reference

`docs/mpk/deepseek_v4/_generated_cuda/mla_v4_decode.cu` — produced by
running FlashMLA's CUTLASS C++ codegen (extracted from
`flash_mla_with_kvcache`) on the V4-Flash shapes `(H=64, D=512,
top_k_swa=128, top_k_comp=512)`. If extraction fails (FlashMLA is not
TileLang and the CUTLASS template is heavy), the spec authors will dump
the equivalent **TileLang sparse_attn kernel** from
`deps/deepseek_v4/DeepSeek-V4-Flash/inference/kernel.py` `sparse_attn`
(see `kernel.py:277-371`, which has a TileLang `sparse_attn_kernel`) to
`mla_v4_decode.cu`. That TileLang variant is the cleanest semantic
reference and is appropriate for the v1 naive port.

### 2.6 Grid design

**Natural grid.** Following V3's `mla_decode_layer` (`persistent_kernel.py:1467,
1469` — `grid = (num_splits, num_head_groups, max_num_batched_requests)`):

For V4, decode Q_LEN==1, so:

- `num_head_groups = H / HEADS_PER_BLOCK = 64 / HEADS_PER_BLOCK`. For v1 use
  `HEADS_PER_BLOCK = 8` ⇒ `num_head_groups = 8`. (V3 uses `HEADS_PER_BLOCK
  = 128/Q_LEN` to saturate MMA-K; for V4 we override with a fixed 8 because
  H is only 64.)
- `num_splits = ceil( (top_k_swa + top_k_comp) / TILE_S )` where
  `TILE_S = 128` matches V3.
- `B = max_num_batched_requests`.

So `grid_dim = (num_splits, num_head_groups, B)`. Output is a partial
`[B, num_head_groups, num_splits, HEADS_PER_BLOCK, D]` plus partial
`LSE[B, num_head_groups, num_splits, HEADS_PER_BLOCK]`; a separate
`mla_v4_reduce` task (clone V3 `mla_reduce_layer`,
`persistent_kernel.py:1483-1511`) collapses splits → final O. **This spec
keeps the reduce task as a direct V3 reuse**; the only new task is the
per-split decode body.

**CTA selects its slice from task_metadata.** Per the V3 mla_decode kernel,
each CTA reads from `task_desc->task_metadata` the fields
`{request_id, head_group_id, split_id}` (the names mirror what
`register_mla_decode_sm100_task` emits, `task_register.cc:3439`). The CTA
then:

1. Reads `swa_lens[request_id]`, `topk_lens[request_id]` (if ratio>0).
2. Determines its split's `[s_lo, s_hi)` over the **concatenated** valid
   index list (SWA tail first, then topk).
3. For each `s ∈ [s_lo, s_hi)`, looks up `(index_arr, cache_ptr) =
   (swa_indices, swa_cache)` if `s < swa_len[req]` else
   `(topk_indices, compressed_cache)`. (Boundary index `swa_len[req]` is
   the split point.)
4. Gathers `K[s]` (and same row as V) into shared memory; computes
   QK·scale, applies attn_sink, softmax, V matmul.

**MPK partitioning policy.** Like V3 `mla_decode`, all inputs are passed
with map `(-1, -1, -1)` or `(0, -1, -1)` for batch-first (q) — see
`persistent_kernel.py:1470-1477`. Each CTA reads its slice via
`task_metadata`, not auto-partition. The persistent runtime dispatches one
CTA per `(num_splits × num_head_groups × B)` grid slot to arbitrary worker
thread blocks.

**Alignment constraints.** `D = 512` divides 128 (TMA chunk size) cleanly.
`top_k_swa = 128` and `top_k_comp ≤ 512` both divide `TILE_S = 128`. We
require `num_splits * TILE_S ≥ top_k_swa + top_k_comp` (host-side check at
layer registration).

### 2.7 `/add-mpk-task` conformance checklist

- [ ] `blockIdx`-agnostic. All routing via `task_desc->task_metadata.{request_id,
  head_group_id, split_id}` (mirroring `mla_decode_sm100`'s metadata pattern at
  `task_register.cc:3439-3535`).
- [ ] `TaskType` enum entry `TASK_MLA_V4_DECODE_SM100` in
  `runtime_header.h`, plus `TASK_MLA_V4_REDUCE_SM100` if we add a
  V4-specific reduce (likely not needed — V3 reduce is shape-agnostic).
- [ ] `register_mla_v4_decode_sm100_task` in `task_register.cc`, cloned
  from `register_mla_decode_sm100_task` at `task_register.cc:3439`. New
  task accepts **6 input pointers** instead of 4: `q`, `swa_cache`,
  `compressed_cache`, `swa_indices`, `topk_indices`, `attn_sink`, plus
  `output_partial` and `output_lse`. When ratio==0, the layer method
  passes a null `DTensor` for `compressed_cache` and `topk_indices` and
  passes a compile-time param `HAS_COMPRESSED=false` so the kernel's
  templated path skips the second-cache loop.
- [ ] Dispatch entry in `src/kernel/graph.cc`: `"mla_v4_decode_sm100"` →
  `register_mla_v4_decode_sm100_task`.
- [ ] Layer-method in `persistent_kernel.py` follows
  `mla_decode_layer` (lines 1451-1481), extended:

  ```python
  def mla_v4_decode_layer(
      self,
      q: DTensor,
      swa_cache: DTensor,
      compressed_cache: Optional[DTensor],
      swa_indices: DTensor, swa_lens: DTensor,
      topk_indices: Optional[DTensor], topk_lens: Optional[DTensor],
      attn_sink: DTensor,
      output_partial: DTensor, output_lse: DTensor,
      mla_params: tuple,           # (num_heads=64, d_k=512, d_v=512,
                                   #  num_splits, kv_len_swa, kv_len_comp,
                                   #  compress_ratio)
      grid_dim: tuple, block_dim: tuple,
  ):
      ...
      self.kn_graph.register_task(tb_graph, "mla_v4_decode_sm100", params)
  ```

### 2.8 Initial (naive) implementation strategy

Plain CUDA, FP32 accumulator, no TMA, no tcgen05. One CTA per (request,
head_group, split). 256 threads/CTA (Blackwell convention,
`WORKER_NUM_THREADS=256`):

1. Load `Q[req, 0, head_group*HPB:(head_group+1)*HPB, :]` into shmem
   (`HPB×D = 8×512 = 4096 bf16 = 8 KB`).
2. For each `s ∈ [s_lo, s_hi)`:
   - Select index source: `(arr, cache_ptr) = (swa_indices, swa_cache)` if
     `s < swa_lens[req]` else `(topk_indices, compressed_cache)`.
   - Read index `idx = arr[req, s_in_arr]`. If `idx == -1`, treat logit as
     `-inf`.
   - Gather K-row (= V-row, MLA latent) `K[idx, :]` into shmem (`512 bf16
     = 1 KB`). FP8 dequant on the fly for the 448-d FP8 chunk; bf16 for
     the 64-d RoPE tail.
3. Compute logit `l_s,h = scale * dot(Q[h], K[idx]) + attn_sink_factor`
   (attn_sink contributes additively only to the first split's slot 0
   per the FlashMLA convention; document explicit per-split handling in
   `_generated_cuda/mla_v4_decode.cu`).
4. Online softmax across this CTA's split (Flash-attention style: maintain
   per-head `m, l` running max/normalizer).
5. Accumulate `O_split[h, :] += softmax_weight_s * V[idx, :]` into
   FP32 register tile (`HPB × D`).
6. After loop, store FP32 `O_split` and FP32 `LSE` to global
   (`output_partial`, `output_lse`) — the reduce task collapses splits.

**v1 simplifications:** uniform 256-thread CTA; no warp specialization;
gather K via direct global loads (no TMA descriptor); FP8 dequant via
inline `__nv_fp8_e4m3 → __nv_bfloat16` conversions.

### 2.9 Test-mode unit test plan

File: `tests/runtime_python/test_mode/test_mla_v4_decode_testmode.py`.

PyTorch oracle (extracted from `model.py:528-534`):

```python
def sparse_attn_ref(q, kv, attn_sink, topk_idxs, scale):
    # q: [B, 1, H, D]; kv: [B, S_total, D]; topk_idxs: [B, 1, K] int (-1 = pad)
    B, _, H, D = q.shape
    K = topk_idxs.shape[-1]
    out = torch.zeros_like(q)
    for b in range(B):
        idx = topk_idxs[b, 0]   # [K]
        mask = idx >= 0
        valid = idx[mask]
        k_gathered = kv[b, valid]              # [Kv, D]
        logits = scale * q[b, 0] @ k_gathered.T   # [H, Kv]
        # attn_sink contributes a (-inf, sink_h) pair
        full = torch.cat([logits, attn_sink.unsqueeze(-1).expand(H, 1)], dim=-1)
        w = full.softmax(-1)   # [H, Kv+1]
        w_data = w[:, :-1]
        out[b, 0] = w_data @ k_gathered   # MLA: V == K
    return out
```

Small-shape unit test: `B=2`, `H=4`, `D=128`, `D_rope=16`, `swa_len=8`,
`top_k_comp=8`, `compress_ratio=4`. Run kernel and reduce; compare
`o_final` against `sparse_attn_ref(q, kv, sink, topk, scale)` with
`torch.allclose(rtol=1e-3, atol=1e-3)`.

Parametrized over `ratio ∈ {0, 4, 128}`:
- `ratio == 0`: pass empty `compressed_cache` and `topk_indices`; the test
  reduces to V3-style SWA decode and should match a plain windowed-attention
  reference.
- `ratio == 4`: feed a precomputed `topk_indices` matching the PyTorch
  reference's Indexer-selected positions (test taps the official Indexer
  to generate them).
- `ratio == 128`: feed the precomputed deterministic positions
  (`get_compress_topk_idxs` from `model.py:268-276`).

### 2.10 Reuse from V3

- V3's `mla_decode_layer` Python pattern and grid shape
  (`persistent_kernel.py:1451-1481`) ports directly; only the inputs list
  grows.
- V3's `mla_reduce_layer` (`persistent_kernel.py:1483-1511`) is reused
  verbatim — same partial/LSE shapes.
- V3's `task_register.cc:register_mla_decode_sm100_task` is the codegen
  template.
- V3's paged-KV-cache meta-tensors (`paged_kv_indptr_buffer`,
  `paged_kv_indices_buffer`, `paged_kv_last_page_len_buffer` —
  `persistent_kernel.py:48-50, 3312-3314`) are reused **for the SWA cache
  only**. The compressed cache is a **second paged cache** with the same
  per-page layout but its own indptr/indices/last_page_len triple. The
  audit task in Wave 1 (per plan §A) is responsible for confirming the
  buffer-naming extension; this spec assumes the new triple is named
  `paged_compressed_kv_indptr_buffer` etc.

---

## 3. `mla_v4_prefill_layer` — sparse prefill over gathered-KV workspace

### 3.1 When it runs

Prefill step (`T > 1`, multi-token chunked prefill). Operates **after**
`mla_v4_prefill_gather_layer` (§4) has staged SWA + compressed cache into a
contiguous workspace. One launch per prefill chunk per layer.

### 3.2 Math

```
for token t in [chunk_start, chunk_end):
    indices_t   = combined_indices[t, :]   # SWA-window + topk_compressed
    K_t = V_t   = gathered_kv[ indices_t ]
    logits_t    = scale * Q[t] @ K_tᵀ + attn_sink (per head)
    o[t]        = softmax(logits_t) @ V_t
```

Same MLA-style single-tensor K==V (latent) as decode. The key vs. decode:
prefill is **multi-query** within a chunk and uses a **single contiguous
workspace `kv[chunk_size, M, D]`** rather than paged caches; this
eliminates the dual-cache dispatch entirely (gather handled it).

Reference call site (vLLM, `deepseek_v4_attention.py:983-991`):

```python
output_chunk, _, _ = flash_mla_sparse_fwd(
    q=q[query_start:query_end],
    kv=kv.view(-1, 1, q.shape[-1]),
    indices=combined_indices.unsqueeze(1),
    sm_scale=self.scale,
    attn_sink=self.attn_sink,
    topk_length=combined_lens,
    out=output[query_start:query_end],
)
```

Note: `kv.view(-1, 1, q.shape[-1])` — single contiguous 3D view. `q.shape[-1]
== D == 512` since this is post-RoPE. **No SWA / compressed split inside
this kernel**; the gather has flattened them.

### 3.3 Inputs / Outputs / State

| Tensor | Shape | Dtype | Source |
|---|---|---|---|
| **inputs** | | | |
| `q`                  | `[T_prefill, H=64, D=512]` | bf16 | from per-head-norm + RoPE on `wq_b` output |
| `gathered_kv`        | `[T_prefill, M, D]` where `M = N_compressed + W_swa + max_num_batched_tokens` (vLLM `deepseek_v4_attention.py:924`) | bf16 | staged by `mla_v4_prefill_gather_layer` (§4) |
| `combined_indices`   | `[T_prefill, top_k_total]` int32 (-1 = pad) | int32 | `combine_topk_swa_indices(...)` (line 969-981) |
| `combined_lens`      | `[T_prefill]` int32 | int32 | per-token valid index count |
| `attn_sink`          | `[H=64]` | fp32 | persistent |
| **outputs** | | | |
| `o`                  | `[T_prefill, H_padded=64, D=512]` | bf16 | feeds §5 |

### 3.4 Source to migrate

Primary: `flash_mla_sparse_fwd` call site at
`deepseek_v4_attention.py:873-991`. Underlying TileLang reference for the
**sparse attention math** is
`deps/deepseek_v4/DeepSeek-V4-Flash/inference/kernel.py:277-371`
(`sparse_attn` + `sparse_attn_kernel`). The TileLang DSL is the v1 port
target. Excerpt (`kernel.py:277-371`, ~15 critical lines):

```python
@tilelang.jit
def sparse_attn_kernel(h: int, d: int, scale=None):
    ...
    @T.prim_func
    def sparse_attn_kernel_(Q: T.Tensor[(B, H, D)],
                            KV: T.Tensor[(B, T, D)],
                            sink: T.Tensor[(H,)],
                            idxs: T.Tensor[(B, K)],
                            O: T.Tensor[(B, H, D)]):
        for bi in T.Parallel(B):
            ...
            for k in T.serial(0, K, block_K):
                T.gather(K_smem, KV[bi], idxs[bi, k:k+block_K])
                ...
                qk = T.dot(Q_smem, K_smem) * scale
                # softmax with sink
                ...
                O_smem += w_smem @ K_smem
```

(Full DSL ported to `_generated_cuda/mla_v4_prefill.cu` in Wave 1.)

V3 analog: `mla_prefill_layer` (`persistent_kernel.py:1513-1540`) and
`mla_prefill_sm100.cuh`. V3 prefill is **dense over a gathered single
cache**; V4 prefill is **sparse over a flattened single workspace** — the
math is simpler (no per-step dispatch) but adds an indexed gather inside
the K loop.

### 3.5 Generated CUDA reference

`docs/mpk/deepseek_v4/_generated_cuda/mla_v4_prefill.cu` — TileLang JIT
of `sparse_attn_kernel` (above) with shapes `H=64, D=512, K=top_k_total`.

### 3.6 Grid design

**Natural grid.** Per token, per head-tile, per Q-tile:
`grid_dim = (num_q_blocks, num_head_groups, T_prefill_chunk)`. V3's
`mla_prefill_sm100` uses `grid_dim = (H, num_q_blocks, B)` — see
`persistent_kernel.py:1521`. For V4 we instead group heads (HPB=8) and
parallelize across chunk tokens:

- `num_q_blocks = ceil(T_chunk / BM)` with `BM = 64` (matches V3
  `mla_prefill_tp8`).
- `num_head_groups = H / HPB = 64 / 8 = 8`.
- Outer dim = `T_prefill_chunk` (chunks are `PREFILL_CHUNK_SIZE`, see
  vLLM line 925).

**CTA selects slice via task_metadata.** Each CTA reads
`task_desc->task_metadata.{q_block_id, head_group_id, chunk_token_id}`.
The CTA loads its 64-token Q tile (`BM=64`) of HPB heads, then loops
over `K = combined_lens[chunk_token_id]` indices in `BN=64` blocks,
gathering K-rows via `combined_indices`. Online softmax + V accumulation
as in decode.

**MPK partitioning.** Follow V3 `mla_prefill_layer`
(`persistent_kernel.py:1532-1540`): all inputs mapped `(-1, -1, -1)` so MPK
does not auto-partition; the CTA picks via metadata.

**Alignment constraints.** `BM = 64` requires `T_chunk` to be padded to a
multiple of 64 (vLLM uses `PREFILL_CHUNK_SIZE = 4096` per the constant;
exact value is in `deps/vllm/vllm/v1/attention/backends/...`). `K` (top_k
total) need not divide 64 — the kernel masks the tail with -inf via the
`-1` index path.

### 3.7 `/add-mpk-task` conformance checklist

- [ ] `blockIdx`-agnostic, metadata fields `{q_block_id, head_group_id, chunk_token_id}`.
- [ ] `TaskType` entry `TASK_MLA_V4_PREFILL_SM100`.
- [ ] `register_mla_v4_prefill_sm100_task` cloned from
  `register_mla_prefill_sm100_task` (`task_register.cc:3605-3691`). Inputs:
  `q`, `gathered_kv`, `combined_indices`, `combined_lens`, `attn_sink`,
  `output`. Params: `(num_heads=64, max_T_chunk, D=512, top_k_total)`.
- [ ] Dispatch in `graph.cc`.
- [ ] Layer-method in `persistent_kernel.py`:

  ```python
  def mla_v4_prefill_layer(
      self,
      q: DTensor, gathered_kv: DTensor,
      combined_indices: DTensor, combined_lens: DTensor,
      attn_sink: DTensor,
      output: DTensor,
      mla_params: tuple,         # (H=64, T_chunk, D=512, top_k_total)
      grid_dim: tuple, block_dim: tuple,
  ):
      ...
      self.kn_graph.register_task(tb_graph, "mla_v4_prefill_sm100", params)
  ```

### 3.8 Initial (naive) implementation strategy

256 threads/CTA. Per CTA:

1. Load `Q[q_lo:q_hi, head_group, :]` tile (`BM × HPB × D = 64 × 8 × 512
   = 256 KB`). Too large; instead load `BM × D = 32 KB` and reuse across
   heads in the inner loop. **Or**: load `HPB heads × BM=16` (`16 × 8 ×
   512 = 64 KB`) at the cost of more outer iterations. v1 picks `BM=16`
   for memory safety.
2. Per-token: read `K_t = combined_lens[t]`; loop `k_lo` over `[0, K_t)`
   in `BN=64` steps:
   - Read 64 indices `idxs = combined_indices[t, k_lo:k_lo+64]`.
   - Gather `K_smem[64, D] = gathered_kv[t, idxs, :]`. (Indexed gather; v1
     uses scalar loads; v2 may use TMA-gather.)
   - QK·scale → logits[BM=16, BN=64]; add sink at logits[:, 0] of first
     iteration.
   - Online softmax update of (m, l).
   - V-accum: `O[BM, D] += softmax_w @ K_smem` (V == K).
3. Store final `O[q_lo:q_hi, head_group_heads, :]` to `output`.

**v1 simplifications:** scalar-gather, no TMA, FP32 accum, no warp
specialization.

### 3.9 Test-mode unit test plan

File: `tests/runtime_python/test_mode/test_mla_v4_prefill_testmode.py`.

Oracle: same `sparse_attn_ref` as §2.9 but applied across `T_prefill > 1`
tokens. Compose with §4's gather oracle so the test exercises gather +
prefill end-to-end on a tiny case (`T=4`, `H=4`, `D=128`, `M=24`,
`top_k_total=8`). Tolerance `rtol=1e-3, atol=1e-3`.

### 3.10 Reuse from V3

- Codegen template `register_mla_prefill_sm100_task`
  (`task_register.cc:3605`).
- Python layer-method shape `mla_prefill_layer`
  (`persistent_kernel.py:1513-1540`) — adapt inputs.
- TBGraph `(-1, -1, -1)` input map pattern (lines 1532-1536) used verbatim.
- The CUDA Q-tile/online-softmax structure is essentially a stripped-down
  `mla_prefill_sm100.cuh` with a gather replacing the linear K-row scan.

---

## 4. `mla_v4_prefill_gather_layer` — gather SWA + compressed pages

### 4.1 When it runs

Prefill, **before** `mla_v4_prefill_layer` (§3). One launch per prefill
chunk per layer. For SWA-only layers (ratio==0), the gather operates only
on the SWA cache; for ratio ∈ {4, 128}, the gather concatenates the
compressed-cache pool (size `N`) **then** the SWA window (size `≤ W`).

### 4.2 Math

Effectively a paged-cache gather + dequant. For each prefill request `r`
and each of its tokens `t`, the gather produces a contiguous workspace row
`gathered_kv[chunk_local_t, m, :]` where `m` covers `[0, N)` for compressed
positions and `[N, N+W)` for SWA. Pseudo-code follows vLLM's
`dequantize_and_gather_k_cache` calls
(`deepseek_v4_attention.py:939-959`):

```python
# 1. Compressed gather (skip for ratio == 0)
if not swa_only:
    dequantize_and_gather_k_cache(
        kv[:chunk_size],
        compressed_k_cache,
        seq_lens=seq_lens[chunk_start:chunk_end] // compress_ratio,
        gather_lens=None,
        block_table=block_table[chunk_start:chunk_end],
        block_size=attn_metadata.block_size // compress_ratio,
        offset=0,
    )

# 2. SWA gather
dequantize_and_gather_k_cache(
    kv[:chunk_size],
    swa_k_cache,
    seq_lens=seq_lens[chunk_start:chunk_end],
    gather_lens=gather_lens[chunk_start:chunk_end],
    block_table=swa_block_table[chunk_start:chunk_end],
    block_size=swa_metadata.block_size,
    offset=N,
)
```

The cache entries are FP8-packed for the 448-d nope chunk and bf16 for the
64-d RoPE tail; gather **dequantizes inline** (per-block FP8 scales are
embedded in the page).

### 4.3 Inputs / Outputs / State

| Tensor | Shape | Dtype | Source |
|---|---|---|---|
| **inputs** | | | |
| `swa_paged_cache`         | `[num_pages_swa, page_size, 1, head_bytes]` | uint8-packed FP8 + bf16 + scales | written by KV-insert |
| `compressed_paged_cache`  | same, ratio>0 only | uint8-packed | written by Compressor |
| `swa_block_table`         | `[T_prefill_chunk_reqs, max_pages_per_req]` int32 | int32 | SWA paged-cache indirection (V3 reuses `paged_kv_indices_buffer`) |
| `compressed_block_table`  | similar | int32 | second set of paged buffers |
| `seq_lens`                | `[T_prefill_chunk_reqs]` int32 | int32 | prefill metadata |
| `gather_lens`             | `[T_prefill_chunk_reqs]` int32 | int32 | how many SWA slots to copy (≤ W) |
| **outputs** | | | |
| `gathered_kv`             | `[T_prefill_chunk, M, D=512]` where `M = N + W + max_num_batched_tokens` (vLLM line 924) | bf16 | feeds §3 |
| **state** | | | |
| paged_indptr/indices/last_page_len buffers (one set per cache) | | int32 | reuse V3 meta-tensors |

### 4.4 Source to migrate

Primary: `dequantize_and_gather_k_cache` calls
(`deepseek_v4_attention.py:939-959`). The underlying implementation is
in `deps/vllm/vllm/attention/ops/...` (Triton or CUDA — `grep -rn "def
dequantize_and_gather_k_cache" deps/vllm/`). v1 ports the **bf16-only
path**: for v1 we may store the cache as bf16 directly (matching V3's
single-bf16 cache) and add the FP8-pack/unpack as a v2 optimization.
**Decision**: Wave-1 spec assumes bf16 cache rows for v1, deferring FP8
pack/unpack to v2. This needs ratification by the Wave-1 audit task per
plan §A.

V3 analog: `mla_kv_gather_split_layer` (`persistent_kernel.py:1422-1449`)
+ `mla_kv_cache_gather_sm100.cuh` (already opened, lines 1-80 reviewed).
V4 gather extends V3 by:
1. Gathering from **two** caches into one contiguous output.
2. Per-row `gather_lens` (SWA can have fewer valid slots than W).

### 4.5 Generated CUDA reference

`docs/mpk/deepseek_v4/_generated_cuda/mla_v4_prefill_gather.cu` —
extracted from the vLLM `dequantize_and_gather_k_cache` Triton source (or
hand-ported from V3's gather .cuh).

### 4.6 Grid design

**Natural grid.** `(num_pages_per_chunk, num_requests_in_chunk, 1)` —
each CTA copies one paged-cache page into the workspace for one request.

V3 `mla_kv_cache_gather_sm100` uses `grid = (max_num_batched_requests, 1,
1)` with a single CTA per request looping over pages
(`mla_kv_cache_gather_sm100.cuh:26-27`). For V4 we use the same simple
shape for v1: `grid_dim = (num_prefill_requests, 1, 1)` and each CTA loops
both caches; this avoids cross-CTA dependencies.

**CTA selects slice via task_metadata.** `request_id` field; everything
else (page count, byte offsets) computed inside the CTA from
`paged_kv_indptr_buffer[request_id]`, etc. — mirror of V3 lines 56-64 of
the gather kernel.

**MPK partitioning.** Same as V3 gather: all inputs mapped `(-1, *, -1)`
with the request dim available for splitting. See
`persistent_kernel.py:1442-1449` for the existing pattern.

**Alignment.** Page size and head_bytes per V3.

### 4.7 `/add-mpk-task` conformance checklist

- [ ] `blockIdx`-agnostic, metadata field `request_id`.
- [ ] `TaskType` entry `TASK_MLA_V4_PREFILL_GATHER_SM100`.
- [ ] `register_mla_v4_prefill_gather_sm100_task` cloned from
  `register_mla_kv_gather_split_sm100_task` (`task_register.cc:4242`).
- [ ] Dispatch in `graph.cc`.
- [ ] Layer-method `mla_v4_prefill_gather_layer` mirroring
  `mla_kv_gather_split_layer` (`persistent_kernel.py:1422-1449`) with two
  paged-cache inputs and two block-tables.

### 4.8 Initial (naive) implementation strategy

128 threads/CTA. Per CTA:

1. Read `request_id` from `task_desc->task_metadata`.
2. From SWA paged buffers, copy `gather_lens[req]` rows from the SWA
   ring-buffer (handling the wraparound at `start_pos % W`) into
   `gathered_kv[req_offset_in_chunk : ..., N:N+gather_lens[req], :]`. Each
   row is 512 bf16 = 1 KB; threads cooperate on a single row at 4 bf16 /
   thread.
3. If ratio>0, from compressed paged buffers, copy `seq_lens[req] // ratio`
   rows into `gathered_kv[..., 0:N_used, :]`.
4. Zero-fill any unused tail of `[0, M)` so the §3 kernel can safely treat
   `-1` indices.

**v1 simplifications:** bf16-direct cache (no FP8 dequant); scalar copies
(no TMA); two sequential copy passes per CTA.

### 4.9 Test-mode unit test plan

File: `tests/runtime_python/test_mode/test_mla_v4_prefill_gather_testmode.py`.

Oracle (PyTorch):

```python
def gather_ref(swa_cache, comp_cache, swa_bt, comp_bt, swa_lens, gather_lens,
               seq_lens, ratio, page_size, N, W):
    chunk = swa_lens.shape[0]
    M = N + W
    out = torch.zeros(chunk, M, 512, dtype=torch.bfloat16)
    for r in range(chunk):
        # compressed
        if comp_cache is not None:
            n_used = seq_lens[r] // ratio
            for p in range(n_used):
                page_idx = comp_bt[r, p // (page_size // ratio)]
                ...
        # swa
        n_swa = gather_lens[r]
        for p in range(n_swa):
            ...
    return out
```

Small shape: chunk=2, page_size=4, ratio=4, N=4, W=8, D=128. Verify
`torch.equal(mpk_out, ref_out)` (this is a copy task; no tolerance).

### 4.10 Reuse from V3

- `register_mla_kv_gather_split_sm100_task` (codegen template,
  `task_register.cc:4242-...`).
- `mla_kv_cache_gather_sm100.cuh` lines 36-80 (page indexing pattern,
  bounds checks, qo_indptr/paged_kv_indptr usage).
- Python `mla_kv_gather_split_layer` (`persistent_kernel.py:1422-1449`).

**Identified V3 gap:** V3 has a single cache; V4 gather needs to handle
two caches. The audit task in plan Phase A is responsible for confirming
whether the V3 meta-tensors generalize. If not, V4 introduces a second
triple `paged_compressed_kv_{indptr,indices,last_page_len}_buffer` as
**additive** new meta-tensors.

---

## 5. `inv_rope_fp8_quant_o_layer` — inverse RoPE + per-block FP8 quant of O

### 5.1 When it runs

Every attention layer, **immediately after** `mla_v4_decode_layer` or
`mla_v4_prefill_layer`. Feeds the grouped FP8 `wo_a` GEMM.

### 5.2 Math

Per token, per head, per head-dim chunk of `quant_group_size = 128`:

1. **Inverse RoPE** on the last `rope_dim=64` dims (and only those):
   GPT-J-style pairing, with conjugate (negative sin) frequencies.
2. **Per-block FP8 quant** for the full `head_dim=512` row, grouped into
   `chunks_per_head = 512/128 = 4` blocks of 128 elements each. Scale =
   `2^ceil(log2(absmax / fp8_max))` (UE8M0 power-of-two quant).
3. **Pack scales** into INT32 (4 UE8M0 bytes per group) for SM100, OR
   FP32 scales for SM90. V4-Flash on B200 uses **SM100 / UE8M0 packed**.

Source quote (vLLM Triton,
`deps/vllm/vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py:71-129`):

```python
input_base = o_ptr + pid_token * o_stride_token + global_head * o_stride_head
HEAD_DIM: tl.constexpr = CHUNKS_PER_HEAD * QUANT_GROUP_SIZE
offsets = tl.arange(0, HEAD_DIM)
x = tl.load(input_base + offsets).to(tl.float32)

rope_abs_start: tl.constexpr = (CHUNKS_PER_HEAD - 1) * QUANT_GROUP_SIZE + ROPE_START
pos = tl.load(positions_ptr + pid_token)
cache_base = cos_sin_cache_ptr + pos * cache_stride_pos
is_rope = offsets >= rope_abs_start
rope_local = offsets - rope_abs_start

x_partner = tl.load(input_base + (offsets ^ 1), mask=is_rope, other=0.0).to(tl.float32)
cs_idx = tl.maximum(rope_local >> 1, 0)
cos_v  = tl.load(cache_base + cs_idx, mask=is_rope, other=1.0)
sin_v  = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope, other=0.0)
x_add  = x * cos_v + x_partner * sin_v
x_sub  = x * cos_v - x_partner * sin_v
is_even = (rope_local & 1) == 0
rotated = tl.where(is_even, x_add, x_sub)
x = tl.where(is_rope, rotated, x)

x_2d = tl.reshape(tl.abs(x), (CHUNKS_PER_HEAD, QUANT_GROUP_SIZE))
block_absmax = tl.maximum(tl.max(x_2d, axis=1), eps)
scale_raw    = block_absmax * (1.0 / fp8_max)
scales       = tl.math.exp2(tl.ceil(tl.log2(scale_raw)))   # UE8M0: power-of-2
...
x_quant = tl.clamp(x / scales_exp, -fp8_max, fp8_max).to(tl.float8e4nv)
...
if TMA_ALIGNED_SCALES:
    scale_bits = scales.to(tl.int32, bitcast=True)
    ue8m0_bytes = (scale_bits >> 23) & 0xFF
    packed_val = tl.sum(ue8m0_bytes << (block_offsets * 8))   # 4 bytes packed into int32
    tl.store(scale_addr, packed_val)
```

The inverse RoPE differs from forward by using the **conjugate** of
`freqs_cis` (`model.py:236-237`: `if inverse: freqs_cis = freqs_cis.conj()`)
— which the Triton kernel implements directly by using cos with **positive**
sign in the rotation formula but applying the SUB pattern (`x*cos -
x_partner*sin`) at odd indices vs forward's `+`. Verify against the PyTorch
oracle in §5.9.

### 5.3 Inputs / Outputs / State

| Tensor | Shape | Dtype | Source |
|---|---|---|---|
| **inputs** | | | |
| `o`              | `[T, H=64, D=512]` | bf16 | output of §2 (decode) or §3 (prefill) |
| `positions`      | `[T]` | int64 | per-token absolute position (for RoPE) |
| `cos_sin_cache`  | `[max_pos, D_rope=64]` packed as `cos||sin` (`HALF_ROPE = rope_dim/2 = 32`) | fp32 | precomputed |
| **outputs** | | | |
| `o_fp8`          | `[n_groups=8, T, d=64*512/8=4096]`, strides `(d, T*d, 1)` — vLLM line 230-231 | float8_e4m3fn | feeds `wo_a` group FP8 GEMM (V3 `linear_fp8_layer`) |
| `o_scale`        | `[n_groups, num_tokens, scale_inner]` where `scale_inner = ceil(num_scale_blocks/4) = ceil(d/128/4) = 8` (since d=4096 → 32 scale blocks → 8 packed int32) | int32 (UE8M0 packed) | feeds wo_a fp8 GEMM |

Where `n_groups = o_groups = 8` (Flash-Base config). Each group has
`heads_per_group = H/n_groups = 64/8 = 8` heads. `d = heads_per_group *
head_dim = 8 * 512 = 4096` (vLLM line 177).

### 5.4 Source to migrate

Primary: `_fused_inv_rope_fp8_quant_per_head` Triton kernel
(`deps/vllm/vllm/v1/attention/ops/deepseek_v4_ops/fused_inv_rope_fp8_quant.py:16-134`).
Quote of the 25 critical lines is in §5.2 above.

PyTorch reference for inverse RoPE: `model.py:232-244` (`apply_rotary_emb`
with `inverse=True`).

### 5.5 Generated CUDA reference

`docs/mpk/deepseek_v4/_generated_cuda/inv_rope_fp8_quant_o.cu` — Triton
AOT dump on representative shape `(T=128, H=64, D=512, n_groups=8,
quant_group_size=128, tma_aligned=True)`.

### 5.6 Grid design

**Natural grid.** vLLM uses `(tma_aligned_T, n_groups * heads_per_group) =
(tma_aligned_T, 64)` (line 244). For MPK V4 we keep the same shape:
`grid_dim = (tma_aligned_T, H) = (T_padded, 64)`. Each CTA owns one
(token, head); inverse-RoPE + quant the 512-d head row; write FP8 + 1
packed-int32 scale.

`tma_aligned_T` = `get_tma_aligned_size(num_tokens, 4)` (vLLM line 184):
T rounded up to multiple of 4 for TMA-aligned scale buffer. Tail rows
(beyond `num_tokens`) just write zeros into the scale (line 50-69).

**CTA selects slice via task_metadata.** Two fields: `token_id`, `head_id`.
Group index = `head_id / heads_per_group`. Head-in-group =
`head_id % heads_per_group`.

**MPK partitioning.** All inputs mapped `(-1, -1, -1)` or `(0, -1, -1)`
for token-dim split on `o` and `positions`. `cos_sin_cache` replicated.

**Alignment.** `T` padded to multiple of 4 (TMA scales). `D=512 = 4 ×
128` quant chunks. `D_rope=64` is multiple of 2 (pair rotation).

### 5.7 `/add-mpk-task` conformance checklist

- [ ] `blockIdx`-agnostic, metadata fields `{token_id, head_id}`.
- [ ] `TaskType` entry `TASK_INV_ROPE_FP8_QUANT_O_SM100`.
- [ ] `register_inv_rope_fp8_quant_o_sm100_task` in `task_register.cc`.
  No close V3 analog; closest is `register_linear_fp8_sm100_task`
  (`task_register.cc:4083`) for the FP8-quant + scale-pack pattern, and
  the rotary-embed handling inside `attention_layer`
  (`persistent_kernel.py:1054`+) for the cos/sin lookup pattern.
- [ ] Dispatch in `graph.cc`.
- [ ] Layer-method:

  ```python
  def inv_rope_fp8_quant_o_layer(
      self,
      o: DTensor, positions: DTensor, cos_sin_cache: DTensor,
      o_fp8: DTensor, o_scale: DTensor,
      params: tuple,    # (n_groups=8, heads_per_group=8,
                        #  nope_dim=448, rope_dim=64, quant_group=128,
                        #  tma_aligned=True)
      grid_dim: tuple, block_dim: tuple,
  ):
      ...
      self.kn_graph.register_task(tb_graph, "inv_rope_fp8_quant_o_sm100", params)
  ```

### 5.8 Initial (naive) implementation strategy

64 threads/CTA. Per CTA owns 1 token × 1 head = 512 elements:

1. Read `token_id`, `head_id`, derive `group_id`, `head_in_group`.
2. Read `pos = positions[token_id]`. Load `cos_sin_cache[pos, :]` (64 fp32
   = 256 B) into shmem.
3. Cooperatively load `o[token_id, head_id, :]` (512 bf16 = 1 KB) into
   shmem as fp32.
4. For each of 64 RoPE elements (last 64 of the 512): compute the GPT-J
   pair rotation using the conjugate frequencies (see exact formula in
   §5.2). Replace in shmem.
5. Per quant-group (4 groups of 128 each): compute absmax →
   `scale = 2^ceil(log2(absmax/fp8_max))` (UE8M0). Each thread handles 2
   elements; warp reduction for absmax.
6. Quantize: `x / scale` clamp to `[-fp8_max, fp8_max]`, cast to
   `__nv_fp8_e4m3`. Store contiguously to `o_fp8[group, token, head_in_group*512
   : (head_in_group+1)*512]`.
7. Pack 4 UE8M0 bytes into one int32 and store to
   `o_scale[group, token, head_in_group]`.

**v1 simplifications:** scalar IO, no TMA, naive log2-and-ceil for UE8M0.
The strided-output write `(d, T*d, 1)` is non-standard — MPK needs
explicit `as_strided` handling in the layer registration (host-side
pre-stride the output tensor).

### 5.9 Test-mode unit test plan

File: `tests/runtime_python/test_mode/test_inv_rope_fp8_quant_o_testmode.py`.

PyTorch oracle:

```python
def inv_rope_quant_ref(o, positions, cos, sin, n_groups, hpg, nope, rope, quant_g):
    # 1. Inverse RoPE on last rope dims
    T, H, D = o.shape
    o = o.float().clone()
    pair = o[..., -rope::2].clone(), o[..., -rope+1::2].clone()
    c = cos[positions]   # [T, rope/2]
    s = sin[positions]
    o[..., -rope::2]   = pair[0] * c + pair[1] * s
    o[..., -rope+1::2] = pair[1] * c - pair[0] * s    # conjugate ⇒ minus sign
    # 2. Reshape into (n_groups, T, hpg*D), per-128 block UE8M0 quant
    o = o.view(T, n_groups, hpg, D).permute(1, 0, 2, 3).reshape(n_groups, T, hpg*D)
    # 3. UE8M0 quant per 128-element chunk
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    blocks = o.view(n_groups, T, -1, quant_g)
    absmax = blocks.abs().amax(-1).clamp_min(1e-10)
    scale  = 2 ** torch.ceil(torch.log2(absmax / fp8_max))
    o_q    = (o / scale.repeat_interleave(quant_g, -1).expand_as(o)).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return o_q, scale     # pack to int32 outside
```

Small shape: `T=4`, `H=4`, `D=128`, `D_rope=16`, `n_groups=2`,
`hpg=2`. Tolerances on the **dequantized roundtrip**:
`(o_fp8.float() * scale_expand)` vs the reference (use atol `2*fp8_unit`).
For the packed scale, compare bit-equal.

### 5.10 Reuse from V3

- The **FP8 quant + scale-pack** subroutine reuses the
  `per_token_group_quantize_fp8.cuh` UE8M0 logic — see plan §"Files to
  modify additively" line in
  `python/mirage/mpk/persistent_kernel.py:1974-1988`
  (`quantize_fp8_layer(scale_ue8m0=True)`).
- The RoPE cos/sin lookup mirrors V3 `attention_layer`'s rotary path
  (`persistent_kernel.py:1054-1104`), but for the **inverse** direction.
- Codegen template: `register_linear_fp8_sm100_task` for the FP8-write
  side (`task_register.cc:4083`).

---

## 6. Per-ratio decode walkthrough

### 6.1 `compress_ratio = 0` (SWA-only — layers 0, 1, 43-MTP)

Per `model.py:466-475`: when `compress_ratio == 0`, the `Attention.__init__`
**does not allocate** `self.compressor` or `self.indexer`, and
`kv_cache_size = window_size + 0`. The forward path
(`model.py:529-534`, decode branch):

```python
self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)  # SWA ring insert only
# (no compressor call)
o = sparse_attn(q, self.kv_cache[:bsz], attn_sink, topk_idxs, scale)
# topk_idxs == window-only indices (from get_window_topk_idxs, line 507)
```

MPK kernel sequence (per layer, decode):

```
mla_v4_q_kv_rmsnorm_layer(qr_in, kv_in, ...)
        ↓
linear_fp8_layer(qr_out → wq_b → q_pre_norm)        [V3 reuse]
per_head_rmsnorm + GPT-J RoPE (folded into KV-insert; sparse.md)
        ↓
fp8_kv_cache_insert (SWA cache only; sparse.md)
        ↓
mla_v4_decode_layer(
    q=q,
    swa_cache=swa_paged,
    compressed_cache=None,         # null DTensor
    swa_indices=swa_idx,           # window indices
    swa_lens=swa_len,
    topk_indices=None,             # null
    topk_lens=None,                # null
    attn_sink=...,
    ...,
    compress_ratio=0,
)
        ↓
mla_reduce_layer(...)              [V3 reuse]
        ↓
inv_rope_fp8_quant_o_layer(o, positions, cos_sin, → o_fp8, o_scale)
        ↓
moe_w13_fp8_layer (wo_a) + linear_layer (wo_b)     [V3 reuse]
```

### 6.2 `compress_ratio = 4` (SWA + indexed-compressed)

`model.py:508-515` builds `topk_idxs = cat([window_idxs,
compress_topk_idxs])` where `compress_topk_idxs` comes from the
**Indexer**. The kv_cache layout is `[W=128 SWA slots | (max_seq_len/4)
compressed slots]`. The decode branch (`model.py:529-534`):

```python
self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)     # SWA insert
self.compressor(x, start_pos)                            # compressed insert (sparse.md)
o = sparse_attn(q, self.kv_cache[:bsz], attn_sink, topk_idxs, scale)
```

MPK kernel sequence:

```
mla_v4_q_kv_rmsnorm_layer (...)
indexer_q_transform + indexer_score_topk     [sparse.md]
    → topk_indices[B, 1, 512] into compressed pool
compressor_layer                              [sparse.md, writes compressed cache]
fp8_kv_cache_insert (SWA)                     [sparse.md]
        ↓
mla_v4_decode_layer(
    q=q,
    swa_cache=swa_paged,
    compressed_cache=compressed_paged,
    swa_indices=swa_idx, swa_lens=swa_len,
    topk_indices=topk_idx,  topk_lens=topk_len,
    compress_ratio=4,
)
        ↓
mla_reduce_layer(...)
inv_rope_fp8_quant_o_layer(...)
wo_a + wo_b
```

### 6.3 `compress_ratio = 128` (SWA + deterministic-compressed)

Same as ratio=4 **except** there is **no Indexer**. `topk_indices` are
deterministic positions from `get_compress_topk_idxs(128, ...)`
(`model.py:268-276`), pre-baked into `attn_metadata.c128a_global_decode_topk_indices`
(`deepseek_v4_attention.py:813`).

MPK kernel sequence: identical to ratio=4 minus the indexer tasks.
Compressor still runs (writes compressed cache); topk_indices come from
a host-side precomputed buffer.

---

## 7. Per-ratio prefill walkthrough

### 7.1 `ratio = 0` (SWA-only)

```
mla_v4_q_kv_rmsnorm_layer
wq_b + per_head_rmsnorm + RoPE (sparse.md)
fp8_kv_cache_insert (SWA)
        ↓
mla_v4_prefill_gather_layer(
    swa_paged_cache,
    compressed_paged_cache=None,
    seq_lens, gather_lens,
    → gathered_kv[T_chunk, M, D]
)
        ↓
mla_v4_prefill_layer(
    q, gathered_kv,
    combined_indices=swa_window_indices_only,
    combined_lens=swa_lens,
    attn_sink, → o
)
        ↓
inv_rope_fp8_quant_o_layer
wo_a + wo_b
```

The gather workspace size is `M = W + max_num_batched_tokens` (compressed
size `N = 0`).

### 7.2 `ratio = 4`

```
mla_v4_q_kv_rmsnorm_layer
indexer (sparse.md) → topk_indices_buffer
compressor (sparse.md) → writes compressed cache
fp8_kv_cache_insert (SWA)
        ↓
mla_v4_prefill_gather_layer (BOTH caches)
        ↓
combine_topk_swa_indices  ← host-side metadata builder (vLLM line 969)
        ↓
mla_v4_prefill_layer(q, gathered_kv, combined_indices, combined_lens, ...)
        ↓
inv_rope_fp8_quant_o_layer
wo_a + wo_b
```

`combine_topk_swa_indices` is **not a new MPK task** for v1 — it is a CPU
metadata-builder that runs once per prefill chunk per layer. Its outputs
are uploaded into a `combined_indices` device buffer that is a meta-tensor
of the kernel call (no compilation needed). **OPEN:** see §10.

### 7.3 `ratio = 128`

Same as `ratio = 4` minus the Indexer; topk_indices are precomputed
deterministically.

---

## 8. MTP-decode reuse decision

The MTP block (`model.py:739-769`, `MTPBlock`) inherits from `Block` and
therefore runs the **same `Attention` module** as the base layers, with
`compress_ratio=0` (see plan §Context: "layer 43 (MTP) = 0").

V3's `mla_mtp_decode_layer` (`persistent_kernel.py:1571-1599`) was written
for V3's 128-head MLA + Q_LEN > 1 multi-token speculative path; its
internal constants (128 heads, TILE_S=128, `hpb = 128/q_len`) assume V3
shapes. For V4 with **H=64 and Q_LEN=1 (per MTP token)**, those constants
don't fit cleanly.

**Decision (Wave-1 spec):** For v1, **MTP decode reuses `mla_v4_decode`
with `compress_ratio=0`**. Specifically, the MTP path is identical to a
ratio=0 base layer for the attention portion. We do **not** port
`mla_mtp_decode_layer` to V4 for v1. If MTP needs Q_LEN > 1 in v2
(multi-token speculation), we add a `mla_v4_mtp_decode` task that follows
the same dual-cache design but parameterized on `q_len`. Open question
remains: §10.

---

## 9. Module-level test plan

After Wave 2 lands these 5 kernels (plus the SWA paged-cache insert from
sparse.md), the module test
`tests/runtime_python/test_mode/test_attention_v4_module_testmode.py`
exercises a full `Attention.forward` (one layer, one step):

1. Instantiate the official `Attention` class (`model.py:436`) with a
   chosen `compress_ratio ∈ {0, 4, 128}` and load real Flash-Base
   shard-0 weights for layer 0/2/3.
2. Build the MPK graph for the equivalent kernel sequence (per §6/§7).
3. Run both with identical input `x[T, dim]` and identical RNG-seeded
   `kv_cache` initial state. For `ratio>0`, also identical compressor
   state and indexer cache.
4. Compare final `attention.forward` output vs the MPK pipeline output
   (which is `wo_b(z.flatten(1))`). Tolerance `rtol=1e-2, atol=1e-2`
   accounts for the FP8 round-trip through `wo_a`.

The module test is the bisection target for the Wave-3 end-to-end gate;
it directly maps "this layer of MPK matches PyTorch for this layer".

---

## 10. Wiring summary

Build sequence inside one Block's attention portion (host Python):

```python
# Pre-attention input GEMMs (V3 reuse)
qr_kv_fused = linear_fp8_layer(layer_input, wqa_wkv_fused_weight)   # [T, 1024+512]
qr_in, kv_in = split_at(qr_kv_fused, 1024)

# (1) Joint RMSNorm
qr_out, kv_out = mla_v4_q_kv_rmsnorm_layer(qr_in, kv_in, q_norm_w, kv_norm_w, eps,
                                            grid_dim=(T, 2), block_dim=(128, 1, 1))

# Q lora-B + per-head RMSNorm + GPT-J RoPE (per-head norm has NO weight, vLLM line 215)
q = linear_layer(qr_out, wq_b_weight)                                # [T, H*D]

# (1.5) sparse.md tasks
#   - per-head q-norm + RoPE (folded into "fused_qnorm_rope_kv_insert")
#   - KV RoPE + FP8 quant + SWA paged-cache insert
#   - For ratio > 0: compressor_layer (writes compressed cache)
#   - For ratio == 4: indexer_q_transform_layer + indexer_score_topk_layer

# (2 / 4) Decode OR prefill+gather
if step_is_decode:
    o_partial, o_lse = mla_v4_decode_layer(
        q=q.view(B, 1, H, D),
        swa_cache=swa_paged,
        compressed_cache=compressed_paged_or_null,
        swa_indices=meta.swa_indices, swa_lens=meta.swa_lens,
        topk_indices=meta.topk_indices_or_null, topk_lens=meta.topk_lens_or_null,
        attn_sink=attn_sink,
        ...,
    )
    o = mla_reduce_layer(o_partial, o_lse, ...)                       # V3 reuse
else:
    gathered_kv = mla_v4_prefill_gather_layer(...)
    combined_indices, combined_lens = host_meta.combine_topk_swa_indices(...)
    o = mla_v4_prefill_layer(q, gathered_kv, combined_indices, combined_lens,
                              attn_sink, ...)

# (5) Inverse RoPE + per-block FP8 quant
o_fp8, o_scale = inv_rope_fp8_quant_o_layer(o, positions, cos_sin_cache, ...)

# Grouped wo_a (FP8 GEMM, V3 moe_w13_fp8_layer-shaped) + wo_b (V3 linear)
z = linear_fp8_layer(o_fp8, o_scale, wo_a_weight, wo_a_scale, ...)    # [T, n_groups, o_lora_rank]
attn_out = linear_layer(z.flatten(1), wo_b_weight)                    # [T, dim]
```

The 5 kernels in this spec correspond to the (1), (2/3/4), and (5)
sections above. The (1.5) section is owned by `sparse.md`, and the
inputs/outputs across the boundary are listed in this doc's §1.3 / §2.3 /
§3.3 / §5.3 tables.

**Files added** (Wave 2 implementation, NOT this spec):

- `include/mirage/persistent_kernel/tasks/blackwell/mla_v4_q_kv_rmsnorm_sm100.cuh`
- `include/mirage/persistent_kernel/tasks/blackwell/mla_v4_decode_sm100.cuh`
- `include/mirage/persistent_kernel/tasks/blackwell/mla_v4_prefill_sm100.cuh`
- `include/mirage/persistent_kernel/tasks/blackwell/mla_v4_prefill_gather_sm100.cuh`
- `include/mirage/persistent_kernel/tasks/blackwell/inv_rope_fp8_quant_o_sm100.cuh`

**Files modified additively**:

- `include/mirage/persistent_kernel/runtime_header.h` — 5 new `TaskType`
  enums.
- `src/kernel/task_register.cc` — 5 new `register_<task>_task` codegen
  entries; one new `register_mla_v4_decode_sm100_task` that templates on
  `HAS_COMPRESSED`.
- `src/kernel/graph.cc` — 5 task-name → register-fn dispatch entries.
- `python/mirage/mpk/persistent_kernel.py` — 5 new layer methods.
- `python/mirage/mpk/persistent_kernel.py` get_default_init_parameters —
  add a second paged-cache triple for compressed cache (additive; default
  empty when no V4 layer is built).

---

## 11. Open questions  **OPEN:**

1. **OPEN: Second paged-cache triple in meta-tensors.** §4.10. Wave-1
   audit task per plan Phase A must confirm whether MPK's existing
   `paged_kv_indptr_buffer` triple
   (`python/mirage/mpk/persistent_kernel.py:48-50`) can carry **two**
   logical caches by widening the indices, or whether we need a parallel
   `paged_compressed_kv_indptr_buffer` triple. This spec assumes the
   parallel-triple approach for clarity. **Resolve before Wave 2.**

2. **OPEN: FP8 vs. bf16 SWA cache row layout for v1.** §4.4. vLLM stores
   the SWA cache as FP8 (nope) + bf16 (rope) + per-block scales + pad
   (`deepseek_v4_attention.py:217-223`); V3's MPK cache is bf16
   throughout. For v1 correctness we propose **bf16 throughout** to avoid
   forcing a quant kernel during cache insert. This adds memory cost but
   the v1 subset-of-layers target (1–3 layers, plan §Outcome) is tiny.
   Confirm with plan owner before Wave 2.

3. **OPEN: `combine_topk_swa_indices` location.** §7.2. vLLM does this
   inside a Python helper on every prefill chunk
   (`deepseek_v4_attention.py:969-981`). For v1 we propose to keep this
   on the **host** (Python builder) since prefill is not on the
   tight-loop critical path; the function only needs to write into a
   pre-allocated device buffer. v2 may fuse it into the gather task.

4. **OPEN: Attn_sink semantics across splits.** §2.8. FlashMLA-Sparse
   gives the sink one shared softmax slot. In a split decode kernel, only
   one split should add the sink contribution (or each adds 1/num_splits
   then the reduce reassembles). Confirm with FlashMLA reference and
   document precisely in the kernel `_generated_cuda/mla_v4_decode.cu`.

5. **OPEN: Padded heads for MPK.** vLLM pads `H` to 64 or 128 because
   FlashMLA-Sparse demands it (`deepseek_v4_attention.py:151-161`).
   V4-Flash has H=64 exactly, so padding is a no-op. For non-Flash
   variants (Pro / non-Base Flash) with different H, we may revisit.
   v1 hard-codes `padded_heads = 64`.

6. **OPEN: MTP decode reuse.** §8 — confirmed for v1 that MTP reuses
   `mla_v4_decode` with `compress_ratio=0`. If v2 introduces Q_LEN > 1
   (speculative decoding) for MTP, a new `mla_v4_mtp_decode` task is
   needed; document at that time.

7. **OPEN: Inverse-RoPE conjugate sign convention.** §5.2/§5.9. The
   Triton kernel and PyTorch `apply_rotary_emb(inverse=True)` must give
   bit-identical results. Validate this in the §5.9 test by constructing
   inputs where forward RoPE then inverse RoPE round-trips to the input
   within bf16 tolerance.

8. **OPEN: `tma_aligned_T` rounding interaction with the variable-T
   decode path.** §5.6. In decode `T = B` (one query per request, so
   small T). Whether `tma_aligned_T = round_up_4(B)` introduces enough
   padding rows to affect the wo_a GEMM input layout needs verification
   against vLLM's downstream `fp8_einsum` consumer
   (`deepseek_v4_attention.py:326-334`). Likely fine since `o_fp8` strides
   `(d, T*d, 1)` were chosen for exactly this case.
