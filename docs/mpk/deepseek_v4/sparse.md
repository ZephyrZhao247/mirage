# DeepSeek V4-Flash — Sparse-Attention Tasks (Compressor + Indexer)

**Scope.** This spec covers the three new MPK tasks required to port the
sliding-window + KV-compressed + top-K-sparse attention machinery from
DeepSeek V4-Flash:

1. `compressor_layer`        — gated-softmax KV compression + RMSNorm + RoPE
                                 + FP8 quant + paged-cache insert (head=512 for
                                 the attention compressor, head=128 for the
                                 indexer compressor).
2. `indexer_q_transform_layer` — low-rank Q expand + GPT-J RoPE + Hadamard
                                  rotation + FP8 (or MXFP4) per-token quant
                                  with UE8M0 scale and weight-fold.
3. `indexer_score_topk_layer` — FP8/FP4-paged MQA logits × per-head weights →
                                 reduce-over-heads → top-K (512) over the
                                 compressed-KV positions, written into the
                                 attention metadata's `topk_indices` buffer.

This is the Wave-1 sparse spec described in
`/home/zepengz/.claude/plans/i-want-to-add-dapper-pascal.md:185-189` and
`Wave-2 task list / Sparse-attention tasks`
(`/home/zepengz/.claude/plans/i-want-to-add-dapper-pascal.md:255-260`).

Cross-references in this document:
- `attention.md` — owns `mla_v4_decode_layer`, `mla_v4_prefill_gather_layer`,
  `mla_v4_prefill_layer`, `inv_rope_fp8_quant_o_layer`. The
  compressor's compressed-KV cache and the indexer's topk-indices buffer are
  *consumed* by `attention.md` tasks; this spec only writes them.
- `overview.md` — owns the per-layer `compress_ratios` map and the model
  config table. The dispatch table in §A below is the local copy.

**Plan binding.** Per `i-want-to-add-dapper-pascal.md:75-131` (Mandatory
kernel-authoring requirements), every kernel section in this spec includes:
explicit grid design (§G), `/add-mpk-task` conformance checklist (§H),
test-mode unit test plan (§J), V3 reuse table (§K), and a "migrate, don't
reinvent" citation pointing to the vLLM/official source (§E).

---

## §A. Per-layer dispatch map

DeepSeek V4-Flash Base has `n_layers = 43` base transformer layers plus
`n_nextn_predict_layers = 1` (MTP). `compress_ratios` is the 44-entry array
verified at
`/raid/catalyst/models/DeepSeek-V4-Flash-Base/config.json:66`
and
`/raid/user_data/zepengz/projects/mirage_2/deps/deepseek_v4/DeepSeek-V4-Flash/inference/config.json:34`:

```
compress_ratios = [
  0, 0,        # layers 0, 1
  4, 128,      # layers 2, 3
  4, 128,      # layers 4, 5
  4, 128,      # ...
  4, 128,      # layers 40, 41
  4,           # layer 42
  0            # layer 43 (MTP)
]
```

Layer-by-layer dispatch (verified against `Attention.__init__` /
`Attention.forward` in `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:439-482`
and `Compressor`/`Indexer` constructors in lines 283-298 / 384-400):

| Layer idx | `compress_ratio` | `has_attn_compressor` | `has_indexer` (=> has indexer_compressor) | `overlap` | `topk_static` |
|---|---|---|---|---|---|
| 0 | 0 | No | No | n/a | n/a |
| 1 | 0 | No | No | n/a | n/a |
| 2 | 4 | Yes | Yes | True | False (per-step topk) |
| 3 | 128 | Yes | No | False | True (precomputed at metadata-build) |
| 4 | 4 | Yes | Yes | True | False |
| 5 | 128 | Yes | No | False | True |
| … | (alternating 4,128 through layer 41) | | | | |
| 40 | 4 | Yes | Yes | True | False |
| 41 | 128 | Yes | No | False | True |
| 42 | 4 | Yes | Yes | True | False |
| 43 (MTP) | 0 | No | No | n/a | n/a |

Tally: 21 layers with `ratio=4`, 20 layers with `ratio=128`, 3 layers with
`ratio=0`. The Indexer is constructed **only** when `compress_ratio == 4`
(`model.py:468-471`). When `compress_ratio == 128`, the topk is computed
*statically* at metadata-build time (the compressed pool is sparse already —
one slot every 128 tokens — and the topk simply selects the
`min(index_topk=512, total_compressed_pool_size)` newest entries; per plan
§3 this is metadata work, not a GPU task).

---

## §B. Compress-ratio semantics (which tasks fire)

| `ratio` | `compressor_layer` (head=512) | `indexer_q_transform_layer` | `indexer_score_topk_layer` | indexer's `compressor_layer` (head=128) | Attn caches read | Attn caches written |
|---|---|---|---|---|---|---|
| 0   | — | — | — | — | SWA only | SWA only |
| 4   | runs every step, writes when `(pos+1)%4==0`, `overlap=True` | runs every step | runs every step | runs every step, writes when `(pos+1)%4==0`, `overlap=True` | SWA + compressed (head=512) gated by per-token `topk_idxs` from Indexer | SWA + (boundary tokens →) compressed |
| 128 | runs every step, writes when `(pos+1)%128==0`, `overlap=False` | — | — | — | SWA + compressed (head=512), `topk_idxs` is **static** (precomputed) | SWA + (boundary tokens →) compressed |

Cross-ref to `attention.md`: the `mla_v4_decode_layer` always reads the
attention SWA cache; when `compress_ratio != 0` it additionally reads the
compressed-KV cache *and* the per-token `topk_indices` (filled by this
spec's tasks when `ratio==4`, by metadata when `ratio==128`). For `ratio==0`
no `topk_indices` exist and the kernel takes its `ratio==0` branch.

---

## §C. Key dimensions (verified)

| Symbol | Value | Source |
|---|---|---|
| `hidden_size`              | 4096 | `config.json` (root level) |
| `q_lora_rank`              | 1024 | `model.py:58` |
| `head_dim_attn`            | 512  | attention `head_dim`, `model.py:59` |
| `rope_head_dim`            | 64   | `model.py:60` |
| `nope_head_dim_attn`       | 448  | = 512 − 64, derived |
| `head_dim_compressor`      | 512 (attn compressor) / **128 (indexer compressor)** | `model.py:289` |
| `compress_ratio`           | 4 or 128 per layer | §A |
| `coff` (compressor coff)   | `1 + (ratio==4)` = 2 if `overlap` else 1 | `model.py:292`, `deepseek_compressor.py:207` |
| `sliding_window`           | **128** | `model.py:64`, `i-want-to-add-dapper-pascal.md:54` |
| `index_n_heads`            | **64** | `model.py:75` |
| `index_head_dim`           | **128** | `model.py:75` |
| `index_topk`               | **512** | `model.py:76` |
| `compress_rope_theta`      | **160000** in V4-Flash Base config (default in `model.py:67` is 40000; checkpoint config overrides) | `i-want-to-add-dapper-pascal.md:59` |
| `rope_theta` (base)        | 10000 | `model.py:69` |
| `block_size` (FP8 group)   | 128 (linear quant), 64 (Compressor head=512 quant), 128 (Compressor head=128 quant), 32 (MXFP4) | `fused_compress_quant_cache.py:61, 251, 428` |
| `MXFP4_BLOCK_SIZE`         | **32** | `fused_indexer_q.py:8` |
| `FP8_MAX`                  | 448.0 | `fused_compress_quant_cache.py:60` |
| Paged cache page (attn compressor) | block_size = 4 for ratio=4, 8 for ratio=128 (FlashMLA 576-B alignment) | `deepseek_compressor.py:154-158` |

Throughout this doc the unqualified symbol `D_h` denotes the
*per-compressor* `head_dim`, i.e. 512 for the attention compressor (called
from `Attention.forward`) and 128 for the indexer's internal compressor
(`Indexer.compressor` in `model.py:398`).

---

## §D. Note: official PyTorch reference vs vLLM production

**Important.** The official `deps/deepseek_v4/.../inference/model.py`
*simulates* FP4 on the indexer side: `Indexer.forward` calls
`fp4_act_quant(q, fp4_block_size, True)` (`model.py:416`) and `q = rotate_activation(q)`
(`model.py:414`) but writes the indexer `kv_cache` back as **bf16** (the
buffer is registered with no explicit dtype at `model.py:399`, defaulting to
bf16 via `default_dtype = torch.bfloat16` at `model.py:19`). The score
einsum `torch.einsum("bshd,btd->bsht", q, self.kv_cache[...])` at
`model.py:420` therefore runs in bf16 with QAT-style noise.

**vLLM** (`deps/vllm/vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py`
+ `deps/vllm/vllm/model_executor/layers/deepseek_compressor.py`
+ `deps/vllm/vllm/model_executor/layers/sparse_attn_indexer.py`)
materializes both paths for real:

- The attention compressor (head_dim=512) writes a paged cache with
  448 FP8 bytes (nope) + 128 bf16 bytes (rope, 64 half-pairs × 2 bytes) per
  token plus 8 UE8M0 scale bytes per token (7 real + 1 pad) —
  `fused_compress_quant_cache.py:70-73`.
- The indexer compressor (head_dim=128) writes either FP8+fp32-scale
  (`fused_compress_quant_cache.py:260-263`) or **real MXFP4**
  (`fused_compress_quant_cache.py:437-446`) — 64 nibble bytes + 4 UE8M0
  scale bytes per token.
- The indexer Q transform produces either FP8 (`fused_indexer_q.py:67`)
  or MXFP4 (`fused_indexer_q.py:172`).

**MPK v1 follows vLLM**: real FP8 / real MXFP4 paged storage. We do *not*
implement the bf16-with-QAT simulation. Per
`i-want-to-add-dapper-pascal.md:65-73`, v1 targets per-layer correctness
to bf16 tolerance against the *official* `model.py`; the FP8/FP4 paths
introduce numerical drift on top of that and so the FP4 tests run with
looser tolerances (see §M and `i-want-to-add-dapper-pascal.md:277-279`).

**v1 default for the indexer compressor.** Per the discussion at
`i-want-to-add-dapper-pascal.md:432-433`, the Hadamard rotation in v1 is a
naive matmul against a stored Hadamard constant (see §N). To bound the
implementation surface for v1 we default the **indexer kv-cache to FP8**
(the `_fused_kv_compress_norm_rope_insert_indexer_attn` variant at
`fused_compress_quant_cache.py:220-391`) and leave the MXFP4 variant
(`_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn` at
`fused_compress_quant_cache.py:397-584`) as a v2 follow-up. The Q-side
quant likewise defaults to FP8 in v1
(`fused_indexer_q.py:67-169`). The MXFP4 path on both sides is fully
specified in this doc for v2.

**OPEN:** Confirm with reviewers that FP8 indexer cache is the v1 target
(matches `i-want-to-add-dapper-pascal.md:432-433` — "v1 may compute
Hadamard naively"). If v1 *must* match vLLM's FP4 default (set by
`use_fp4_indexer_cache` in `deepseek_v4_attention.py:1059`), promote the
MXFP4 task variants to v1 scope.

---

## §E. Hadamard policy

DeepSeek V4-Flash's Compressor (when `rotate=True`, i.e. the indexer's
internal compressor — `model.py:398`, `deepseek_compressor.py:184-193`)
and the Indexer's Q transform (`model.py:414`) both call
`rotate_activation(x)`, which is

```python
# model.py:247-251
def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    """Applies randomized Hadamard rotation to spread information across dims before FP8 quant."""
    assert x.dtype == torch.bfloat16
    from fast_hadamard_transform import hadamard_transform
    return hadamard_transform(x, scale=x.size(-1) ** -0.5)
```

The rotated dim is `index_head_dim = 128` on both sides (the Q has shape
`[bsz, seqlen, n_heads, head_dim=128]` after unflatten at
`model.py:412`; the indexer compressor's input is `[bsz, seqlen,
coff*head_dim=2*128]` — the rotation in `rotate_activation` applies along
the last dim and inside `_fused_kv_compress_norm_rope_insert_indexer_*` is
applied after the compressed pool collapses the `coff*head_dim` dim down to
`head_dim` via gated softmax, so the rotated width is exactly 128 on the
write path).

**v1 policy (per `i-want-to-add-dapper-pascal.md:432-433`).** The Compressor
and Indexer-Q both perform the Hadamard rotation as a plain GEMM:

```
y[..., j] = sum_i x[..., i] * H[i, j] * (1 / sqrt(128))
```

where `H ∈ {-1, +1}^{128×128}` is a normalized Hadamard matrix. We use
Sylvester's construction (`H_n = H_{n/2} ⊗ H_2`, with `H_2 =
[[1,1],[1,-1]]`), seeded from a fixed constant so both Compressor and
Indexer-Q use the **same** matrix (the reference `fast_hadamard_transform`
implements a deterministic Hadamard transform of length-128 starting from
the canonical `H_128`). The matrix is stored as a static
`__device__ __constant__` bf16 array of length 128·128 = **16 KiB**, well
within constant-memory limits.

In Python, the constant is materialized once at import time of the V4 model
(generated by Sylvester construction in
`python/mirage/mpk/models/deepseek_v4/builder.py`, to be added in Wave 3;
written into the CUDA constant via `cudaMemcpyToSymbol` or hardcoded into
the generated CUDA source by `task_register.cc`). **OPEN:** decide between
(a) host-side init via cudaMemcpyToSymbol (cleanest reuse, requires runtime
wiring) and (b) literal-embedded constant table in the codegen.

v2 will replace this with the in-register fast Hadamard transform
(`hadamard_transform` from `fast-hadamard-transform`, ~10 fused passes for
n=128).

---

## §F. The three tasks

Each subsection has the same structure:

1. Task name + `.cuh` filename + when it runs.
2. Math step-by-step.
3. Inputs / Outputs / State / Cache-layout tables (incl. byte-level).
4. Source to migrate (vLLM + line range, with a 10–25-line excerpt).
5. Generated CUDA reference path (`docs/mpk/deepseek_v4/_generated_cuda/<kernel>.cu`).
6. Grid design.
7. `/add-mpk-task` conformance checklist.
8. Initial (naive) implementation strategy.
9. Test-mode unit-test plan.
10. Reuse from V3.

---

### §F.1 `compressor_layer` (task `compressor_sm100`)

**File.** `include/mirage/persistent_kernel/tasks/blackwell/compressor_sm100.cuh`.

**When it runs.** Layers with `compress_ratio ∈ {4, 128}` (the 41 base
layers 2..42 inclusive). The task is dispatched per token **every step**,
but only commits a write to the compressed-KV cache when
`(positions[token_idx] + 1) % compress_ratio == 0` — the boundary check at
`fused_compress_quant_cache.py:81`. All other tokens still execute the
partial-state update (the score-state `+= ape[pos % ratio]` term + KV
write into the ring buffer) but exit before quant/insert.

The task has **two operating modes**, selected by a templated
`HEAD_DIM` constexpr (a Python-side branch picks which task name to
register — see §H.4):

- **`HEAD_DIM=512`** (attention compressor): used by
  `Attention.compressor` in `model.py:466-467`. nope=448 FP8 + rope=64
  bf16; 7 UE8M0 scales + 1 pad byte per token; one block per
  COMPRESS_RATIO=4 or 128.
- **`HEAD_DIM=128`** (indexer compressor): used by
  `Indexer.compressor` in `model.py:398`. v1 default: all-FP8, single
  fp32 scale per token (4 bytes). v2 MXFP4 layout fully spec'd here for
  reference but not implemented in Wave 2.

#### F.1.1 Math

For one boundary token at `position = p` where `(p+1) % R == 0` (R = compress_ratio):

1. **Gather state**: read `(1+overlap)·R` previous partial states from the
   ring buffer. With overlap (R=4) we gather 8 entries; without (R=128) we
   gather 128. The starting position is `start = p - (1+overlap)*R + 1`
   (`fused_compress_quant_cache.py:87`). For each gathered token `t`,
   load both `kv_state[req, ...]` (width D_h or 2·D_h, depending on
   overlap) and `score_state[req, ...]` (same width, packed in the second
   half of state_cache's last dim — `STATE_WIDTH` parameter,
   `deepseek_compressor.py:301-302`). For positions before request
   start, mask=-inf for score and 0 for kv.

   Overlap detail (`fused_compress_quant_cache.py:99`,
   `model.py:307-314`): when overlap=True the gathered window splits into
   a leading R rows (with `head_offset = 0`, reading the first D_h half
   of the 2·D_h-wide row) and a trailing R rows (with `head_offset = D_h`,
   reading the second half). This is the official `overlap_transform`'s
   layout: `new_tensor[:, 1:, :ratio] = tensor[..., :, :d]` (the
   second-half values shifted one window left). The Triton kernel folds
   this into a per-row offset on the read pointer.

2. **Gated softmax pool**:
   `score = softmax(score_state, dim=window)` (`compress_quant_cache.py:121`).
   `compressed_kv[d] = sum_t (kv[t, d] * score[t, d])`
   (`fused_compress_quant_cache.py:129`). All in fp32. Output is
   `[HEAD_SIZE]` fp32.

3. **RMSNorm** (fp32): `rrms = rsqrt(mean(x²) + eps)`,
   `normed = x * rrms * rms_w` (`fused_compress_quant_cache.py:131-135`).
   `rms_w` is `Compressor.norm.weight` — same RMSNorm semantics as
   `python/mirage/mpk/persistent_kernel.py: rmsnorm_layer`. Width =
   `head_dim`.

4. **RoPE on last `rope_head_dim = 64` dims** (head=512 path) or
   the last `rope_head_dim = 64` dims of the head=128 path:
   GPT-J interleaved (NOT NeoX), forward-only on the writer side
   (`fused_compress_quant_cache.py:189-208, 347-366`). The position used
   is the boundary-aligned `(p // R) * R`
   (`fused_compress_quant_cache.py:201, 359`). The cos/sin cache is
   precomputed with `rope_theta = compress_rope_theta = 160000` — distinct
   from the base `rope_theta = 10000` used by the SWA path
   (`model.py:475-481`). cos and sin are stored interleaved as
   `cos_sin[max_pos, rope_head_dim]` with cos in `[0, 32)` and sin in
   `[32, 64)` (`fused_compress_quant_cache.py:202-204`).

5. **Hadamard rotation (head=128, indexer compressor only)**: see §E. In
   v1 implemented as a `[128]·[128,128]` matmul against the stored
   constant immediately *after* RoPE.

6. **Quant + write to paged cache**: branch on HEAD_DIM:

   **HEAD_DIM=512, attention compressor**
   (`fused_compress_quant_cache.py:155-187`): FP8 e4m3 per-block quant with
   `QUANT_BLOCK=64`, only over the nope portion (`NOPE_HEAD_DIM=448 →
   7 blocks`). The rope portion stays bf16. Per-block scale is UE8M0
   `2^ceil(log2(absmax / FP8_MAX))`; 7 scale bytes + 1 pad byte stored
   together. Output bytes per token: 448 (FP8 nope) + 128 (bf16 rope =
   64 × 2 bytes) + 8 (scale + pad) = **584 bytes/token**.

   **HEAD_DIM=128, indexer compressor (v1, FP8)**
   (`fused_compress_quant_cache.py:368-391`): single quant block of 128
   elements. `absmax`, `exp = ceil(log2(absmax/FP8_MAX))`,
   `inv = 2^(-exp)`. Multiply, cast to fp8, write 128 bytes. Store
   `2^exp` as a single fp32 scale (4 bytes). Output bytes per token:
   128 (FP8) + 4 (fp32 scale) = **132 bytes/token**.

   **HEAD_DIM=128, indexer compressor (v2, MXFP4)**
   (`fused_compress_quant_cache.py:552-583`): tile into `N_BLOCKS=4`
   blocks of QUANT_BLOCK=32. For each block:
   `amax = max(|even|, |odd|)`, clamp ≥ `6·2^-126`,
   `exp = ceil(log2(amax/6.0))`, ue8m0 byte = exp+127,
   `inv_scale = 2^-exp`. Pack `(even*inv, odd*inv)` as two E2M1 nibbles
   per byte via the `cvt.rn.satfinite.e2m1x2.f32` PTX
   (`fused_indexer_q.py:27-42`). Output bytes per token: 64 (nibbles) +
   4 (UE8M0) = **68 bytes/token**.

#### F.1.2 Inputs / Outputs / State / Cache layout

**Inputs (per task invocation, per token):**

| Name | Shape | Dtype | Source |
|---|---|---|---|
| `kv_score` | `[num_tokens, 2·coff·D_h]` (split into `kv` and `score` halves) | bf16 / fp32 (see note) | Output of `fused_wkv_wgate` GEMM (`deepseek_compressor.py:219-227`). For the attention compressor invoked from `Attention.forward` the GEMM operates in bf16 over the input `x`; for the indexer it is bf16 over `hidden_states`. The `kv` and `score` halves are immediately consumed in fp32 inside the kernel. |
| `positions` | `[num_tokens]` | int32 | Per-token absolute position. Same as MPK's existing `meta_tensors["step"]`/`tokens` infrastructure (see `persistent_kernel.py:563-571`). |
| `slot_mapping` | `[num_tokens]` | int32 | State-cache slot per token. From MPK paged-cache metadata. Negative → padding sentinel; the kernel must early-exit if `slot_id < 0` (`fused_compress_quant_cache.py:77-78`). |
| `block_table` | `[num_reqs, max_pages]` | int32 | Per-request page-id array for the state cache. From MPK paged-cache metadata (`paged_kv_indices_buffer` slice; see V3 reuse in §K). |
| `token_to_req_indices` | `[num_tokens]` | int32 | Maps each token to its owning request, built once per scheduler step (`deepseek_compressor.py:113-121`). |
| `kv_slot_mapping` | `[num_tokens]` | int32 | Per-token compressed-KV-cache slot id (only valid when `(p+1)%R==0`). |
| `cos_sin_cache` | `[max_pos, rope_head_dim]` | bf16 | Precomputed RoPE table at `compress_rope_theta = 160000`. cos in `[:, :rope/2]`, sin in `[:, rope/2:]`. Built once per layer (the layer keeps its own freqs because `compress_rope_theta != rope_theta`). |
| `ape` | `[R, coff·D_h]` | fp32 (stored bf16 in checkpoint, promoted on load — `model.py:294-295`) | Absolute-position embedding added into `score_state`. |
| `rms_norm_weight` | `[D_h]` | bf16 (promoted to fp32) | `Compressor.norm.weight` (`model.py:299`). |
| `wkv`, `wgate` weights | `[D, coff·D_h]` each | bf16 | Handled by an *upstream* FP8 / bf16 linear task (MPK's existing `linear_layer` or `linear_fp8_layer` — see §K). NOT inputs of `compressor_layer` itself. |

**State (the partial-state ring buffer):**

A device tensor owned by the layer:

```
state_cache : [num_blocks_state, block_size_state, 2 * coff * D_h]   fp32
```

For ratio=4: `block_size_state = 4` (`deepseek_compressor.py:154-155`),
`coff=2`, last dim `= 2·2·D_h = 4·D_h`. For ratio=128: `block_size_state = 8`,
`coff=1`, last dim `= 2·D_h`. The last dim packs `[kv_state |
score_state]`, each `coff·D_h` wide (`deepseek_compressor.py:230-235`).
This packing lets the boundary-token gather load both halves with a single
row pointer + an offset (`STATE_WIDTH`,
`fused_compress_quant_cache.py:115-127`).

The two updates per token (boundary or not) are factored into a separate
*pre-pass* `_save_partial_states_kernel` in vLLM
(`fused_compress_quant_cache.py:381-433`); in MPK we fold this directly
into `compressor_layer` to keep the task surface area small (one task per
Compressor invocation per scheduler step):

```
# inside compressor_sm100.cuh, per token, before the boundary check
block_idx = slot_id / block_size_state
pos_in_block = slot_id % block_size_state
base = state_cache + block_idx*stride0 + pos_in_block*stride1
store(base + 0,            kv_t)                          # coff·D_h fp32
store(base + STATE_WIDTH,  score_t + ape[p % R])           # coff·D_h fp32
__threadfence();   # state-cache writes must be visible to gather below
```

Only **after** all CTAs (or all warps within the same CTA, depending on
the grid design — see §G.4) finish the partial-state writes do the
boundary tokens proceed to gather + compress. The vLLM design uses two
separate kernel launches to avoid this fence, accepting the launch
overhead. For MPK we have two choices, both acceptable:

**Option A (chosen for v1):** Issue the partial-state writes during one
task and the boundary-gather+compress during a *separate* task name
(`compressor_state_update_sm100` + `compressor_compress_sm100`). This
mirrors vLLM's two-kernel design exactly and removes the cross-CTA
synchronization problem. The Python `compressor_layer` method then
registers both tasks with sequential edges in the kn_graph so MPK's
scheduler guarantees ordering across thread blocks.

**Option B:** Single task with a device-wide barrier. Rejected for v1
because MPK's persistent-kernel scheduler does not currently expose a
device-wide barrier primitive within a task; cross-task ordering is the
existing supported pattern.

**OPEN:** Verify Option A's two-task design is compatible with MPK's
`kn_graph.register_task` API. Look at how
`mla_kv_gather_layer → mla_decode_layer` chain encodes its
producer/consumer edge — likely a precedent.

**Outputs (when `(p+1)%R==0` only):**

| Name | Shape | Dtype | Source / target |
|---|---|---|---|
| compressed-KV cache (paged) | `[num_blocks_kv, block_size_kv, page_bytes]` | uint8 | Per-token write at slot `kv_slot_mapping[token_idx]`. Layout in §F.1.4. |

When the boundary check fails, the task is a no-op for that token (it
already wrote the partial state at the top).

#### F.1.3 State-cache (ring buffer) byte layout — `state_cache`

```
state_cache.shape = (num_blocks_state, block_size_state, 2 * coff * D_h)
state_cache.dtype = float32          # 4 B / element
state_cache.stride(0) = block_size_state * 2*coff*D_h * 4   # bytes
state_cache.stride(1) =                   2*coff*D_h * 4
state_cache.stride(2) = 4
STATE_WIDTH = coff * D_h            # in elements; kv_state at [0, SW), score at [SW, 2·SW)
```

**Why `block_size_state` varies by ratio.** The state-cache page size is
constrained to match the compressed-KV cache page size *in bytes* so the
two can share physical pages (`deepseek_compressor.py:147-159`). With
`page_size_kv == 256/4 = 64` × `head_size_kv = 584` bytes, the equivalent
state-cache page must be `block_size_state × 2·coff·D_h × 4`. For ratio=4:
`block_size_state=4`, last dim = `4·512 = 2048` floats = 8192 B/row, page
= 32768 B. For ratio=128: `block_size_state=8`, last dim = `2·512 = 1024`
floats = 4096 B/row, page = 32768 B. Both equal **the V3 paged-KV-cache
page size** so MPK's existing paged-cache buffer infra (see §K) can host
them without modification.

**Overlap layout** (ratio=4). In the official PyTorch reference
(`model.py:307-314, 331-359`), the per-token state row of width `2·D_h`
packs:

```
state_row[:D_h]      = current-window slot           (the "overlap-with-next" copy of kv at this token)
state_row[D_h:2*D_h] = previous-window-overlap slot  (the "overlap-with-prev" copy from one window earlier)
```

The boundary gather reads `state_row[:D_h]` for tokens in the leading R
positions (the "incoming" half-window) and `state_row[D_h:]` for tokens in
the trailing R positions (the "outgoing" half-window). The
`head_offset = (tokens >= COMPRESS_RATIO) * HEAD_SIZE` term in
`fused_compress_quant_cache.py:99` is this exact selector.

#### F.1.4 Compressed-KV cache byte layout (per page)

A page holds `block_size_kv` (=64 in the V3 reuse path; matches FlashMLA's
576-B alignment, `deepseek_compressor.py:168`) tokens.

**HEAD_DIM=512 (attention compressor).** Per-token stride = 584 bytes.
Per-page bytes (`fused_compress_quant_cache.py:70-73`):

```
[0,            bs * 576):                 token data (one slot per token)
[bs * 576,     bs * 576 + bs * 8):        UE8M0 scale bytes (7 real + 1 pad / token)
```

`576 != 584` because the per-token data is 448 (FP8 nope) + 128 (bf16 rope
= 64 × 2 bytes) = **576 bytes/token** in the data area, and the 8 scale
bytes live in a separate trailing region. Implementer note: per-token
write at slot `s = kv_slot_mapping[token]` goes to
```
fp8_ptr   = page_base + (s % bs) * 576                          # 576 bytes from here
bf16_ptr  = fp8_ptr + 448                                       # last 128 bytes of the 576
scale_ptr = page_base + bs * 576 + (s % bs) * 8                 # 8 bytes (7 real + 1 pad)
```

Verbatim from `fused_compress_quant_cache.py:137-152`:

```python
# ── KV cache pointers ────────────────────────────────────────────
kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
if kv_slot_idx < 0:
    return
kv_block_idx = kv_slot_idx // kv_cache_block_size
kv_pos_in_block = kv_slot_idx % kv_cache_block_size

cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
fp8_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
scale_ptr = (
    cache_block_ptr
    + kv_cache_block_size * TOKEN_STRIDE
    + kv_pos_in_block * SCALE_DIM
)
```

…with `TOKEN_STRIDE=576`, `SCALE_DIM=8`, `KV_BLOCK_STRIDE` = the full
per-page byte stride supplied at launch (==`bs*576 + bs*8`).

**HEAD_DIM=128 indexer (v1, FP8).** Per-token stride = 132 bytes
(`deepseek_compressor.py:261-263`, `deepseek_v4_attention.py:1101`).
Per-page bytes (`fused_compress_quant_cache.py:260-263`):

```
[0,         bs * 128):                FP8 data (128 bytes/token)
[bs * 128,  bs * 128 + bs * 4):       fp32 scales (4 bytes/token)
```

The scale is a single fp32 (not UE8M0; see
`fused_compress_quant_cache.py:389-391`). It is stored *after* exponent
encoding as `2^exponent` directly, so the downstream
`fp8_fp4_paged_mqa_logits` kernel can multiply by it without re-decoding.

**HEAD_DIM=128 indexer (v2, MXFP4).** Per-token stride = 68 bytes
(`fused_compress_quant_cache.py:437-446`):

```
[0,         bs * 64):                 packed E2M1 nibbles (64 B/token = 128 values, 2/byte)
[bs * 64,   bs * 64 + bs * 4):        UE8M0 scale bytes (4 bytes/token = 4 blocks × 1 byte)
```

The `DeepseekV4IndexerCache` in `deepseek_v4_attention.py:1097-1108`
allocates the *FP8-equivalent* memory size (132 B/token) even for the
MXFP4 path, then strides the kernel's writes through only the first
64+4 bytes — a conservative allocation policy. In MPK we replicate this:
allocate 132 B/token regardless and let the kernel choose its TOKEN_STRIDE
from a constexpr template parameter.

#### F.1.5 Source to migrate

**Primary (per `i-want-to-add-dapper-pascal.md:79-93`):**
- `deps/vllm/vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py:30-214`
  — the head=512 path. Migrate to `compressor_sm100.cuh` with
  `HEAD_DIM=512` constexpr.
- `deps/vllm/vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py:220-391`
  — the head=128 FP8 path. Same `.cuh`, `HEAD_DIM=128`, `USE_MXFP4=false`.
- `deps/vllm/vllm/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py:397-584`
  — the head=128 MXFP4 path (v2).

**Excerpt (the head=512 FP8 quant + scale store, lines 155-187):**

```python
# FP8 UE8M0 quant: cast fp32 → bf16 → fp32 before quant to match reference.
N_QUANT_BLOCKS: tl.constexpr = TRITON_BLOCK_SIZE // QUANT_BLOCK
N_NOPE_BLOCKS: tl.constexpr = NOPE_HEAD_DIM // QUANT_BLOCK  # 7
INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX

quant_input = normed.to(tl.bfloat16).to(tl.float32)
quant_2d = tl.reshape(quant_input, (N_QUANT_BLOCKS, QUANT_BLOCK))
abs_2d = tl.abs(quant_2d)
block_absmax = tl.max(abs_2d, axis=1)  # [N_QUANT_BLOCKS] fp32
block_absmax = tl.maximum(block_absmax, 1e-4)

raw_scales = block_absmax * INV_FP8_MAX
exponents = tl.ceil(tl.log2(raw_scales))
inv_scales = tl.exp2(-exponents)
inv_scales_col = tl.reshape(inv_scales, (N_QUANT_BLOCKS, 1))
x_scaled = quant_2d * inv_scales_col
x_clamped = tl.clamp(x_scaled, -FP8_MAX, FP8_MAX)
x_fp8 = x_clamped.to(tl.float8e4nv)
x_uint8 = x_fp8.to(tl.uint8, bitcast=True)
x_uint8_flat = tl.reshape(x_uint8, (TRITON_BLOCK_SIZE,))

nope_mask = block < NOPE_HEAD_DIM
tl.store(fp8_ptr + block, x_uint8_flat, mask=nope_mask)

scale_idx = tl.arange(0, N_QUANT_BLOCKS)
encoded = exponents + 127.0
encoded = tl.maximum(tl.minimum(encoded, 255.0), 0.0)
tl.store(
    scale_ptr + scale_idx,
    encoded.to(tl.uint8),
    mask=scale_idx < N_NOPE_BLOCKS,
)
tl.store(scale_ptr + N_NOPE_BLOCKS, tl.zeros((), dtype=tl.uint8))
```

The `bf16 → fp32` round-trip on line `quant_input = normed.to(tl.bfloat16).to(tl.float32)`
is **load-bearing for numerical parity** with the official `act_quant`
(`model.py:108-118`); preserve it in CUDA.

**Excerpt (the partial-state save kernel,
`fused_compress_quant_cache.py:381-433`):**

```python
@triton.jit
def _save_partial_states_kernel(
    kv_ptr, kv_stride, score_ptr, score_stride,
    ape_ptr, ape_stride, positions_ptr,
    state_cache_ptr, state_cache_stride0, state_cache_stride1,
    slot_mapping_ptr, block_size,
    HEAD_SIZE: tl.constexpr, TRITON_BLOCK_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr, COMPRESS_RATIO: tl.constexpr,
):
    token_idx = tl.program_id(0)
    slot_id = tl.load(slot_mapping_ptr + token_idx)
    if slot_id < 0:
        return
    block_idx = slot_id // block_size
    pos_in_block = slot_id % block_size
    base_ptr = state_cache_ptr + block_idx*state_cache_stride0 + pos_in_block*state_cache_stride1

    block = tl.arange(0, TRITON_BLOCK_SIZE)
    mask = block < HEAD_SIZE
    kv = tl.load(kv_ptr + token_idx * kv_stride + block, mask=mask)
    tl.store(base_ptr + block, kv, mask=mask)

    position = tl.load(positions_ptr + token_idx)
    ape_row = position % COMPRESS_RATIO
    ape = tl.load(ape_ptr + ape_row * ape_stride + block, mask=mask)
    score = tl.load(score_ptr + token_idx * score_stride + block, mask=mask)
    tl.store(base_ptr + STATE_WIDTH + block, score + ape, mask=mask)
```

**Secondary (official semantics oracle):**
`deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:316-377` (the prefill
+ decode branches of `Compressor.forward`).

#### F.1.6 Generated CUDA reference

Per `i-want-to-add-dapper-pascal.md:88-93`, the implementer is provided
with TileLang-extracted CUDA. Save to:

- `docs/mpk/deepseek_v4/_generated_cuda/compressor_sparse_attn.cu`
  (head=512 path)
- `docs/mpk/deepseek_v4/_generated_cuda/compressor_indexer_fp8.cu`
  (head=128 FP8 path)
- `docs/mpk/deepseek_v4/_generated_cuda/compressor_indexer_mxfp4.cu`
  (head=128 MXFP4 path, v2)

These `.cu` files are produced by the Wave-1 TileLang extraction harness
(not in scope for this spec; per
`i-want-to-add-dapper-pascal.md:88-93` the spec only cites their target
paths). For Triton-only sources (no TileLang counterpart) the implementer
translates the Triton DSL directly from
`fused_compress_quant_cache.py`.

#### F.1.7 Grid design

Per `i-want-to-add-dapper-pascal.md:95-104`, this section is mandatory.

**Natural grid.** `(num_tokens,)` — one program per token. Each program
exits early if `slot_mapping[token] < 0` or `(position+1) % R != 0` (for
the boundary phase).

**Phase split (Option A, §F.1.2).**

- *State-update task* (`compressor_state_update_sm100`): grid =
  `(num_tokens,)`. One CTA per token. The CTA writes 2·coff·D_h fp32
  values to the state-cache ring buffer.
  - For HEAD_DIM=512, `coff∈{1,2}` ⇒ row width ∈ {1024, 2048} floats =
    4 KiB or 8 KiB. With WORKER_NUM_THREADS=256 (Blackwell), 16
    elements/thread suffices: one warp can vectorize 4×32-bit loads, so
    8 warps stride the row.
  - For HEAD_DIM=128, row width ∈ {256, 512} floats = 1 KiB or 2 KiB. Fits
    one warp comfortably.
- *Compress task* (`compressor_compress_sm100`): grid = `(num_tokens,)`.
  One CTA per token. Most CTAs exit at the boundary check. The boundary
  CTAs load the window, do softmax/sum, RMSNorm, RoPE, quant, paged-cache
  write.

**CTA slice from `task_desc->task_metadata`.** The natural per-program
data is `token_idx`. In the MPK persistent-kernel model the kernel is
forbidden from reading `blockIdx.x` for routing
(`i-want-to-add-dapper-pascal.md:107-111`, `CLAUDE.md` Key Concepts). The
slice is therefore taken from `task_desc->task_metadata`:

```
struct CompressorMetadata {
    int token_idx;          // which token this CTA owns
    int compress_ratio;     // 4 or 128 (constexpr at task-build time, repeated for runtime sanity)
    int overlap;            // 0 or 1
    int head_dim;           // 512 or 128
};
```

For per-token tasks (the vast majority of MPK's per-token operators), this
is the same pattern as `rmsnorm_layer` / `linear_layer` already use; the
MPK runtime builds one task descriptor per token automatically via the
`new_input(..., -1, True)` partitioning on `num_tokens`-leading-axis
inputs (see existing
`python/mirage/mpk/persistent_kernel.py:1401-1450` for the `mla_kv_gather`
analogue). The token_idx field comes from
`task_desc->task_metadata[0]`. **`blockIdx` is unused** apart from intra-CTA
thread/warp identity (`blockIdx` of the *physical* worker block does not
correspond to `token_idx`).

**Runtime partitioning policy.** With `num_tokens` per scheduler step
(typically batch_size × {1 for decode | seqlen for prefill}, capped by
`max_num_batched_tokens`), the MPK scheduler dispatches `num_tokens` task
descriptors. Worker count is the persistent-kernel block budget (fixed at
launch, typically 132 thread blocks on B200); the scheduler load-balances
across them.

**Alignment.** None on grid.x. Within each CTA, the row width
(2·coff·D_h) is a multiple of 32 floats in all configurations
(512, 1024, 2048, 4096), so vectorized loads (float4 → 4 elements/lane)
align naturally.

**Block size.** `block_dim = (256, 1, 1)` for the head=512 path
(matches `WORKER_NUM_THREADS=256` for Blackwell per CLAUDE.md). For the
head=128 path, `block_dim = (128, 1, 1)` is sufficient (one warp does
softmax over 8 entries when overlap=False; or 8 entries when overlap=True
and ratio=4 ⇒ window 8) but we keep 256 for uniformity.

#### F.1.8 `/add-mpk-task` conformance checklist

Per `i-want-to-add-dapper-pascal.md:106-126`:

- [x] **blockIdx-agnostic.** `compressor_state_update_sm100` and
  `compressor_compress_sm100` read `token_idx` exclusively from
  `task_desc->task_metadata[0]`. No `blockIdx.x/y/z` is referenced for
  routing.
- [x] **TaskType enum entries.** Add `TASK_COMPRESSOR_STATE_UPDATE_SM100`
  and `TASK_COMPRESSOR_COMPRESS_SM100` to
  `include/mirage/persistent_kernel/runtime_header.h`.
- [x] **Codegen entries.** Add `register_compressor_state_update_task` and
  `register_compressor_compress_task` to `src/kernel/task_register.cc`,
  following `register_mhc_*_task` / `register_linear_*` patterns.
- [x] **Dispatch entries.** Add the two task names to the
  task-name → registration-function dispatch in `src/kernel/graph.cc`.
- [x] **Layer-method pattern.** A single Python method
  `compressor_layer(self, ..., compress_ratio, head_dim, overlap, ...)` in
  `python/mirage/mpk/persistent_kernel.py` constructs two TBGraphs (one
  for the state-update task, one for compress), passes the metadata, and
  calls `register_task` twice. Pseudocode:

```python
def compressor_layer(
    self,
    kv_score: DTensor,          # [T, 2·coff·D_h] bf16 (split into kv, score halves)
    positions: DTensor,         # [T] int32
    state_cache: DTensor,       # paged ring buffer
    kv_cache_compressed: DTensor,  # paged compressed-KV cache
    ape: DTensor,
    rms_norm_weight: DTensor,
    cos_sin_cache: DTensor,
    block_table: DTensor,
    slot_mapping: DTensor,
    kv_slot_mapping: DTensor,
    token_to_req_indices: DTensor,
    grid_dim: tuple,
    block_dim: tuple,
    compress_ratio: int,
    head_dim: int,
    overlap: bool,
    rope_head_dim: int = 64,
    use_mxfp4_cache: bool = False,    # v1: always False
):
    params_state = [compress_ratio, head_dim, int(overlap)]
    tb_state = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    # bind inputs with new_input(..., partition_map, ...) — pattern from rmsnorm_layer
    tb_state.new_input(kv_score, (0, -1, -1), -1, True)
    tb_state.new_input(positions, (-1,), -1, True)
    tb_state.new_input(state_cache, (-1, -1, -1), -1, True)
    tb_state.new_input(ape, (-1, -1), -1, True)
    tb_state.new_input(slot_mapping, (-1,), -1, True)
    self.kn_graph.customized([kv_score, positions, state_cache, ape, slot_mapping], tb_state)
    self.kn_graph.register_task(tb_state, "compressor_state_update_sm100", params_state)

    params_c = [compress_ratio, head_dim, int(overlap), int(use_mxfp4_cache), rope_head_dim]
    tb_c = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_c.new_input(state_cache,     (-1, -1, -1), -1, True)
    tb_c.new_input(kv_cache_compressed, (-1, -1, -1), -1, True)
    tb_c.new_input(positions, (-1,), -1, True)
    tb_c.new_input(slot_mapping, (-1,), -1, True)
    tb_c.new_input(kv_slot_mapping, (-1,), -1, True)
    tb_c.new_input(block_table, (-1, -1), -1, True)
    tb_c.new_input(token_to_req_indices, (-1,), -1, True)
    tb_c.new_input(cos_sin_cache, (-1, -1), -1, True)
    tb_c.new_input(rms_norm_weight, (-1,), -1, True)
    self.kn_graph.customized(
        [state_cache, kv_cache_compressed, positions, slot_mapping, kv_slot_mapping,
         block_table, token_to_req_indices, cos_sin_cache, rms_norm_weight], tb_c)
    self.kn_graph.register_task(tb_c, "compressor_compress_sm100", params_c)
```

- [x] **Rebuild.** Recompile via `pip install -e . -v --no-deps`
  (CLAUDE.md "Recompiling after C++ changes").

#### F.1.9 Initial (naive) implementation strategy

- **Plain CUDA, one block per token, 256 threads.** No warp specialization
  beyond the natural reduction patterns. No persistent inner-loop tiling
  beyond what's required for the row width.
- **All reductions in registers, then warp-shuffle.** Softmax over the
  window (`max R·coff = 256` entries when ratio=128 — fits one warp's
  shuffle tree). RMSNorm over 512 elements (head=512) or 128 (head=128).
- **Per-block FP8 quant** translated directly from
  `fused_compress_quant_cache.py:160-187`: each warp owns one
  `QUANT_BLOCK=64` chunk of the nope; absmax via `__shfl_xor_sync`;
  `ceilf(log2f(absmax/448.0))`; cast via `__nv_fp8_e4m3` /
  `__nv_cvt_float_to_fp8`. Reuse the helpers in
  `include/mirage/persistent_kernel/tasks/blackwell/per_token_group_quantize_fp8.cuh:33-37`
  (`encode_ue8m0`).
- **Hadamard (head=128 indexer-compressor only)**: §E — naive 128×128
  matmul against the stored Hadamard constant, in fp32, before quant.
- **No TMA, no `tcgen05.mma`.** v1 path is correctness-first; the entire
  Compressor is memory-bound (the bulk is the gather + ring-buffer write,
  not GEMM). TMA can be revisited in v2 if profiling demands it.
- **GPT-J interleaved RoPE.** Read the (cos, sin) pair for each of the
  `rope_head_dim/2 = 32` rotation pairs from the `cos_sin_cache`,
  compute the rotation in fp32 register, store back as bf16 for the
  head=512 path or fold into the FP8 quant input for the head=128 path.
  Reuse no V3 RoPE task — V3's MLA decode applies its own RoPE
  in-kernel.

#### F.1.10 Test-mode unit test plan

Per `/test-mode` and `i-want-to-add-dapper-pascal.md:222-225`:

- **File:** `tests/runtime_python/test_mode/test_compressor_testmode.py`.
- **Shapes.** Two configurations (one per `HEAD_DIM`):
  - `head_dim=512`, `compress_ratio=4`, `overlap=True`, `T=8` tokens (so
    we cross at least one boundary), `B=1` request, `block_size_state=4`.
  - `head_dim=128`, `compress_ratio=128`, `overlap=False`, `T=128` tokens,
    `B=1`, `block_size_state=8`. (`ratio=128, head_dim=128`
    combination doesn't arise in real V4-Flash but is a useful corner
    test for the `coff=1` code path with the indexer head width.)
  - One canonical real-config test: `head_dim=128`, `compress_ratio=4`,
    `overlap=True`, `T=8`.
- **Oracle.** Extract `Compressor.forward` from
  `model.py:316-377` into a torch-only snippet; feed the same inputs.
- **Tolerances** (per `i-want-to-add-dapper-pascal.md:277-279`):
  - bf16 paths (rope stores, state writes): `rtol=1e-3, atol=1e-3`.
  - FP8 paths (FP8 nope, FP8 indexer cache): dequantize then compare in
    fp32 with `rtol=5e-2, atol=5e-2` — same loose bound used by vLLM's
    own `tests/kernels/test_compressor_kv_cache.py`.
  - MXFP4 paths (v2 only): `rtol=1e-1, atol=1e-1`.
- **Test setup.** Use `PersistentKernel.get_default_init_parameters()`
  with `params["test_mode"] = True`, then invoke `pk()`. Only the
  meta-tensors the compressor reads (positions, slot_mapping,
  token_to_req_indices, block_table) need to be populated.
- **Assertions on cache layout.** Independent of numerical correctness,
  the test must verify:
  - `state_cache[boundary_token, :D_h]` matches the official's
    `self.kv_state[bsz, ratio + (start_pos % ratio)]`.
  - `kv_cache[kv_slot_mapping[boundary_token], :448]` reinterpreted as
    `float8_e4m3fn` × `2^(scale_byte - 127)` reconstructs the official's
    `nope` portion within FP8 tolerance.
  - `kv_cache[..., 448:576].view(torch.bfloat16)` matches the rotated
    rope portion within bf16 tolerance.

#### F.1.11 Reuse from V3

- **Paged-KV-cache infrastructure** — `paged_kv_indptr_buffer`,
  `paged_kv_indices_buffer`, `paged_kv_last_page_len_buffer`
  (`persistent_kernel.py:48-50, 563-571, 3205-3314`). These map directly
  to the Compressor state cache **and** the compressed-KV cache, both of
  which use the same 64-token / 32 KiB page model.
  - The attn compressor's compressed cache: pages of
    `block_size_kv × 584` bytes; allocate via the existing
    `paged_kv_*` buffer API by setting `head_size=584`,
    `dtype=torch.uint8` at meta-tensor build time.
  - The indexer's compressed cache: pages of `block_size_kv × 132` bytes
    (FP8). Same allocation API.
  - The state cache: pages of `block_size_state × 2·coff·D_h × 4` bytes;
    same API.
  - The block_table is the same `paged_kv_indices_buffer` slice already
    used for the V3 attention KV cache.
- **`rmsnorm_layer`** — NOT directly reusable; the Compressor's RMSNorm is
  *inside* the same fused kernel as the pool and the RoPE. A
  separate-kernel RMSNorm would force a fp32 round-trip through HBM which
  the fused vLLM kernel explicitly avoids
  (`deepseek_compressor.py:330-335`).
- **`quantize_fp8_layer`** (`persistent_kernel.py:1967-1988`) — partially
  reusable. The existing layer registers `quantize_fp8_sm100`
  (scale_ue8m0=True) or `quantize_fp8_f32scale_sm100` (scale_ue8m0=False);
  the Compressor wants UE8M0 with `GROUP_SIZE=64` (not the default 128)
  for the head=512 path. The
  `per_token_group_quantize_fp8_task_impl` template
  (`per_token_group_quantize_fp8.cuh:39-130`) is already
  parametric on `GROUP_SIZE` so the *kernel template* is reusable; only a
  new task-registration entry with `GROUP_SIZE=64` is needed for the
  Compressor's nope. We inline this inside `compressor_compress_sm100`
  instead, because fusing keeps the data in registers between RMSNorm and
  quant. **Verdict.** Inline, but cite the V3 helper as the algorithmic
  reference.

---

### §F.2 `indexer_q_transform_layer` (task `indexer_q_transform_sm100`)

**File.** `include/mirage/persistent_kernel/tasks/blackwell/indexer_q_transform_sm100.cuh`.

**When it runs.** Layers with `compress_ratio == 4` (the 21 layers with an
Indexer). Runs every decode step on the current query token(s) only — for
prefill, runs on all prompt tokens.

#### F.2.1 Math

Per token, per index head (T=num_tokens, H=index_n_heads=64):

1. **Q expand (upstream, not in this task).** Q is produced by an FP8
   linear `q = wq_b(qr)` where `qr` is the q-LoRA-A normalized output
   (shape `[T, q_lora_rank=1024]`), `wq_b` weight is
   `[index_n_heads * index_head_dim, q_lora_rank] = [8192, 1024]`.
   Output shape after unflatten:
   `q = q.view(-1, n_heads=64, head_dim=128)`
   (`model.py:411-412`, `deepseek_v4_attention.py:1143-1144`). Reuse
   MPK's existing `linear_fp8_layer` for this — it is not part of the
   `indexer_q_transform_layer` task.

2. **GPT-J RoPE on last 64 dims.** Same interleaved (cos, sin) layout as
   the Compressor (§F.1.1). Position used is the *raw* `positions[t]`
   (NOT the compressed-boundary-aligned position used by the Compressor —
   `fused_indexer_q.py:101-117`).

3. **Hadamard rotation** on the full 128-dim (§E). `q ← q · H · (1/√128)`.

4. **Per-token-per-head FP8 quant**
   (`fused_indexer_q.py:124-149`).
   - Compute `absmax = max(|q_nope|, |q_rope_even|, |q_rope_odd|)`
     across all 128 dims of the (token, head).
   - `q_scale = exp2(ceil(log2(max(absmax,1e-4)/448.0)))`.
   - Quantize: `q_fp8 = (q / q_scale).to(float8_e4m3fn)` (saturating).
   - Output: `[T, H, head_dim]` FP8 (no per-token scale tensor stored —
     it is folded into the weight, see step 5).

5. **Weight-fold for FP8 path**
   (`fused_indexer_q.py:159-169`).
   ```
   index_weights_out[t, h] =
       index_weights[t, h] * q_scale[t, h] * softmax_scale * head_scale
   ```
   where `softmax_scale = head_dim^-0.5 = 1/√128`
   (`deepseek_v4_attention.py:1081`) and
   `head_scale = n_heads^-0.5 = 1/√64`
   (`deepseek_v4_attention.py:1152`).

**MXFP4 alternative path (v2, `fused_indexer_q.py:172-279`).** For each of
the 4 MXFP4 blocks of 32 elements per (token, head):
- `amax = max(|even_pairs|, |odd_pairs|)` over the block.
- `ue8m0 = ceil(log2(amax/6.0)) + 127`, clamped to `[0, 255]`.
- Pack `(even, odd)` as two E2M1 nibbles per byte via the PTX
  `cvt.rn.satfinite.e2m1x2.f32` (`fused_indexer_q.py:27-42`).
- Weight-fold: `index_weights_out[t, h] = index_weights[t, h] *
  softmax_scale * head_scale` (NO `q_scale` term —
  the per-block ue8m0 stays with the values,
  `fused_indexer_q.py:264-279`).

#### F.2.2 Inputs / Outputs

| Name | Shape | Dtype | Source |
|---|---|---|---|
| `index_q` | `[T, H=64, head_dim=128]` | bf16 | Output of `wq_b` linear (upstream). |
| `positions` | `[T]` | int32 | meta-tensor. |
| `cos_sin_cache` | `[max_pos, rope_head_dim=64]` | bf16 | Same RoPE table as the *base* attention (theta=10000, NOT compress theta) — `deepseek_v4_attention.py:1149` passes `rotary_emb.cos_sin_cache`, which is the Indexer-specific one in this codepath. **OPEN:** verify whether V4-Flash uses base rope_theta or compress_rope_theta for the Indexer's Q RoPE; `model.py:481` constructs `freqs_cis` for Attention based on `compress_rope_theta` when `compress_ratio>0`, and the Indexer's `freqs_cis` is then taken from the same Attention layer's table (`model.py:410` — `self.compressor.freqs_cis = self.freqs_cis` — but the Indexer's *own* freqs_cis used at line 404 is set externally to it; trace required). |
| `index_weights` | `[T, H]` | bf16 | Output of `Indexer.weights_proj(x)` (`model.py:418`) — a separate FP8 linear that runs upstream. |
| Hadamard constant `H_128` | `[128, 128]` | bf16, `__constant__` | §E. |

| Output | Shape | Dtype | Notes |
|---|---|---|---|
| `index_q_fp8` (FP8 path) | `[T, H, head_dim=128]` | float8_e4m3fn | Quantized Q, scale folded into weights. |
| `index_q_packed` (MXFP4 path, v2) | `[T, H, head_dim/2=64]` | uint8 | Two E2M1 nibbles per byte. |
| `index_q_scale` (MXFP4 path, v2) | `[T, H, head_dim/MXFP4=4]` | uint8 (ue8m0) | Per-block exponents. |
| `index_weights_out` | `[T, H]` | fp32 | Weight-fold result (input to `indexer_score_topk`). |

#### F.2.3 Source to migrate

`deps/vllm/vllm/v1/attention/ops/deepseek_v4_ops/fused_indexer_q.py:67-169`
(FP8 path) and `:172-279` (MXFP4 path). The PTX helper at lines 27-42 is
required verbatim.

**Excerpt (the FP8 RoPE + quant + weight-fold loop, lines 100-169):**

```python
tok_idx = tl.program_id(0)
head_idx = tl.program_id(1)

pos = tl.load(pos_ptr + tok_idx)
cos, sin = _get_cos_sin(index_q_cos_sin_ptr, index_q_cos_sin_stride,
                        pos, INDEX_Q_HALF_ROT_DIM)
half_offset = tl.arange(0, INDEX_Q_HALF_ROT_DIM)
base_ptr = index_q_ptr + tok_idx * index_q_stride0 + head_idx * index_q_stride1

# Interleaved (GPT-J) RoPE on dims [NOPE_DIM, HEAD_DIM):
rot_base = base_ptr + INDEX_Q_NOPE_DIM
x_even = tl.load(rot_base + half_offset * 2).to(tl.float32)
x_odd  = tl.load(rot_base + half_offset * 2 + 1).to(tl.float32)
r_even = x_even * cos - x_odd * sin
r_odd  = x_odd  * cos + x_even * sin

# bf16 roundtrip for parity
r_even = r_even.to(tl.bfloat16).to(tl.float32)
r_odd  = r_odd.to(tl.bfloat16).to(tl.float32)

amax = tl.maximum(tl.max(tl.abs(r_even)), tl.max(tl.abs(r_odd)))
if INDEX_Q_NOPE_DIM > 0:
    nope_offset = tl.arange(0, INDEX_Q_NOPE_DIM)
    x_nope = tl.load(base_ptr + nope_offset).to(tl.float32)
    amax = tl.maximum(amax, tl.max(tl.abs(x_nope)))
index_q_scale = tl.div_rn(tl.maximum(amax, 1e-4), 448.0)
index_q_scale = tl.math.exp2(tl.math.ceil(tl.math.log2(index_q_scale)))

# … FP8 store …

index_weights = tl.load(index_weights_ptr + tok_idx * index_weights_stride + head_idx)
index_weights = index_weights.to(tl.float32)
index_weights *= index_q_scale
index_weights *= index_weights_softmax_scale
index_weights *= index_weights_head_scale
tl.store(index_weights_out_ptr + tok_idx * index_weights_out_stride + head_idx,
         index_weights)
```

**Note on Hadamard.** The vLLM `fused_indexer_q_rope_quant_kernel` does
**NOT** include the Hadamard step — vLLM relies on the upstream
`wq_b` weight matrix having been *pre-rotated* offline so that
`H · wq_b` is the stored weight, equivalently absorbing the rotation
into the weight (`deepseek_v4_attention.py:1066-1072`,
`DeepseekCompressor.fused_wkv_wgate`). The official PyTorch reference
applies `rotate_activation` at runtime (`model.py:414`) — this is the
key semantic difference.

**v1 MPK policy.** We follow vLLM's convention: **pre-rotate `wq_b`
offline** by `H · wq_b` in the weight conversion script
(`demo/deepseek_v4/models/convert.py`, to be added in Wave 3). That way,
the runtime `wq_b` GEMM produces an already-Hadamard-rotated Q, and the
`indexer_q_transform_layer` kernel does NOT need a Hadamard pass. This
matches `i-want-to-add-dapper-pascal.md:432-433` ("v1 may compute
Hadamard naively") with the cheapest naive: don't compute it at runtime
at all.

The Hadamard constant must still be materialized in the conversion
script. The `H_128` constant is the same Sylvester matrix described in
§E. This means **the kernel as specified does NOT contain the matmul** —
§E's "naive matmul against stored Hadamard" applies to the
Compressor's `head_dim=128` path, which has no upstream weight to absorb
into. **OPEN:** confirm that the Compressor's head=128 Hadamard cannot
similarly be absorbed; the Compressor rotates `kv` *after* the gated
softmax pool, so absorbing into `wkv` is not equivalent (the softmax mass
varies per token). Confirm with reviewers.

#### F.2.4 Generated CUDA reference

`docs/mpk/deepseek_v4/_generated_cuda/indexer_q_transform_fp8.cu` and
`indexer_q_transform_mxfp4.cu` (v2).

#### F.2.5 Grid design

**Natural grid.** `(T, H)` — one program per (token, head). vLLM uses
exactly this in `fused_indexer_q_rope_quant_kernel`
(`fused_indexer_q.py:98-99`) and again in the MXFP4 variant
(`fused_indexer_q.py:210-211`).

**CTA slice from `task_desc->task_metadata`.**

```
struct IndexerQMetadata {
    int token_idx;
    int head_idx;
    int use_mxfp4;
    int rope_head_dim;
};
```

Each CTA processes one (token, head) pair, head_dim=128 columns. The CTA
reads `token_idx` and `head_idx` from `task_metadata[0:2]`. No
`blockIdx` reads.

**Runtime partitioning policy.** With T tokens and H=64 heads, the
scheduler dispatches `T·64` task descriptors per step. For a typical
batch of B=4, decode T=4, we get 256 tasks; with B=4 prefill at
seqlen=64, T=256, we get 16384 tasks. The persistent kernel's 132 worker
blocks process these in a round-robin (typical MPK pattern).

**Alignment.** `head_dim = 128` aligns cleanly to 128-bit vector loads
(8 bf16 / lane × 16 lanes / warp). No grid-level alignment constraint.

**Block size.** `block_dim = (128, 1, 1)` — one warp does the full
per-head reduction; the remaining threads handle the per-lane work.

#### F.2.6 `/add-mpk-task` conformance checklist

- [x] **blockIdx-agnostic.** Yes — token_idx and head_idx come from
  task_metadata.
- [x] **TaskType enum.** Add `TASK_INDEXER_Q_TRANSFORM_SM100`.
- [x] **Codegen.** `register_indexer_q_transform_task` in
  `src/kernel/task_register.cc`.
- [x] **Dispatch.** Entry in `src/kernel/graph.cc`.
- [x] **Layer method.**

```python
def indexer_q_transform_layer(
    self,
    index_q: DTensor,          # [T, H, head_dim] bf16
    positions: DTensor,        # [T] int32
    cos_sin_cache: DTensor,    # [max_pos, rope_head_dim] bf16
    index_weights: DTensor,    # [T, H] bf16
    index_q_fp8: DTensor,      # [T, H, head_dim] fp8
    index_weights_out: DTensor,# [T, H] fp32
    grid_dim: tuple,
    block_dim: tuple,
    softmax_scale: float = 1.0 / math.sqrt(128),
    head_scale: float = 1.0 / math.sqrt(64),
    rope_head_dim: int = 64,
):
    params = [rope_head_dim, _pack_f32(softmax_scale), _pack_f32(head_scale)]
    tb = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb.new_input(index_q, (0, 1, -1), -1, True)   # grid.x=tok, grid.y=head
    tb.new_input(positions, (-1,), -1, True)
    tb.new_input(cos_sin_cache, (-1, -1), -1, True)
    tb.new_input(index_weights, (0, 1), -1, True)
    tb.new_input(index_q_fp8, (0, 1, -1), -1, True)
    tb.new_input(index_weights_out, (0, 1), -1, True)
    self.kn_graph.customized(
        [index_q, positions, cos_sin_cache, index_weights, index_q_fp8,
         index_weights_out], tb)
    self.kn_graph.register_task(tb, "indexer_q_transform_sm100", params)
```

#### F.2.7 Initial (naive) implementation

- One block per (token, head). 128 threads, one warp + a bit; head_dim=128
  means each lane handles 4 bf16 / fp32 elements.
- Load 128 bf16 into registers. (Re-Hadamard not needed; see §F.2.3.)
- RoPE on dims `[64, 128)`: load (cos, sin) once for the 32 pairs, do
  32 fp32 rotations.
- Warp absmax via `__shfl_xor_sync` (3 rounds for warp=32, 1 cross-warp
  round if `block_dim>32`).
- `__expf(__logf(absmax * (1/448.0)) * <=>)` — actually use
  `ceilf(log2f(...))` directly (`compressor_sm100.cuh` shares
  `encode_ue8m0`).
- Cast 128 fp32 → 128 fp8 via `__nv_cvt_float_to_fp8_e4m3` (saturating),
  store as `uint8`.
- One thread writes `index_weights_out[t, h]`.

#### F.2.8 Test-mode unit test plan

- **File:** `tests/runtime_python/test_mode/test_indexer_q_transform_testmode.py`.
- **Shape.** T=8, H=64, head_dim=128. Two oracle modes:
  - Without Hadamard (vLLM-style; `wq_b` pre-rotated): match
    `fused_indexer_q_rope_quant` from
    `fused_indexer_q.py:282-400` numerically.
  - With Hadamard (official-style): match a Python snippet of
    `Indexer.forward` lines 411-418 with `rotate_activation` applied,
    then `fp4_act_quant` simulated.
- **Tolerances.** FP8 path: `rtol=5e-2, atol=5e-2` after dequant.
- **Oracle wiring.** Reuse `fused_indexer_q_rope_quant_kernel` directly
  by importing the vLLM module if available in the test env; otherwise
  re-implement the math in PyTorch (it is short).

#### F.2.9 Reuse from V3

- **`quantize_fp8_layer`** — not directly reused; this task is a *fused*
  RoPE + quant + weight-fold, and splitting would cost an HBM
  round-trip. We cite `per_token_group_quantize_fp8.cuh:33-37` for
  `encode_ue8m0`, and the fp8 cast helpers from
  `per_token_group_quantize_fp8.cuh:39-130`.
- **`linear_fp8_layer`** (`persistent_kernel.py:1990-2015`) — used by the
  *upstream* `wq_b` GEMM that produces `index_q`. Not part of this task.

---

### §F.3 `indexer_score_topk_layer` (task `indexer_score_topk_sm100`)

**File.** `include/mirage/persistent_kernel/tasks/blackwell/indexer_score_topk_sm100.cuh`.

**When it runs.** Layers with `compress_ratio == 4` (the 21 indexed
layers), every step. Consumes outputs of `indexer_q_transform_layer` and
the indexer's compressed-KV cache (written by
`compressor_layer` with `head_dim=128`).

#### F.3.1 Math

Per token `t`, with `q_fp8[t, h, :]` (FP8) and per-head weight
`w[t, h] = index_weights_out[t, h]` (fp32, already containing
`q_scale[t, h] * softmax_scale * head_scale`):

1. **Logits.** For each candidate position `j ∈ [0, end_pos/R)`:
   ```
   logits[t, j] = relu(  Σ_h  q_fp8[t, h, :] · kv_fp8[j, :]  )
                       · sum_h_weighted (folded below)
   ```
   In the FP8 path, the actual fused kernel
   `fp8_fp4_paged_mqa_logits` computes the score as
   ```
   logits[t, j] = sum_h ( relu(q_fp8[t, h, :] · kv_fp8[j, :])
                          * dequant(scales)
                          * w[t, h] )
   ```
   matching the official's
   `index_score = relu(score).sum_over_heads * weights.unsqueeze(-1)`
   (`model.py:420-421`). The order of operations differs (vLLM applies
   relu inside the head sum after the per-head weight; the official
   applies relu before scalar weight multiply and head reduce). For
   `index_weights ≥ 0` (the `weights_proj` output is bf16 with no sign
   constraint; **OPEN:** verify weights_proj output sign in the
   checkpoint by inspecting a layer's weights; if it can be negative, the
   relu-then-weight ordering matters and we must follow the official's
   order to match numerics — this is the more conservative choice).

   v1 MPK follows the **official's** ordering for correctness:
   ```
   for j in 0 .. end_pos/R:
     s_j = 0
     for h in 0 .. H:
       partial = dot(q[t,h,:], kv[j,:])           # 128-d dot, FP8 × FP8 → fp32
       partial = partial * q_scale[t,h] * kv_scale[j]   # dequant
       s_j += relu(partial) * w[t,h]              # w already contains softmax_scale*head_scale
     logits[t, j] = s_j
   ```

2. **Causal mask** (`model.py:425-426`, `model.py:428-430`):
   - Prefill (`start_pos == 0`): for each query token `q_idx`, mask
     positions `j ≥ (q_idx+1) / R` to `-inf`. Equivalently, only
     positions whose source-token range ends ≤ q_idx are valid.
   - Decode (`start_pos > 0`): all positions in `[0, end_pos / R)` are
     valid (no future-position issue since the current token's
     compressed position has not yet been written when it queries).

3. **Top-K.** Top-`min(index_topk=512, end_pos/R)` over the `logits[t, :]`
   row, returning **indices** (not values). Output shape `[T, K]` int32.

4. **Offset + invalid-mask** (`model.py:429-432`):
   - Each topk index is offset by the per-request compressed-cache base
     offset (an additional input).
   - For prefill, additionally mask out any index `j ≥ (q_idx+1)/R`
     (already -inf'd but topk could pick padded -1 entries) by replacing
     with -1.

#### F.3.2 Inputs / Outputs

| Name | Shape | Dtype | Source |
|---|---|---|---|
| `index_q_fp8` | `[T, H=64, 128]` | float8_e4m3fn | Output of §F.2. |
| `index_weights_out` | `[T, H]` | fp32 | Output of §F.2 (weight-fold). |
| `kv_cache_indexer` (paged) | `[num_blocks_idx, block_size_idx, 132]` | uint8 (FP8 + 4-B fp32 scale) | Output of `compressor_layer(head_dim=128, use_mxfp4_cache=False)`. |
| `block_table_indexer` | `[B, max_pages_idx]` | int32 | Paged-cache metadata for the indexer cache (separate from the main attention cache). |
| `seq_lens_indexer` | `[B, q_len]` | int32 | Per-request compressed-pool length (=`(seq_len)/R`). |
| `positions` | `[T]` | int32 | meta-tensor. |
| `request_offsets` | `[B]` | int32 | Per-request base offset in the global topk index space (used at line `topk_idxs += offset` in `model.py:432`). |

| Output | Shape | Dtype | Notes |
|---|---|---|---|
| `topk_indices` | `[T, index_topk=512]` | int32 | Per-query top-K indices into the *global* compressed-pool index space. `-1` marks invalid slots (prefill mask, or fewer than K valid candidates). |

#### F.3.3 Source to migrate

This task fuses two production kernels from vLLM:

- **Logits** = `fp8_fp4_paged_mqa_logits` from DeepGEMM
  (`deepseek_v4_attention.py:14-17` via `vllm.utils.deep_gemm`,
  `sparse_attn_indexer.py:310-319`). DeepGEMM source is in
  `deps/vllm/vllm/third_party/deep_gemm/include/deep_gemm/impls/` (per
  `i-want-to-add-dapper-pascal.md:14-17`). Cite specific source: **OPEN**
  — the path to the FP8 paged MQA logits impl needs to be confirmed by
  grepping the deep_gemm tree; the `fp8_fp4_paged_mqa_logits` python
  binding lives in `deps/vllm/vllm/utils/deep_gemm.py` but the C++/CUDA
  source is in `deps/vllm/vllm/third_party/deep_gemm/include/deep_gemm/impls/`.
- **TopK** = `torch.ops._C.persistent_topk` for `topk ∈ {512, 1024,
  2048}` (the decode fast path,
  `sparse_attn_indexer.py:323-335`), or
  `torch.ops._C.top_k_per_row_prefill` for prefill
  (`sparse_attn_indexer.py:248-258`).

**v1 MPK policy.** A full port of DeepGEMM's `fp8_fp4_paged_mqa_logits`
is well outside Wave-2's per-task scope (the DeepGEMM kernel is a fully
tensorized MQA logits + softmax-scale pipeline with multi-stage
software-pipelined TMA loads, cluster-mode dispatch, and TCGEN05
accumulator regs). For v1 we write a **naive equivalent**:

- One block per `(token, kv_block)` pair where `kv_block` is the page id
  in the indexer cache for this request. Reuse the existing
  `mla_kv_gather_layer` precedent for paged-cache iteration
  (`persistent_kernel.py:1401-1420`).
- Within the block, iterate over `block_size_idx` tokens, dequant FP8 KV
  by multiplying by the page's per-token fp32 scale, dot with `q_fp8[t,
  h, :]` (cast to fp32 first), apply relu, scale by per-head weight, sum
  over heads, and *atomically add* into a per-`token_idx, j_global`
  partial-result buffer.
- Subsequent `topk_reduce_sm100` task runs the top-K. For `K=512` and
  per-token candidate counts up to `max_seq_len/R = 8192` (with R=4 and
  max_seq_len=32768 in v1 tests), a bitonic top-K or radix top-K is
  feasible in one block.

This split:
- `indexer_score_sm100` — produces logits (B·T × num_compressed_positions
  fp32 buffer, sparse-write per page).
- `indexer_topk_sm100` — reads logits, writes topk_indices.

For the spec we name the combined Python-side layer
`indexer_score_topk_layer`. It registers **both** task names internally.

If `K=512` is fixed across the run (it is, per
`i-want-to-add-dapper-pascal.md:55`), and `num_compressed_positions ≤
4096` in v1 tests, a single fused score+topk task per token is feasible:
build a streaming top-K heap of size 512 as you iterate `j`. **OPEN:** v1
should pick the heap-based fused task (simpler, no large intermediate
buffer); revisit if profiling shows the heap update is a bottleneck.

**Excerpt (the official's score+topk logic,
`deps/deepseek_v4/.../inference/model.py:418-432`):**

```python
weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)
# We performed QAT here, kv could also use fp8 format, though current implementation uses bf16
index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, :end_pos // ratio])
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
if world_size > 1:
    dist.all_reduce(index_score)
if start_pos == 0:
    mask = torch.arange(seqlen // ratio).repeat(seqlen, 1) >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
    index_score += torch.where(mask, float("-inf"), 0)
topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
if start_pos == 0:
    mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
    topk_idxs = torch.where(mask, -1, topk_idxs + offset)
else:
    topk_idxs += offset
return topk_idxs
```

#### F.3.4 Generated CUDA reference

`docs/mpk/deepseek_v4/_generated_cuda/indexer_score_topk_fp8.cu`. v2:
`indexer_score_topk_mxfp4.cu`.

#### F.3.5 Grid design

**Natural grid.** `(T,)` — one program per token. Each program iterates
over the request's compressed-pool pages and maintains a streaming top-K
heap.

**CTA slice from `task_desc->task_metadata`.**

```
struct IndexerScoreTopkMetadata {
    int token_idx;
    int request_id;     // for looking up the request's page table & seq_len
    int topk;           // 512 in production; templated for testability
    int compress_ratio; // 4 in production (the only ratio with this task)
};
```

**Runtime partitioning policy.** `T` tasks per step. For a typical
prefill T=B·seqlen=64·64=4096 these queue into the 132 worker blocks.

**Alignment.** None on grid.x.

**Block size.** `block_dim = (256, 1, 1)`. The work per token is
`num_compressed × H = (seq_len/R) × 64` dot products of length 128. With
`seq_len/R=2048` and `H=64`, that's 2048·64 = 131072 fp32 dot-products of
length 128 = 16.78 M MACs per token. At 256 threads, each thread does
65536 / 256 = 256 dots, i.e. one thread processes 256 (j, h) pairs and
streams the partial sum into a per-token-j accumulator in shared memory.
Top-K of size 512 over `num_compressed ≤ 2048` candidates is done as a
fully-warp-parallel radix select or as a partial sort.

#### F.3.6 `/add-mpk-task` conformance checklist

- [x] **blockIdx-agnostic.** Yes — token_idx and request_id come from
  task_metadata.
- [x] **TaskType enum.** Add `TASK_INDEXER_SCORE_TOPK_SM100`.
- [x] **Codegen.** `register_indexer_score_topk_task` in
  `src/kernel/task_register.cc`.
- [x] **Dispatch.** Entry in `src/kernel/graph.cc`.
- [x] **Layer method.**

```python
def indexer_score_topk_layer(
    self,
    index_q_fp8: DTensor,         # [T, H, head_dim] fp8
    index_weights_out: DTensor,   # [T, H] fp32
    kv_cache_indexer: DTensor,    # paged uint8
    block_table_indexer: DTensor, # [B, max_pages]
    seq_lens_indexer: DTensor,    # [B]
    request_offsets: DTensor,     # [B] int32
    topk_indices: DTensor,        # [T, index_topk] int32  (OUTPUT)
    grid_dim: tuple,
    block_dim: tuple,
    index_topk: int = 512,
    head_dim: int = 128,
    n_heads: int = 64,
    compress_ratio: int = 4,
):
    params = [index_topk, head_dim, n_heads, compress_ratio]
    tb = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb.new_input(index_q_fp8, (0, -1, -1), -1, True)
    tb.new_input(index_weights_out, (0, -1), -1, True)
    tb.new_input(kv_cache_indexer, (-1, -1, -1), -1, True)
    tb.new_input(block_table_indexer, (-1, -1), -1, True)
    tb.new_input(seq_lens_indexer, (-1,), -1, True)
    tb.new_input(request_offsets, (-1,), -1, True)
    tb.new_input(topk_indices, (0, -1), -1, True)
    self.kn_graph.customized(
        [index_q_fp8, index_weights_out, kv_cache_indexer, block_table_indexer,
         seq_lens_indexer, request_offsets, topk_indices], tb)
    self.kn_graph.register_task(tb, "indexer_score_topk_sm100", params)
```

#### F.3.7 Initial (naive) implementation

- **Phase 1 (score):** one block per token. For each compressed
  position `j`:
  - Look up the page id from `block_table_indexer[req, j / block_size_idx]`.
  - Compute byte offset to FP8 data + fp32 scale.
  - Load 128 FP8 bytes (=128 fp8 values) into registers; load the fp32
    scale once.
  - For each head h ∈ [0, 64): load 128 fp8 from `q_fp8[t, h, :]`, dot
    in fp32 (cast both sides on the fly), multiply by per-token
    `q_scale × softmax_scale × head_scale` (folded into
    `w[t, h] = index_weights_out[t, h]`), relu, accumulate to
    `logits_partial[t, j]`.
  - The kv_scale per-token (one fp32 stored in the page's scale region)
    is applied as a final multiply on the relu'd partial.
- **Phase 2 (topk):** small block of threads runs a radix top-K or a
  streaming-heap top-K of size 512 over `logits_partial[t, :]`.
- **Mask:** prefill applies the causal mask inside Phase 1 (set
  partial=-inf for `j ≥ (q_idx+1)/R`).
- **Offset:** `topk_indices[t, k] += request_offsets[req]`; for prefill,
  also write `-1` for any topk slot whose value came from a -inf-masked
  position.

For v1 we can collapse Phase 1 + Phase 2 into a single task with a
streaming-heap-of-512 stored in shared memory (128 KiB SMEM budget per
CTA on B200 is ample: 512 × (4 B + 4 B) = 4 KiB heap). The heap-update
is a 9-stage compare-and-swap (lg2(512)=9), happening once per `j`. For
`num_compressed=2048` that's 18432 compares — small.

#### F.3.8 Test-mode unit test plan

- **File:** `tests/runtime_python/test_mode/test_indexer_score_topk_testmode.py`.
- **Shape.** T=8, H=64, head_dim=128, num_compressed=64,
  index_topk=32 (downscaled from 512 for fast test; the kernel is
  templated on `topk`).
- **Oracle.** PyTorch snippet of `Indexer.forward` lines 418-432.
- **Tolerance.** Topk indices are integers, so exact equality is the
  target, **but** for ties (same score) the topk order is
  implementation-defined. For the test we deduplicate the scores by
  adding a tiny `epsilon * j` perturbation, and check the **set** of
  returned indices matches.
- **Decode + prefill paths** both exercised.

#### F.3.9 Reuse from V3

- **Paged-KV-cache buffers** — yes, the indexer KV cache uses
  `paged_kv_*` buffers (see §K).
- **`mla_kv_gather_layer`** (`persistent_kernel.py:1401-1420`) — *not*
  reused (the indexer streams page reads inside the score kernel rather
  than gathering to a contiguous buffer; the data is 128 B/token vs
  584 B/token for MLA, so streaming is cheaper).
- **`argmax_partial_layer` + `argmax_reduce_layer`** — these implement
  top-1, not top-K. Not reusable directly but the partial/reduce pattern
  is a precedent for the v2 top-K split.

---

## §G. Cross-task wiring summary

Layers with `compress_ratio == 4` (e.g. layer 2 of V4-Flash) wire up like:

```
              ┌────────────────────────────────────────────────────────────────┐
              │  upstream: wq_a/q_norm/wq_b (Q lora-A + RMSNorm + lora-B),      │
              │            wkv (Compressor's),                                  │
              │            weights_proj (Indexer's per-head),                   │
              │  all built from V3 linear_fp8_layer / linear_layer.            │
              └─────────┬──────────────────┬──────────────────┬─────────────────┘
                        │qr (q-LoRA)        │kv_score (kv+score) │indexer_kv_score
                        ▼                  ▼                  ▼
                   wq_b (linear_fp8)   compressor_layer    compressor_layer
                                       (HEAD_DIM=512,      (HEAD_DIM=128,
                                        Attention's       Indexer's
                                        compressor)       compressor)
                                              │                  │
                                              ▼                  ▼
                                       Attention's         Indexer's
                                       compressed-KV       FP8 KV cache
                                       cache (paged)       (paged)
                        │                                       │
                        ▼                                       │
                   index_q (bf16)                                │
                        │                                       │
                        ▼                                       │
              indexer_q_transform_layer                          │
                  │            │                                │
                  ▼            ▼                                │
              index_q_fp8  index_weights_out                    │
                  └─────────┬──┴─────────────────────────────────┘
                            ▼
                  indexer_score_topk_layer
                            │
                            ▼
                       topk_indices  ────►  consumed by mla_v4_decode_layer
                                            (see attention.md)
```

Layers with `compress_ratio == 128`:

```
                   ...                                 ...
                   ▼                                   (no Indexer)
                                       compressor_layer
                                       (HEAD_DIM=512,
                                        Attention's
                                        compressor,
                                        overlap=False)
                                              │
                                              ▼
                                       Attention's
                                       compressed-KV
                                       cache (paged)
                                              │
                                              ▼
                                topk_indices is *static*, precomputed
                                at metadata-build (selects the newest
                                `min(index_topk, end_pos/128)` slots).
                                                │
                                                ▼
                                        consumed by mla_v4_decode_layer
```

Layers with `compress_ratio == 0` skip this entire spec.

---

## §H. Module-level test plan

After all three per-task tests pass (§F.x.10 each), a *module test*
exercises the full Compressor + Indexer pipeline for a single attention
layer with `compress_ratio == 4`:

- **File:** `tests/runtime_python/test_mode/test_sparse_attn_module_testmode.py`.
- **Setup.** Construct a V4-Flash layer-2 `Attention` module from the
  official `model.py` with `compress_ratios=[0,0,4]`. Build the
  corresponding MPK kernel sequence using the three layer methods above
  (no decode yet — just the sparse-attention prelude through
  `topk_indices` output).
- **Oracle.** Run the official module's prefill on a B=1, seqlen=16
  prompt; capture `topk_idxs` as the oracle. Then run the MPK kernel
  with the same weights.
- **Pass criterion.**
  - For tokens where `topk_indices < 0` in the oracle, MPK must also
    produce `< 0` at the same slots.
  - For valid topk slots, the *set* of indices returned must match the
    oracle (order may differ on ties).
- **Tolerance for upstream values.** Per
  `i-want-to-add-dapper-pascal.md:277-279`, bf16 paths use
  `rtol=1e-3, atol=1e-3`; FP8 paths use a looser bound after dequant.

---

## §I. Reuse summary

| MPK V3 facility | Reused here? | Notes |
|---|---|---|
| `paged_kv_indptr_buffer`, `paged_kv_indices_buffer`, `paged_kv_last_page_len_buffer` (`persistent_kernel.py:48-50, 563-571, 3205, 3312-3314`) | **Yes**, for all 3 caches: compressor state, Attention's compressed-KV, Indexer's FP8 KV. Per-page byte width differs (32 KiB state-cache pages vs 576-aligned compressed-KV pages vs 132-B/token indexer pages) but the index-table machinery is the same. | A per-cache `head_size` parameter at allocation time selects the byte width. |
| `mla_kv_gather_layer` (`persistent_kernel.py:1401-1420`) | **Indirectly** — same paged-iteration pattern, but the indexer score task streams reads instead of gathering. | Cite as the "how to iterate paged KV" precedent for the implementer. |
| `quantize_fp8_layer` with `scale_ue8m0=True` (`persistent_kernel.py:1967-1988`) | **Inlined**, not called. The Compressor fuses RMSNorm + RoPE + quant; Indexer Q fuses RoPE + quant + weight-fold. | Cite as the "how to do UE8M0 per-block quant" precedent. |
| `per_token_group_quantize_fp8.cuh` (`tasks/blackwell/per_token_group_quantize_fp8.cuh:39-130`) | **Algorithmically reused**. `encode_ue8m0`, the per-warp reduce, and the cast are copied verbatim into both new `.cuh` files. | Helper `encode_ue8m0` at line 33 is the only fully callable piece — extract into a shared header. |
| `rmsnorm_layer` (`persistent_kernel.py`) | **Not reused** — RMSNorm is fused into the Compressor task to avoid an HBM round-trip. | Algorithm copied. |
| `linear_fp8_layer` (`persistent_kernel.py:1990-2015`) | **Reused** for the upstream `wq_b`, `wkv`, `wgate`, `weights_proj` GEMMs. Not part of this spec's tasks. | |

---

## §J. Open questions

`**OPEN:**` Confirm with reviewers that v1 targets FP8 indexer KV cache
(matching `i-want-to-add-dapper-pascal.md:432-433`); if vLLM's
`use_fp4_indexer_cache=True` default
(`deepseek_v4_attention.py:1059`) must be matched, the MXFP4 task
variants get promoted from v2 to v1.

`**OPEN:**` In §F.1.2 Option A: verify MPK's `kn_graph.register_task`
chain encodes producer→consumer ordering across two task names so we
don't need a device-wide barrier inside a single task. Look at
`mla_kv_gather_layer → mla_decode_layer` for precedent.

`**OPEN:**` In §E: choose between (a) host-side init via
`cudaMemcpyToSymbol` and (b) literal-embedded constant table in the
codegen, for the Hadamard 128×128 bf16 constant.

`**OPEN:**` In §F.2.3: confirm whether V4-Flash's Indexer Q RoPE uses
the base `rope_theta = 10000` or the `compress_rope_theta = 160000`.
Trace: `model.py:481` builds Attention's `freqs_cis` from
`compress_rope_theta` when `compress_ratio > 0`; `model.py:410` assigns
`self.compressor.freqs_cis = self.freqs_cis`, but the *Indexer's own*
`self.freqs_cis` at `model.py:400` is set by `Block.__init__` in code
not shown — needs to be confirmed by reading the surrounding `Block`
construction.

`**OPEN:**` In §F.2.3: confirm with reviewers that the Compressor's
`head_dim=128` Hadamard cannot be absorbed into the upstream `wkv` /
`wgate` weight the way the Indexer-Q Hadamard can be absorbed into
`wq_b` (the per-token softmax over the gated pool prevents pre-multiplying
into `wkv`). If true, the indexer's Compressor must apply Hadamard at
runtime — and v1's choice is the naive 128×128 matmul against a
`__constant__` table.

`**OPEN:**` In §F.3.1: confirm `weights_proj` output can be negative; if
it can, the relu-then-weight ordering matters and we must follow the
official's order to match numerics. Inspect a checkpoint's
`weights_proj.weight` distribution.

`**OPEN:**` In §F.3.3: pin down the source path for DeepGEMM's
`fp8_fp4_paged_mqa_logits` CUDA implementation under
`deps/vllm/vllm/third_party/deep_gemm/include/deep_gemm/impls/`. We
intentionally do **not** port this kernel for v1 — we implement a naive
equivalent — but the spec should cite the source for the v2 optimization.

`**OPEN:**` In §F.1.4: the `kv_cache_block_size=64` for the head=512
compressed-KV cache is the *V3* value; verify V4-Flash uses the same
page size for the compressed-KV cache or if it should adopt the
`SlidingWindowMLASpec(block_size=4, ...)` 4-token pages from
`deepseek_compressor.py:165-169` (the state cache's page size).

`**OPEN:**` In §F.1.10: the corner test "head_dim=128, compress_ratio=128"
exercises the `coff=1` code path with the indexer head width — confirm
it's worth the implementation effort (it doesn't exist in the real
config but catches off-by-one bugs in the `head_offset` term).
