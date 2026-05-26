# MPK Spec: DeepSeek V4-Flash MTP Block

**Wave 1, spec-only.** Sources of truth:
- Official:  `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`, class `MTPBlock`, lines **739–767**.
- vLLM:  `deps/vllm/vllm/model_executor/models/deepseek_v4.py` lines **1296–1337**, **1565–1569**.
- vLLM MTP draft model: `deps/vllm/vllm/model_executor/models/deepseek_v4_mtp.py` lines **61–150**, **217–240**.
- V3 MTP (MPK): `python/mirage/mpk/models/deepseek_v3/builder.py:1640–1900`, `python/mirage/mpk/persistent_kernel.py:1571–1633` (`mla_mtp_decode_layer`).

This document is *additive* to the rest of `docs/mpk/deepseek_v4/`. Everything
"inside" the MTP block (mHC, attention with `compress_ratio=0`, MoE with
`sqrtsoftplus` routing, mhc_head, lm_head, argmax) is owned by other Wave‑1
specs (`hc.md`, `attention.md`, `sparse.md`, `moe.md`).

## 0. Verified key facts

| Fact | Value | Citation |
|---|---|---|
| `num_nextn_predict_layers` | 1 | plan key facts + checkpoint `config.json` |
| MTP layer global index | **43** | `n_layers=43`, `args.n_layers + 0` in `model.py:792` |
| `compress_ratios[43]` | **0** | plan key facts ("last entry") |
| `tie_word_embeddings` | **false** | plan key facts; `lm_head` and `embed` have separate weights |
| MTP shares `embed`? | **Yes** | `model.py:793`: `self.mtp[-1].embed = self.embed` |
| MTP shares `head` (lm_head module)? | **Yes** | `model.py:794`: `self.mtp[-1].head = self.head` |
| MTP shares `hc_head_fn` / `hc_head_base` / `hc_head_scale`? | **NO** | MTPBlock owns its own copies, `model.py:751–753` |
| MTP shares the final `self.norm`? | **NO** | MTPBlock has its own `self.norm = RMSNorm(...)`, `model.py:747` |
| Inner Block weights (attn/ffn/attn_norm/ffn_norm/hc_attn_*/hc_ffn_*) | **MTP-owned** | `MTPBlock(Block)` calls `super().__init__(layer_id, args)` so the inherited Block has its own fresh parameters; checkpoint stores them under the MTP-layer prefix (`model.layers.43.*` in HF naming) |

## 1. Scope

ONE new task: **`mtp_embed_hidden_fuse_layer`**.

Everything else (`mhc_prenorm_gemm`, `mhc_pre`, `mla_v4_q_kv_rmsnorm`,
`mla_v4_decode`/`mla_v4_prefill`, `inv_rope_fp8_quant_o`, `mhc_post`,
MoE pipeline with `sqrtsoftplus` routing, `mhc_head`, `lm_head`, argmax,
`embed_layer`, `rmsnorm_layer`) is **reuse** from other V4 specs and V3 MPK.

## 2. The math (verified from `model.py`)

`MTPBlock.forward`, `model.py:758–767`:

```
e         = embed(input_ids)                        # [T, D]      bf16  (line 761)
e_norm    = enorm(e)                                # [T, D]      bf16  (line 762, RMSNorm dim=D)
h_norm    = hnorm(x)                                # [T, hc, D]  bf16  (line 763, RMSNorm dim=D, broadcast across hc)
x_fused   = e_proj(e_norm).unsqueeze(2)             # [T, 1, D]   bf16
          + h_proj(h_norm)                          # [T, hc, D]  bf16
                                                     #             (line 764)
x_block   = Block.forward(x_fused, start_pos, input_ids)   # [T, hc, D]   (line 765 / model.py:689)
logits    = ParallelHead.forward(x_block,
              hc_head_fn, hc_head_scale, hc_head_base,
              self.norm)                            # [T, vocab_size]   (line 766)
```

Block.forward (`model.py:689–701`) is *exactly* the same code path used for
every other V4 layer:  `hc_pre → attn_norm → attn → hc_post → hc_pre → ffn_norm → ffn → hc_post`.
With `compress_ratios[43] = 0` the attention path takes the no-compressor,
no-indexer branch (SWA-only decode/prefill).

ParallelHead.forward (`model.py:719–727`, `hc_head` body at 729–736):
```
y      = hc_head(x_block, hc_head_fn, hc_head_scale, hc_head_base)  # [T, D]   (the MTP-owned hc_head reduces hc→1)
y_norm = self.norm(y)                                               # [T, D]   (the MTP-owned final norm)
logits = lm_head_linear(y_norm.float())                             # [T, V]   (uses shared lm_head)
```

The `e_proj` / `h_proj` order in `model.py:764` is `e_proj(e_norm).unsqueeze(2) + h_proj(h_norm)`; vLLM line **141** writes the same expression as `h_proj(prev_hidden) + e_proj(inputs_embeds).unsqueeze(-2)` — addition is commutative; semantics match.

## 3. The new task: `mtp_embed_hidden_fuse_layer`

### 3.a Strategy summary

Two implementation strategies are documented. **v1 ships the decomposed
strategy** — zero new CUDA — because the math fits cleanly inside existing
`linear_layer` + `elementwise_add_layer` primitives, and the MTP block runs
once per token (it is not on the per-layer hot path). The fused single-CUDA
variant is recorded for v2 as an optimization opportunity.

### 3.b Math

Inputs:
- `e_norm: [T, D] bf16`   — output of `enorm(embed(input_ids))`
- `h_norm: [T, hc, D] bf16` — output of `hnorm(x)`, where `x` is the HC hidden from prev layer (layer 42)

Weights:
- `e_proj_W: [D, D] bf16` (or fp8 in vLLM; see §3.f) — `model.py:743`
- `h_proj_W: [D, D] bf16` (or fp8) — `model.py:744`

Output:
- `fused: [T, hc, D] bf16` where  `fused[t, j, d] = sum_k e_proj_W[d, k] · e_norm[t, k] + sum_k h_proj_W[d, k] · h_norm[t, j, k]`

(Broadcast of `e_proj_e` across the `hc` axis, as written in `model.py:764`.)

### 3.c Source to migrate

| Reference | Path | Lines |
|---|---|---|
| Official math | `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py` | **764** (the one-liner that defines the fuse), plus `model.py:743–744` for weight shapes |
| vLLM production shape | `deps/vllm/vllm/model_executor/models/deepseek_v4_mtp.py` | **78–94** (separate `ReplicatedLinear` for `e_proj` / `h_proj`, fp8 capable) and **141–143** (the call) |
| vLLM "no fused kernel" confirmation | `deps/vllm/vllm/model_executor/models/deepseek_v4_mtp.py` | comment at **79–80**: *"V4 keeps e\_ and h\_ proj separate (with fp8 linear quant) rather than fusing them the way V3 does with eh\_proj."*  This is the key signal: vLLM ships **no fused MTP-head kernel** for V4; it relies on the per-projection `ReplicatedLinear`s. Therefore there is no upstream CUDA / TileLang reference to port — the fused-CUDA strategy below would be net-new code. |

**Conclusion:** for v1 we follow vLLM's structure (two independent linears +
add). The "fused" alternative has no precedent in V3 MPK, V4 PyTorch, or V4
vLLM; it is a pure optimization for v2.

### 3.d Inputs / Outputs / Weights tables

**Inputs (v1 decomposed):**

| Name | Shape | dtype | Producer |
|---|---|---|---|
| `e_norm_2d` | `[T, D]` | bf16 | `rmsnorm_layer(e, enorm.weight)` where `e = embed_layer(input_ids)` |
| `h_norm_3d` | `[T, hc, D]` | bf16 | `rmsnorm_layer` applied per-D on the HC tensor (see §6 reshape note) |

**Weights:**

| Name | Shape | dtype | Source ckpt key |
|---|---|---|---|
| `e_proj_W` | `[D, D]` | bf16 | `model.layers.43.e_proj.weight` |
| `h_proj_W` | `[D, D]` | bf16 | `model.layers.43.h_proj.weight` |
| `enorm_W`  | `[D]`    | bf16 | `model.layers.43.enorm.weight` |
| `hnorm_W`  | `[D]`    | bf16 | `model.layers.43.hnorm.weight` |

**Output:**

| Name | Shape | dtype |
|---|---|---|
| `fused` | `[T, hc, D]` | bf16 |

For the v1 decomposed plan, MPK allocates `fused` as `[T, hc * D]` flat and
the inner Block reshapes (HC layers in MPK already consume the flat `[T, hc*D]`
layout — see `hc.md` for the `mhc_prenorm_gemm` input contract).

### 3.e Grid design (mandatory per plan §75–104)

**Decomposed strategy (v1):** No new grid. Reuses the existing
`linear_layer` grid contract:

- `e_proj`: grid_dim = `(D // 64, 1, 1)`, block_dim = `(128, 1, 1)`, matching
  `persistent_kernel.py:2434–2463` (`linear_layer`).
- `h_proj` (per hc copy): grid_dim = `(D // 64, 1, 1)`. Each of the `hc=4`
  copies is a separate `linear_with_residual_layer` call so the second-and-later
  ones accumulate into `fused_flat`.
- `elementwise_add` is folded into the residual path of
  `linear_with_residual_layer`, so no separate add task is dispatched on the
  hot path.

(The "decomposition" never invokes `elementwise_add_layer` directly because
`linear_with_residual_layer` already does `out = W @ x + residual` in one
task. We mention `elementwise_add_layer` only as the conceptual building
block, per the user's request.)

**Fused strategy (future v2, not implemented in v1):**

- Task file: `include/mirage/persistent_kernel/tasks/blackwell/mtp_embed_hidden_fuse_sm100.cuh`.
- Natural grid: `(T, D_TILE, hc)` where `D_TILE = D / TN` with `TN = 64`
  (matches V4 dense-linear tile size). Total CTAs = `T * (D/64) * hc`.
- Each CTA computes `fused[t_slot, j, d_tile : d_tile+64]` by accumulating
  `e_proj_W[d_tile:d_tile+64, :] @ e_norm[t_slot, :]` once and
  `h_proj_W[d_tile:d_tile+64, :] @ h_norm[t_slot, j, :]` per j.
- `task_desc->task_metadata[0]` = `t_slot`; `[1]` = `d_tile_idx`; `[2]` = `j`.
- `input_ptrs[0..3]` = `{e_norm, h_norm, e_proj_W, h_proj_W}`; `output_ptrs[0]` = `fused`.
- Alignment: `grid_dim.y` (= `D/64`) must satisfy `D % 64 == 0`. For
  Flash-Base, `D = 4096`, `D/64 = 64`, ok.
- `blockIdx`-agnostic: no `blockIdx.x/y/z` used; CTA picks its slice from
  metadata. (Plan §107.)

### 3.f `/add-mpk-task` conformance checklist

**v1 (decomposed):** **N/A — no new task is registered.** No `runtime_header.h`
TaskType entry, no `task_register.cc` entry, no `graph.cc` dispatch entry,
no new `.cuh`. Builder calls the existing `linear_layer` and
`linear_with_residual_layer` Python methods (already wired).

**v2 (fused, not in v1):**

- [ ] `runtime_header.h`: add `TASK_MTP_EMBED_HIDDEN_FUSE`.
- [ ] `task_register.cc`: add `register_mtp_embed_hidden_fuse_task` following the
      `register_linear_*` patterns (path: ` deps cited as planpath §117–124`).
- [ ] `graph.cc`: add task-name → registration-function dispatch for
      `"mtp_embed_hidden_fuse_sm100"`.
- [ ] `persistent_kernel.py`: add `mtp_embed_hidden_fuse_layer(...)` method
      mirroring `linear_layer`'s shape assertion style — except input shapes
      are `(T, D)` for `e_norm`, `(T, hc, D)` for `h_norm`, output `(T, hc, D)`.
- [ ] Implementation is `blockIdx`-agnostic.
- [ ] Test-mode unit test under `tests/runtime_python/test_mode/`.
- [ ] Rebuild via `pip install -e . -v --no-deps`.

### 3.g Two implementation strategies

#### Strategy A — Decomposed (recommended for v1)

In the builder, the `mtp_embed_hidden_fuse` step is implemented as:

```python
# All tensors flat over the HC axis: shape  [T, hc * D]  bf16.
# `linear_layer`/`linear_with_residual_layer` are 2D, so we treat the
# (hc, D) tail as a single contiguous (hc*D,) row.

# Step 1: e_proj broadcast to all hc copies.
# We write the same e_proj(e_norm)  into  fused[t, j=0..hc-1, :].
# Implementation option A1: tile e_proj_W into [hc*D, D] by row-replicating
# it `hc` times at weight-load time, then a single linear_layer fills the
# whole [T, hc*D] output.
e_proj_W_replicated = e_proj_W.unsqueeze(0).expand(hc, -1, -1).reshape(hc * D, D).contiguous()
pk.linear_layer(
    input  = e_norm_2d,                              # [T, D]
    weight = w_e_proj_replicated,                    # [hc*D, D]
    output = fused_flat,                             # [T, hc*D]
    grid_dim = (grid_for_rmsnorm_linear_layer(hc * D), 1, 1),
    block_dim = (128, 1, 1),
)

# Step 2: h_proj per-hc-copy, accumulating into fused_flat.
# For each j in 0..hc-1, slice h_norm at [T, j*D : (j+1)*D] and write
# h_proj(h_norm[:, j, :]) into the corresponding fused[T, j*D:(j+1)*D] slot.
# Since `linear_with_residual_layer` does  out = W @ x + residual  in one task,
# we use it with residual = fused_flat[:, j*D:(j+1)*D] = e_proj_e.
# Each j is a separate kernel registration that views a sub-DTensor.
for j in range(hc):
    pk.linear_with_residual_layer(
        input    = h_norm_view_2d[j],                # [T, D]   view at hc=j
        weight   = w_h_proj,                         # [D, D]
        residual = fused_view_2d[j],                 # [T, D]   in-place write
        output   = fused_view_2d[j],                 # [T, D]
        grid_dim = (D // 64, 1, 1),
        block_dim = (128, 1, 1),
    )
```

Notes:
1. The `e_proj_W_replicated` precomputation lives in
   `demo/deepseek_v4/models/convert.py` (weight conversion script for v1) so
   the runtime tensor pointer is stable. The replication costs `hc * D * D *
   2 = 4 * 4096 * 4096 * 2 = 128 MiB` of extra weight memory **only on the
   MTP layer**; acceptable for v1.
2. An alternative (A2) avoids weight replication by calling `linear_layer`
   once into a `[T, D]` temp buffer and then `hc` `elementwise_add_layer`s
   that view `fused_flat[:, j*D:(j+1)*D]`. Since `elementwise_add_layer` is
   strictly 2D (verified at `persistent_kernel.py:2566–2584`), each per-j view
   is `[T, D]` — fully compatible.  A2 trades replicated weights for an
   extra `T*D` write of `e_proj_e` and an extra task per hc copy. We pick
   **A1** for v1 (fewer task dispatches; weight memory is cheap on the MTP
   layer).

#### Strategy B — Fused single CUDA task (future v2)

One `.cuh` kernel does both linears + the add in one pass, exploiting that
`e_proj_e` is broadcast across `hc`. The shape of the work matches a
batched-by-hc matmul (`e_proj_W` reused `hc` times, `h_proj_W` applied per
copy). Expected memory traffic reduction: ~30 % (avoid writing the `[T, hc,
D]` intermediate from Strategy A1 step 1).

See §3.e for the grid design. Implementation is deferred to v2 — the v1 spec
explicitly records this design so the v2 PR can land without re-litigating
boundaries.

### 3.h Test-mode unit test plan (v1 decomposed)

The decomposed path has no new task, so the per-task test under
`tests/runtime_python/test_mode/test_mtp_embed_hidden_fuse_testmode.py`
becomes a **micro-module test**:

1. Allocate `T = 8`, `D = 4096`, `hc = 4`.
2. Build a `PersistentKernel` in test mode (see `/test-mode` skill, canonical
   pattern `tests/runtime_python/test_mode/test_rmsnorm_testmode.py`).
3. Call the decomposed pipeline (one `linear_layer` + `hc` `linear_with_residual_layer`).
4. PyTorch oracle: `e_proj(e_norm).unsqueeze(2) + h_proj(h_norm)` from
   `model.py:764`. Compare with `torch.allclose(rtol=1e-3, atol=1e-3)`.

For v2 fused, the test file becomes a standard per-task test driving the new
`mtp_embed_hidden_fuse_layer` Python method directly.

### 3.i Reuse from V3 (verified)

| Component | V3 reference |
|---|---|
| `linear_layer` | `persistent_kernel.py:2434–2463` |
| `linear_with_residual_layer` | `persistent_kernel.py:2465–2500` |
| `elementwise_add_layer` (fallback A2) | `persistent_kernel.py:2566–2584` |

V3 itself uses a *different* fuse — `eh_proj` is a single `[D, 2D]` linear
that operates on `concat([e_norm, h_norm])` (`builder.py:1685–1695`). V4 does
**not** fuse the weights this way (`deepseek_v4_mtp.py:79–80` comment), so we
cannot reuse V3's `eh_proj` decomposition verbatim.  However the layer-method
calls are identical — we just call `linear` twice with different inputs.

## 4. MTP block forward walkthrough (pseudo-code in MPK task names)

This shows the *entire* MTP block, not just the fuse. All names below refer
to existing or already-spec'd MPK layer methods; only one (the fuse step)
is decomposed inline.

Configuration:  `T = max_num_batched_tokens` (e.g. 1), `D = 4096`, `hc = 4`,
`V = 129280`, `layer_idx = 43`, `compress_ratio = 0`.

```python
# ─── MTP block: layer 43, compress_ratio = 0 ──────────────────────────────
# Inputs from prev layer (42):
#   prev_x_hc : [T, hc, D] bf16   (HC hidden state)
#   input_ids : [T, 1]    int64

# 1. Embed input_ids
pk.embed_layer(input=mtp_input_tokens, weight=w_embed, output=e_2d,
               grid_dim=(1,1,1), block_dim=(128,1,1), input_source=1)
# e_2d: [T, D] bf16

# 2. enorm
pk.rmsnorm_layer(input=e_2d, weight=w_enorm, output=e_norm_2d,
                 grid_dim=(T,1,1), block_dim=(128,1,1))

# 3. hnorm (RMSNorm over the inner D dim, broadcast across hc — same as V3 builder.py:1795–1798)
#    Since rmsnorm_layer is 2D, view prev_x_hc as [T*hc, D] (it is contiguous in (T, hc, D)).
pk.rmsnorm_layer(input=prev_x_hc_view_2d, weight=w_hnorm, output=h_norm_view_2d,
                 grid_dim=(T*hc,1,1), block_dim=(128,1,1))
# h_norm_view_2d: [T*hc, D]; reshaped logically to [T, hc, D]

# 4. mtp_embed_hidden_fuse  (Strategy A1, decomposed)
pk.linear_layer(input=e_norm_2d, weight=w_e_proj_replicated, output=fused_flat,
                grid_dim=(grid_for_rmsnorm_linear_layer(hc*D),1,1), block_dim=(128,1,1))
for j in range(hc):
    pk.linear_with_residual_layer(
        input=h_norm_view_2d[j], weight=w_h_proj,
        residual=fused_flat_view_2d[j], output=fused_flat_view_2d[j],
        grid_dim=(D//64,1,1), block_dim=(128,1,1),
    )
# fused_flat: [T, hc*D] bf16  ≡ "x" entering the inner Block

# ─── Inner Block.forward (model.py:689–701), now operating on fused_flat ──

# 5a. HC pre-mix for attention   (hc.md tasks)
pk.mhc_prenorm_gemm_layer(...)                      # produces gemm_out_mul, gemm_out_sqrsum
pk.mhc_pre_layer(...)                               # produces attn_post_mix, attn_comb_mix, attn_layer_input [T, D]

# 5b. attn_norm
pk.rmsnorm_layer(input=attn_layer_input, weight=w_mtp_attn_norm,
                 output=attn_norm_out,
                 grid_dim=(T,1,1), block_dim=(128,1,1))

# 5c. MLA attention (attention.md), compress_ratio = 0 → SWA-only branch
pk.mla_v4_q_kv_rmsnorm_layer(...)                   # joint Q+KV RMSNorm
# Decide decode vs prefill on Q_LEN (matches V3 logic in builder.py:237–249):
pk.mla_v4_decode_layer(...)  # or mla_v4_prefill_layer + mla_v4_prefill_gather
# compress_ratio = 0 means no compressor / no indexer inputs (per attention.md §compress_ratio=0 branch)
pk.inv_rope_fp8_quant_o_layer(...)                  # quantize attention out for o-proj

# 5d. o-projection (FP8) and HC post-mix for attention
pk.linear_fp8_layer(input=attn_o_fp8, weight=w_mtp_wo,  # `mhc_post` expects bf16 attn out
                    output=attn_proj_out, ...)
pk.mhc_post_layer(input=attn_proj_out, residual=fused_flat,
                  post_mix=attn_post_mix, comb_mix=attn_comb_mix,
                  output=after_attn_hc, ...)        # [T, hc*D]

# 5e. HC pre-mix for ffn
pk.mhc_prenorm_gemm_layer(...)                      # uses hc_ffn_fn (MTP-owned)
pk.mhc_pre_layer(...)                               # ffn_post_mix, ffn_comb_mix, ffn_layer_input [T, D]

# 5f. ffn_norm
pk.rmsnorm_layer(input=ffn_layer_input, weight=w_mtp_ffn_norm,
                 output=ffn_norm_out, grid_dim=(T,1,1), block_dim=(128,1,1))

# 5g. MoE pipeline (moe.md).  layer_idx=43, NOT in [0..num_hash_layers-1]=[0..2],
#     so routing is sqrtsoftplus + group-topk (NOT hash_route_lookup).
#     This is the critical point the user flagged: MTP is layer 43, hash gate
#     only applies to layers <3.
pk.sqrtsoftplus_topk_layer(...)                     # scoring_func='sqrtsoftplus'
pk.moe_w13_fp8_layer(...) ; pk.swiglu_clamped_layer(...) ; pk.moe_w2_fp8_layer(...)
pk.moe_mul_sum_add_layer(...)
# (shared expert + allreduce omitted for single-GPU v1)

# 5h. HC post-mix for ffn
pk.mhc_post_layer(input=moe_out, residual=after_attn_hc,
                  post_mix=ffn_post_mix, comb_mix=ffn_comb_mix,
                  output=mtp_block_out_hc, ...)     # [T, hc*D]

# ─── ParallelHead (model.py:719–736) — MTP-specific weights ───────────────

# 6. hc_head  (mhc_head_layer with MTP's own hc_head_fn / scale / base)
pk.mhc_head_layer(input=mtp_block_out_hc,
                  hc_fn=w_mtp_hc_head_fn, hc_scale=w_mtp_hc_head_scale,
                  hc_base=w_mtp_hc_head_base, output=mtp_head_out_2d, ...)
# mtp_head_out_2d: [T, D]

# 7. MTP-owned final norm  (NOT the base model's `model.norm`)
pk.rmsnorm_layer(input=mtp_head_out_2d, weight=w_mtp_norm,
                 output=mtp_norm_out, grid_dim=(T,1,1), block_dim=(128,1,1))

# 8. Shared lm_head  (same weight as base model — but tie_word_embeddings=false
#    means lm_head ≠ embed, see §0)
pk.linear_layer(input=mtp_norm_out, weight=w_lm_head, output=mtp_logits,
                grid_dim=(grid_for_rmsnorm_linear_layer(V),1,1), block_dim=(128,1,1))

# 9. argmax → draft token  (only if running as a spec-decode draft; otherwise
#    the MTP head's logits are the model's primary output — see §5)
pk.argmax_partial_layer(input=mtp_logits, output=(argmax_part_value, argmax_part_index), ...)
pk.argmax_reduce_layer(input=(argmax_part_value, argmax_part_index), output=mtp_draft_tok, ...)
```

**Key task counts in this MTP block:**
- New tasks introduced by MTP: **0** (decomposed path).
- Reused tasks: every layer method above — `embed_layer`, `rmsnorm_layer`,
  `linear_layer`, `linear_with_residual_layer`, `linear_fp8_layer`,
  `mhc_prenorm_gemm_layer`, `mhc_pre_layer`, `mhc_post_layer`,
  `mhc_head_layer`, `mla_v4_q_kv_rmsnorm_layer`,
  `mla_v4_decode_layer`/`mla_v4_prefill_layer`,
  `inv_rope_fp8_quant_o_layer`, `sqrtsoftplus_topk_layer`,
  `moe_w13_fp8_layer`, `swiglu_clamped_layer`, `moe_w2_fp8_layer`,
  `moe_mul_sum_add_layer`, `argmax_partial_layer`, `argmax_reduce_layer`.

## 5. Comparison with V3 MTP — does V4 reuse `mla_mtp_decode_layer`?

**Decision: NO. V4 MTP does NOT use V3's `mla_mtp_decode_layer`.**

Rationale:

1. **`mla_mtp_decode_layer` is a V3 *attention* kernel.** Defined at
   `persistent_kernel.py:1571–1599`, registered as `mla_mtp_decode_sm100`.
   It is purpose-built for V3 MLA attention with a separate cache, a fixed
   `q_len = 1`, and a fused softmax+output reduction in one kernel
   (`mla_mtp_reduce_sm100`).  It encodes V3's compressed-KV cache shape
   (`qk_head_dim = 576`, `kv_lora_rank = 512`, `qk_rope_head_dim = 64`).

2. **V4 attention is a completely different kernel family.** Per `attention.md`
   (sibling Wave-1 spec), V4 introduces `mla_v4_decode_layer`,
   `mla_v4_prefill_layer`, `mla_v4_prefill_gather_layer`,
   `mla_v4_q_kv_rmsnorm_layer`, and `inv_rope_fp8_quant_o_layer`.  These
   kernels have different head dims (`head_dim=512`, `qk_rope_head_dim=64`,
   `qk_nope_head_dim=448`), a different cache (SWA + optional compressed),
   different scaling, and FP8 output quant for the o-projection.

3. **`compress_ratios[43] = 0`** *only* means MTP skips the Compressor /
   Indexer path; it does **not** change which attention kernel runs.  In V4,
   `mla_v4_decode_layer` is *always* the decode-path kernel; with `compress_ratio = 0`
   the optional `compressed_cache` and `topk_indices` inputs are simply
   absent (per attention.md §4.b decode contract).

4. **MTPBlock(Block) inherits Block.forward.** `model.py:739, 765`:
   `MTPBlock(Block)` calls `super().forward(x, start_pos, input_ids)` — so
   the attention call site is the **same code path** as any other layer.
   Since layer 43's `compress_ratio` is 0, that call site picks the SWA-only
   `mla_v4_decode`/`mla_v4_prefill` branch, identical to any other V4 layer
   whose `compress_ratio` is 0 (note: per plan key facts, only layers 0, 1,
   and 43 have ratio 0).

5. **Therefore the MPK builder treats MTP as a structurally normal V4 layer**
   with `compress_ratio = 0`, *plus* a preamble (`embed`, `enorm`, `hnorm`,
   `mtp_embed_hidden_fuse`) and a postamble (MTP-owned `hc_head` + `norm` +
   shared `lm_head` + argmax).  No V3 task is reused.

## 6. MTP head sharing — what is and is not shared with the base model

| Object | Shared with base? | Citation |
|---|---|---|
| `embed_tokens` weight (`w_embed`) | **Yes (shared)** | `model.py:793`: `self.mtp[-1].embed = self.embed` (the *module* is the same Python object → same weight tensor). vLLM `_mtp_hidden_buffer` machinery (`deepseek_v4.py:1296–1326`) similarly reuses the base embed. |
| `lm_head` weight (`w_lm_head`) | **Yes (shared)** | `model.py:794`: `self.mtp[-1].head = self.head`.  The `head` module is `ParallelHead`, which carries `self.weight` (the lm_head linear weight) — sharing the module shares the weight. **Critical:** `tie_word_embeddings = false`, so `lm_head.weight` is a *distinct* tensor from `embed.weight`, even though both are "shared between base and MTP". |
| `mhc_head_fn / base / scale` | **NO (MTP-owned)** | MTPBlock declares its **own** `hc_head_fn`, `hc_head_base`, `hc_head_scale` at `model.py:751–753`, alongside Transformer's at `model.py:798–800`. They are independent parameters. |
| Final `norm` before lm_head | **NO (MTP-owned)** | MTPBlock has `self.norm = RMSNorm(args.dim, args.norm_eps)` at `model.py:747`. The Transformer's `self.norm` at `model.py:788` is separate. `ParallelHead.forward` (`model.py:719–727`) takes `norm` as an argument — `MTPBlock.forward` passes `self.norm` (MTP's), `Transformer.forward` passes the base's. |
| Inner Block (attn/ffn/hc_attn_*/hc_ffn_*/attn_norm/ffn_norm) | **NO (MTP-owned)** | `MTPBlock(Block).__init__` calls `super().__init__(layer_id, args)` so the inherited Block has fresh weights stored at checkpoint key `model.layers.43.*`. |

The vLLM line 1326 (`_mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))`)
is consistent: it stashes the **base model's pre-hc_head HC residual** as the
`previous_hidden_states` input that the MTP draft will consume via `hnorm` +
`h_proj`. This is the same data flow as MPK's `prev_x_hc` in §4 step 4.

## 7. Builder integration sketch

File: `python/mirage/mpk/models/deepseek_v4/builder.py` (created in Wave 3).

```python
def _build_mtp_layer(self, state_dict: dict):
    """Build the single MTP block (layer 43) for DeepSeek V4-Flash.

    Structurally a normal V4 layer with compress_ratio=0, plus an MTP
    preamble (embed/enorm/hnorm/e_proj/h_proj) and a MTP-owned head
    (hc_head + norm + shared lm_head + argmax).
    """
    mtp_idx        = self.num_layers          # 43
    mtp_prefix     = f"model.layers.{mtp_idx}."
    hc, D, T       = self.hc_mult, self.hidden_size, self.max_num_batched_tokens

    # ── MTP-specific weights ────────────────────────────────────────────
    w_enorm           = self._attach(state_dict[f"{mtp_prefix}enorm.weight"], "mtp_enorm")
    w_hnorm           = self._attach(state_dict[f"{mtp_prefix}hnorm.weight"], "mtp_hnorm")
    w_e_proj          = state_dict[f"{mtp_prefix}e_proj.weight"]              # [D, D]
    self._mtp_e_proj_replicated = w_e_proj.unsqueeze(0).expand(hc, -1, -1).reshape(hc*D, D).contiguous()
    w_e_proj_rep      = self._attach(self._mtp_e_proj_replicated, "mtp_e_proj_replicated")
    w_h_proj          = self._attach(state_dict[f"{mtp_prefix}h_proj.weight"], "mtp_h_proj")
    w_mtp_attn_norm   = self._attach(state_dict[f"{mtp_prefix}attn_norm.weight"], "mtp_attn_norm")
    w_mtp_ffn_norm    = self._attach(state_dict[f"{mtp_prefix}ffn_norm.weight"], "mtp_ffn_norm")
    w_mtp_hc_head_fn  = self._attach(state_dict[f"{mtp_prefix}hc_head_fn"],    "mtp_hc_head_fn")
    w_mtp_hc_head_base= self._attach(state_dict[f"{mtp_prefix}hc_head_base"],  "mtp_hc_head_base")
    w_mtp_hc_head_scale=self._attach(state_dict[f"{mtp_prefix}hc_head_scale"], "mtp_hc_head_scale")
    w_mtp_norm        = self._attach(state_dict[f"{mtp_prefix}norm.weight"],   "mtp_norm")

    # ── HC mix weights for the inner Block (MTP-owned, same shapes as base layers) ──
    # hc_attn_fn / hc_ffn_fn etc. live at `model.layers.43.hc_attn_fn`, ...

    # ── Intermediate tensors ────────────────────────────────────────────
    e_2d              = self.mpk.new_tensor((T, D),     bfloat16, "mtp_e")
    e_norm_2d         = self.mpk.new_tensor((T, D),     bfloat16, "mtp_e_norm")
    h_norm_view       = self.mpk.new_tensor((T*hc, D),  bfloat16, "mtp_h_norm")     # 2D view of [T,hc,D]
    fused_flat        = self.mpk.new_tensor((T, hc*D),  bfloat16, "mtp_fused_flat") # = inner Block input

    # ── 1. embed + enorm ────────────────────────────────────────────────
    self.mpk.embed_layer(input=self.mtp_input_tokens, weight=self.w_embed,
                         output=e_2d, ...)
    self.mpk.rmsnorm_layer(input=e_2d, weight=w_enorm, output=e_norm_2d, ...)

    # ── 2. hnorm on prev_x_hc viewed as [T*hc, D] ──────────────────────
    prev_x_hc_view_2d = self._view_2d(self.prev_x_hc, T*hc, D)
    self.mpk.rmsnorm_layer(input=prev_x_hc_view_2d, weight=w_hnorm,
                           output=h_norm_view, ...)

    # ── 3. fuse: e_proj broadcast + h_proj per hc ──────────────────────
    self.mpk.linear_layer(input=e_norm_2d, weight=w_e_proj_rep,
                          output=fused_flat, ...)
    for j in range(hc):
        self.mpk.linear_with_residual_layer(
            input=self._slice_2d(h_norm_view, j*T, (j+1)*T),   # [T, D]
            weight=w_h_proj,
            residual=self._slice_2d(fused_flat, axis=1, lo=j*D, hi=(j+1)*D),
            output  =self._slice_2d(fused_flat, axis=1, lo=j*D, hi=(j+1)*D),
            ...)

    # ── 4. Inner Block (mhc_pre/attn/mhc_post/mhc_pre/ffn/mhc_post) ────
    self.x_hc = fused_flat
    self._build_v4_decoder_layer(
        state_dict, prefix=mtp_prefix,
        layer_idx=mtp_idx,         # 43
        compress_ratio=0,          # SWA-only attention
        is_mtp_layer=True,         # disables `hash_route_lookup` (since 43>=num_hash_layers=3)
    )

    # ── 5. MTP head: mhc_head + final norm + lm_head + argmax ─────────
    self.mpk.mhc_head_layer(input=self.x_hc,
                            hc_fn=w_mtp_hc_head_fn,
                            hc_scale=w_mtp_hc_head_scale,
                            hc_base=w_mtp_hc_head_base,
                            output=mtp_head_2d, ...)
    self.mpk.rmsnorm_layer(input=mtp_head_2d, weight=w_mtp_norm,
                           output=mtp_norm_2d, ...)
    self.mpk.linear_layer(input=mtp_norm_2d, weight=self.w_lm_head,
                          output=mtp_logits, ...)
    self.mpk.argmax_partial_layer(...)
    self.mpk.argmax_reduce_layer(...)
```

**Where this is called.** Inside the builder's main `build_from_dict` loop,
after the 43 base-model layers have run and after the *base* `hc_head` + base
`norm` + base `lm_head` + base `argmax` have produced the next token, the
MTP block is built **only if `num_nextn_predict_layers > 0`** (true for
DeepSeek V4-Flash with value 1).  V4 has one MTP block, so the build is a
single `_build_mtp_layer(state_dict)` call (no draft-loop unrolling — that
loop is a spec-decode feature, optional and orthogonal to MTP-block
construction).

## 8. Module-level test plan

File: `tests/runtime_python/test_mode/test_mtp_v4_module_testmode.py`
(written in Wave 3; references this spec).

Test outline:

1. **Instantiate official `MTPBlock`** from `deps/deepseek_v4/.../inference/model.py`.
   - `args = ModelArgs(n_hash_layers=0)` (as in the file's `__main__` at
     line 817), `args.n_layers + 0 = layer_id` for MTPBlock.
   - Attach a `ParallelEmbedding` and `ParallelHead` (the shared `embed` and
     `head`) — the `model.py:826–828` block already constructs this pattern.
2. **Wire weights into MPK** through the convert script's MTP path.
3. **Random inputs**: `input_ids: [T=8]` ints in `[0, vocab_size)`, `x: [1, 8, hc, D]` bf16 randn.
4. **Run both**: PyTorch `mtp(h, 0, x)` (returns logits `[1, 8, V]`), MPK
   `pk()` with `params["test_mode"] = True`.
5. **Compare**: `torch.allclose(mpk_logits, pt_logits, rtol=1e-2, atol=1e-2)`.
   Looser tolerance than per-task tests because the chain is long (~20 kernels)
   and includes FP8 in the o-proj and MoE paths.
6. **Bisection**: if mismatch, dump intermediate `e_norm`, `h_norm`,
   `fused_flat`, `after_attn_hc`, `mtp_block_out_hc`, `mtp_head_2d`,
   `mtp_norm_2d` and binary-search by comparing each against the matching
   PyTorch tensor.

Per-task unit tests for all reused kernels (mhc_*, mla_v4_*, etc.) are owned
by their respective spec files.

## 9. Wiring summary

| What | Where | New / Reuse |
|---|---|---|
| `mtp_embed_hidden_fuse` math | `_build_mtp_layer` in `builder.py` (Wave 3) | **Decomposed; no new kernel** |
| enorm, hnorm | `rmsnorm_layer` (V3) | **Reuse** |
| e_proj, h_proj | `linear_layer` + `linear_with_residual_layer` (V3) | **Reuse** |
| Inner Block (mhc + attn + moe + mhc) | other V4 specs (`hc.md`, `attention.md`, `moe.md`) | **Reuse from V4 wave** |
| mhc_head with MTP-owned hc_head_* | `mhc_head_layer` (`hc.md`) — **new V4 task**, called with MTP's weights | **Reuse the V4 task** |
| Final norm | `rmsnorm_layer` with MTP's `norm.weight` | **Reuse** |
| lm_head | `linear_layer` with shared `w_lm_head` (tie\_word\_embeddings=false → distinct from `w_embed`) | **Reuse** |
| argmax | `argmax_partial_layer` + `argmax_reduce_layer` | **Reuse** |
| Weight conversion (`e_proj` replication for Strategy A1) | `demo/deepseek_v4/models/convert.py` | **New, MTP-specific** |
| TaskType / task_register / graph.cc | unchanged | **N/A in v1** |

## 10. Open questions

**OPEN:** Confirm the HuggingFace checkpoint key prefix for MTP weights.
`model.py` calls `MTPBlock(args.n_layers + 0, args)` so `layer_id = 43`,
but the *checkpoint* storage convention may be either `model.layers.43.*`
(V3-style, used by `builder.py:1664–1666`) or a separate `model.mtp.0.*`
namespace (vLLM remap: `deepseek_v4.py:1502` shows `"mtp.": "model.mtp."`,
which is the inference-side name, not necessarily the on-disk one). Resolve
by running `safetensors_keys $checkpoint_dir | grep -E '(mtp|43)'` against
`/raid/catalyst/models/DeepSeek-V4-Flash-Base` before Wave 3 conversion code
is written.

**OPEN:** Does the MTP layer actually use FP8 quantization for `e_proj` and
`h_proj`? vLLM's `deepseek_v4_mtp.py:81–94` creates them via `ReplicatedLinear`
with `quant_config=quant_config`, which would make them fp8 in the
Flash-Base checkpoint. The on-disk `e_proj.weight` may therefore be `fp8_e4m3`
with `e_proj.weight_scale_inv`. Strategy A1 must then use
`linear_fp8_layer` + `linear_fp8_with_residual_layer` (already in MPK V3,
see `persistent_kernel.py` `linear_fp8_layer`) instead of bf16 `linear_layer`.
This changes nothing structural in the spec; it does change which existing
layer method is called. Confirm by inspecting checkpoint dtype before Wave 3.

**OPEN:** Should Strategy A1's `e_proj` weight replication (`hc * D * D * 2 = 128 MiB`)
live in the *converter* (offline) or be skipped in favor of Strategy A2
(one bf16 add per hc copy)? Decision is a memory-vs-task-count trade and
should be revisited once the v1 builder is profilable. Default: A1 in v1.

**OPEN:** The MTP block's *attention norm* and *ffn norm* — does V4 use them
identically to the base layer Block (`model.py:659–660`)? Reading `model.py:739–746`,
`MTPBlock(Block).__init__` calls `super().__init__` *and* additionally
creates `enorm`, `hnorm`, and `norm`. The base Block's `attn_norm` and
`ffn_norm` are therefore present as inherited members — yes, MTP uses them.
But the on-disk key for the MTP-layer's `attn_norm.weight` and `ffn_norm.weight`
needs to be confirmed alongside the first open question.

**OPEN:** When `start_pos > 0` (decode), the MTP block is invoked on a single
token (`model.py:828`: `mtp(h[:, 0:1], 1, x[:, 0:1])`). In MPK this maps to
the `mla_v4_decode_layer` path with `q_len = 1`. Are there any MTP-specific
buffer-layout differences (separate paged-KV cache for the MTP block? vLLM
appears to share the cache, since the MTP draft model in
`deepseek_v4_mtp.py` calls into `DeepseekV4DecoderLayer` which is the same
class as the base). Default: **share the paged-KV cache** (single set of
`paged_kv_indptr_buffer` etc.) but use a *layer_id = 43* slot in MPK's
per-layer cache indexing — exactly like the base model's per-layer cache
indexing. Confirm during builder review.

**OPEN:** Does `_mtp_hidden_buffer` (vLLM `deepseek_v4.py:1296`) imply MPK
needs a buffer that *captures* the pre-hc_head HC residual of the base
model and *re-feeds* it into the MTP block?  In vLLM this is a separate
buffer because base and MTP run in two captured CUDA graphs. In MPK
everything runs in one persistent kernel, so the base model's HC tensor
**before** `mhc_head_layer` can be wired directly as `prev_x_hc`. No extra
buffer needed. Confirm by reading the base builder once `hc.md` lands.
