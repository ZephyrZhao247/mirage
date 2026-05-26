# DeepSeek-V4-Flash — MPK Port — Architectural Overview (Wave 1 spec)

Status: Wave-1 architectural spec. No code. Sibling spec files (`hc.md`,
`attention.md`, `sparse.md`, `moe.md`, `mtp.md`) deep-dive each module's
kernels; this file fixes the model-wide dimensions, the per-layer behavior
matrix, the v1 subset-layer plan, and the top-level forward in terms of MPK
task names from the approved plan.

Authoritative sources (all citations are `file:line`):

- Config:
  `/raid/catalyst/models/DeepSeek-V4-Flash-Base/config.json` (lines 1–67).
- Official reference PyTorch:
  `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py` (828 lines).
- vLLM production reference:
  `deps/vllm/vllm/model_executor/models/deepseek_v4.py` (1578 lines).
- Approved plan:
  `/home/zepengz/.claude/plans/i-want-to-add-dapper-pascal.md`.

The plan's **Context**, **Mandatory kernel-authoring requirements**, and
**High-level Strategy** sections are binding on every other spec in
`docs/mpk/deepseek_v4/`.

---

## 1. Model config

### 1.1 Header values

Every entry is taken from `config.json` (line cited) or computed directly
from the official `model.py` (line cited). The "source" column points to the
single line that defines the value.

| Field | Value | Source |
|---|---|---|
| `architectures` | `["DeepseekV4ForCausalLM"]` | `config.json:2-4` |
| `model_type` | `deepseek_v4` | `config.json:21` |
| `torch_dtype` (activation) | `bfloat16` | `config.json:61` |
| `vocab_size` | 129280 | `config.json:64` |
| `hidden_size` (a.k.a. `dim`) | 4096 | `config.json:15` |
| `num_hidden_layers` (base blocks) | **43** | `config.json:28` |
| `num_nextn_predict_layers` (MTP) | **1** | `config.json:31` |
| **total layers in `Transformer`** | **44** = 43 + 1 | `model.py:786-792` (loop over `n_layers` then over `n_mtp_layers`) |
| `num_attention_heads` (`n_heads`) | 64 | `config.json:26` |
| `head_dim` | 512 | `config.json:13` |
| `qk_rope_head_dim` (`rope_head_dim`) | 64 | `config.json:35` |
| `nope_head_dim` = `head_dim - qk_rope_head_dim` | **448** | derived (`model.py:449`) |
| `q_lora_rank` | 1024 | `config.json:34` |
| `o_lora_rank` | 1024 | `config.json:33` |
| `o_groups` | 8 | `config.json:32` |
| `num_key_value_heads` (MLA-style) | 1 | `config.json:30` |
| `sliding_window` (`window_size`) | 128 | `config.json:57` |
| `max_position_embeddings` | 1048576 | `config.json:20` |
| `rope_theta` (default) | 10000 | `config.json:54` |
| `compress_rope_theta` (Compressor's) | **160000** | `config.json:65` |
| `rope_scaling.type` | `yarn` | `config.json:47-53` |
| `rope_scaling.factor` | 16 | `config.json:50` |
| `rope_scaling.original_max_position_embeddings` | 65536 | `config.json:51` |
| `rope_scaling.beta_fast` | 32 | `config.json:48` |
| `rope_scaling.beta_slow` | 1 | `config.json:49` |
| `rms_norm_eps` | 1e-6 | `config.json:46` |
| `attention_bias` | false | `config.json:5` |
| `tie_word_embeddings` | false | `config.json:59` |

### 1.2 Hyper-Connections (HC)

| Field | Value | Source |
|---|---|---|
| `hc_mult` (number of HC copies per token) | **4** | `config.json:11` |
| `hc_sinkhorn_iters` | 20 | `config.json:12` |
| `hc_eps` | 1e-6 | `config.json:10` |
| `mix_hc` = `(2 + hc_mult) * hc_mult` | **24** | `model.py:664`, vLLM `deepseek_v4.py:1120` |
| `hc_dim` = `hc_mult * hidden_size` | **16384** | `model.py:665`, vLLM `deepseek_v4.py:1121` |
| `hc_post_alpha` (vLLM scaling on post mix) | 2.0 | vLLM `deepseek_v4.py:1119` |

The HC pre-norm GEMM weight is `hc_fn[mix_hc, hc_dim] = [24, 16384]` per
attn/ffn site per layer (`model.py:667-668`).

### 1.3 MoE

| Field | Value | Source |
|---|---|---|
| `n_routed_experts` | **256** | `config.json:23` |
| `n_shared_experts` | 1 | `config.json:24` |
| `num_experts_per_tok` (top-k) | **6** | `config.json:27` |
| `moe_intermediate_size` | **2048** | `config.json:22` |
| `expert_dtype` | `fp8` (E4M3) | `config.json:9` |
| `quantization_config.fmt` | `e4m3` | `config.json:38` |
| `quantization_config.scale_fmt` | `ue8m0` | `config.json:40` |
| `quantization_config.weight_block_size` | `[128, 128]` | `config.json:41-44` |
| `scoring_func` | **`sqrtsoftplus`** | `config.json:56` |
| `topk_method` | `noaux_tc` | `config.json:60` |
| `norm_topk_prob` | true | `config.json:25` |
| `routed_scaling_factor` (`route_scale`) | 1.5 | `config.json:55` |
| `swiglu_limit` (Expert clamp) | **10.0** | `config.json:58` |
| `num_hash_layers` (early-layer hash routing) | **3** | `config.json:29` |

`sqrtsoftplus` scoring: `scores = softplus(logits).sqrt()`
(`model.py:571`). The `noaux_tc` bias is added before top-k but **not** to
the weight used downstream (`model.py:574-580`). Routing weights are
renormalized when `score_func != "softmax"` and then multiplied by
`route_scale` (`model.py:581-583`).

Hash routing path (only `layer_id < num_hash_layers == 3`): expert IDs are
looked up directly from a `[vocab_size, num_experts_per_tok] int32` table
`tid2eid` keyed by `input_ids`. The score is still computed but its top-k is
*discarded* — only the gathered `scores[indices]` is kept as the routing
weight (`model.py:556-583`, vLLM `deepseek_v4.py:754-769, 853-875`).

### 1.4 Sparse attention (Compressor + Indexer)

| Field | Value | Source |
|---|---|---|
| `index_n_heads` | 64 | `config.json:17` |
| `index_head_dim` | 128 | `config.json:16` |
| `index_topk` | 512 | `config.json:18` |
| `softmax_scale` (indexer) | `head_dim**-0.5 = 1/sqrt(128)` | `model.py:395` |
| Indexer's compressor uses Hadamard + FP4 sim | true | `model.py:398, 414-416` |
| `compress_rope_theta` ≠ `rope_theta` | yes — Compressor's RoPE uses 160000 | `model.py:476-481`, vLLM `deepseek_v4.py:1011` |

### 1.5 Per-layer `compress_ratios`

Read directly from `config.json:66` — 44-entry list. The Attention forward
selects `args.compress_ratios[layer_id]` (`model.py:453`). Per vLLM
`deepseek_v4.py:951-954`, the MTP layer (`layer_id == num_hidden_layers`)
falls outside the array and is forced to `compress_ratio = 1` (i.e. dense
SWA only) in their implementation; in the official `model.py` the MTPBlock
inherits from `Block` (`model.py:739-742`) and the attention internally sees
`compress_ratios[43]`. The official array supplied in the Flash-Base config
is 44 entries so `compress_ratios[43]` exists and equals 0.

---

## 2. Per-layer behavior map

Full 44-row table. Columns:

- `layer_idx` — 0-based.
- `compress_ratio` — from `config.json:66`.
- `has_compressor` — `compress_ratio != 0` (`model.py:466-471`).
- `has_indexer` — `compress_ratio == 4` (`model.py:468-471`,
  vLLM `deepseek_v4.py:1031`). Note `compress_ratio == 128` produces a
  Compressor but **no** Indexer (the static `get_compress_topk_idxs` path is
  used instead, `model.py:512-513`).
- `is_hash_routing` — `layer_idx < num_hash_layers (==3)` (`model.py:556`).
- `is_mtp` — `layer_idx == num_hidden_layers (==43)` (`model.py:791-792`).

| layer_idx | compress_ratio | has_compressor | has_indexer | is_hash_routing | is_mtp |
|---|---|---|---|---|---|
| 0 | 0 | no | no | **yes** | no |
| 1 | 0 | no | no | **yes** | no |
| 2 | 4 | yes | **yes** | **yes** | no |
| 3 | 128 | yes | no | no | no |
| 4 | 4 | yes | **yes** | no | no |
| 5 | 128 | yes | no | no | no |
| 6 | 4 | yes | **yes** | no | no |
| 7 | 128 | yes | no | no | no |
| 8 | 4 | yes | **yes** | no | no |
| 9 | 128 | yes | no | no | no |
| 10 | 4 | yes | **yes** | no | no |
| 11 | 128 | yes | no | no | no |
| 12 | 4 | yes | **yes** | no | no |
| 13 | 128 | yes | no | no | no |
| 14 | 4 | yes | **yes** | no | no |
| 15 | 128 | yes | no | no | no |
| 16 | 4 | yes | **yes** | no | no |
| 17 | 128 | yes | no | no | no |
| 18 | 4 | yes | **yes** | no | no |
| 19 | 128 | yes | no | no | no |
| 20 | 4 | yes | **yes** | no | no |
| 21 | 128 | yes | no | no | no |
| 22 | 4 | yes | **yes** | no | no |
| 23 | 128 | yes | no | no | no |
| 24 | 4 | yes | **yes** | no | no |
| 25 | 128 | yes | no | no | no |
| 26 | 4 | yes | **yes** | no | no |
| 27 | 128 | yes | no | no | no |
| 28 | 4 | yes | **yes** | no | no |
| 29 | 128 | yes | no | no | no |
| 30 | 4 | yes | **yes** | no | no |
| 31 | 128 | yes | no | no | no |
| 32 | 4 | yes | **yes** | no | no |
| 33 | 128 | yes | no | no | no |
| 34 | 4 | yes | **yes** | no | no |
| 35 | 128 | yes | no | no | no |
| 36 | 4 | yes | **yes** | no | no |
| 37 | 128 | yes | no | no | no |
| 38 | 4 | yes | **yes** | no | no |
| 39 | 128 | yes | no | no | no |
| 40 | 4 | yes | **yes** | no | no |
| 41 | 128 | yes | no | no | no |
| 42 | 4 | yes | **yes** | no | no |
| 43 (MTP) | 0 | no | no | no | **yes** |

**Tallies** (cross-checked against `config.json:66`):

- compress_ratio == 0: **3 layers** (0, 1, 43).
- compress_ratio == 4: **21 layers** (2, 4, 6, …, 42).
- compress_ratio == 128: **20 layers** (3, 5, 7, …, 41).
- hash routing layers: **3** (0, 1, 2). — matches `num_hash_layers == 3`.
- MTP layers: **1** (layer 43). — matches `num_nextn_predict_layers == 1`.

Note one consequence: **layer 2 is the unique layer that combines hash
routing with an Indexer** (compress_ratio=4). It is exercised by the v1
subset pick below.

---

## 3. v1 subset-layer pick

Per plan §3 (High-level Strategy point 3) and §Outcome of this plan, v1 does
**not** target full 44-layer forward. The 275 GB checkpoint
(`config.json` reports 256 experts × 44 layers × FP8) does not fit on a
single B200's 96 GB. v1 instead builds and verifies a 3–4-layer subset that
together exercises every distinct kernel path. The recommended pick is:

| Pick | layer_idx | compress_ratio | hash_routing | MTP | Distinct kernels exercised vs. the others |
|---|---|---|---|---|---|
| **L0** | 0 | 0 | yes | no | Pure SWA attention (no Compressor, no Indexer), **hash_route_lookup** path, plain `mhc_*` pair |
| **L2** | 2 | 4 | yes | no | Compressor + **Indexer** (only ratio=4 has Indexer), `mla_v4_prefill_gather` + `mla_v4_prefill` over swa + indexed compressed cache, hash_route_lookup |
| **L3** | 3 | 128 | no | no | Compressor + **static-indexed** compressed cache (no Indexer), **sqrtsoftplus_topk** scored routing, full `mla_v4_decode` with optional compressed-cache input |
| **L43** | 43 | 0 | no | **yes** | **mtp_embed_hidden_fuse**, MTP-specific hc_head_fn/base/scale, normal Block-style attn/ffn on top |

Distinct-kernel coverage check (every new task from the plan is hit by ≥1
pick):

| Task | L0 | L2 | L3 | L43 |
|---|---|---|---|---|
| `mhc_prenorm_gemm` | x | x | x | x |
| `mhc_pre` | x | x | x | x |
| `mhc_post` | x | x | x | x |
| `mhc_head` (only at end of full forward) | — (covered when assembled) | — | — | — |
| `mla_v4_q_kv_rmsnorm` | x | x | x | x |
| `mla_v4_decode` (decode path, no compressed cache) | x | — | — | x |
| `mla_v4_decode` (decode path, with compressed cache) | — | x | x | — |
| `mla_v4_prefill` | x | x | x | x |
| `mla_v4_prefill_gather` | — (no compressed cache to gather) | x | x | — |
| `inv_rope_fp8_quant_o` | x | x | x | x |
| `compressor` (ratio=4, overlap=true) | — | x | — | — |
| `compressor` (ratio=128, overlap=false) | — | — | x | — |
| `indexer_q_transform` | — | x | — | — |
| `indexer_score_topk` | — | x | — | — |
| `hash_route_lookup` | x | x | — | — |
| `swiglu_clamped` | x | x | x | x |
| `sqrtsoftplus_topk` | — | — | x | x |
| `mtp_embed_hidden_fuse` | — | — | — | x |
| reused V3: `quantize_fp8_layer(scale_ue8m0=True)` | x | x | x | x |
| reused V3: `moe_w13_fp8`, `moe_w2_fp8`, `moe_mul_sum_add` | x | x | x | x |
| reused V3: `rmsnorm`, `linear_fp8`, `linear_fp8_with_residual`, `embed`, `elementwise_add`, `argmax_*` | x | x | x | x |

`mhc_head_layer` is only invoked once per forward (`model.py:721, 766`,
vLLM `deepseek_v4.py:1328-1335`). It is covered by any subset assembled
into a complete forward (the subset still runs full embed → blocks → head).
For per-layer unit testing it is exercised by an explicit
`test_mhc_head_testmode.py` rather than implicitly by a layer pick.

---

## 4. Weight inventory

For the v1 subset, the convert script (plan §Phase C item 3) maps these
named tensors from `/raid/catalyst/models/DeepSeek-V4-Flash-Base/*.safetensors`
to MPK device tensors. Names follow vLLM's renaming
(`deps/vllm/vllm/model_executor/models/deepseek_v4.py:1479-1514`).

**Stored dtype** is what is on disk in the checkpoint (per
`config.json:36-45`):

- Dense weights are FP8 E4M3 with **UE8M0** per-block (128×128) scales
  (`config.json:38-44`, `model.py:138-142`).
- Expert weights are FP8 E4M3 with UE8M0 per-block (128×128) scales
  (`config.json:9`).
- HC params (`hc_*_fn`, `hc_*_base`, `hc_*_scale`) and RMSNorm weights are
  stored in **bf16** on disk and lifted to **fp32** in the official model
  (`model.py:188-189, 666-672, 750-753, 797-800`).
- Embedding and lm_head are stored in bf16 (`model.py:94, 714`).

**Compute dtype** for activations is bf16 except where the kernel demands
fp32 (HC pre/Sinkhorn, RMSNorm reduce, Compressor `wkv/wgate`) or FP8 (linear
GEMMs after `act_quant`).

### 4.1 Global

| Name | Shape | Disk dtype | Source (`model.py`) |
|---|---|---|---|
| `model.embed_tokens.weight` (a.k.a. `embed.weight`) | `[vocab_size, dim] = [129280, 4096]` | bf16 | line 94 |
| `model.norm.weight` (final RMSNorm) | `[dim] = [4096]` | bf16 (lifted fp32) | line 788 |
| `lm_head.weight` (a.k.a. `head.weight`) | `[vocab_size, dim] = [129280, 4096]` | bf16 | line 714 |
| `model.hc_head_fn` | `[hc_mult, hc_dim] = [4, 16384]` | fp32 | lines 797-800 |
| `model.hc_head_base` | `[hc_mult] = [4]` | fp32 | line 799 |
| `model.hc_head_scale` | `[1]` | fp32 | line 800 |

### 4.2 Per base-layer (43 base + the inherited part of the MTP block)

Per Block (`model.py:653-672`, vLLM `deepseek_v4.py:1088-1163`):

| Name (suffix to `model.layers.<i>.`) | Shape | Disk dtype |
|---|---|---|
| `attn_norm.weight` | `[4096]` | bf16 |
| `ffn_norm.weight` | `[4096]` | bf16 |
| `hc_attn_fn` | `[mix_hc, hc_dim] = [24, 16384]` | fp32 |
| `hc_ffn_fn` | `[24, 16384]` | fp32 |
| `hc_attn_base`, `hc_ffn_base` | `[24]` each | fp32 |
| `hc_attn_scale`, `hc_ffn_scale` | `[3]` each | fp32 |

Per Attention (`model.py:439-482`, vLLM `deepseek_v4.py:920-1077`). Some
weights only exist when `compress_ratios[i] != 0`:

| Name (suffix to `model.layers.<i>.attn.`) | Shape | Disk dtype | Notes |
|---|---|---|---|
| `attn_sink` | `[n_heads] = [64]` (padded to 64 for FlashMLA) | fp32 | `model.py:456`; vLLM padding `deepseek_v4.py:960-964` |
| `wq_a.weight` + per-(128×128)-block UE8M0 scale | `[q_lora_rank, dim] = [1024, 4096]` | FP8 E4M3 + UE8M0 | `model.py:457`; vLLM fuses `wq_a + wkv` into `fused_wqa_wkv` `deepseek_v4.py:966-972` |
| `q_norm.weight` | `[1024]` | bf16 | `model.py:458` |
| `wq_b.weight` + scale | `[n_heads * head_dim, q_lora_rank] = [32768, 1024]` | FP8 E4M3 + UE8M0 | `model.py:459` |
| `wkv.weight` + scale | `[head_dim, dim] = [512, 4096]` | FP8 E4M3 + UE8M0 | `model.py:460` (single KV head, MLA-style) |
| `kv_norm.weight` | `[head_dim] = [512]` | bf16 | `model.py:461` |
| `wo_a.weight` + scale | `[n_groups * o_lora_rank, n_heads * head_dim / n_groups] = [8 * 1024, 64 * 512 / 8] = [8192, 4096]` | FP8 E4M3 + UE8M0 | `model.py:462` |
| `wo_b.weight` + scale | `[dim, n_groups * o_lora_rank] = [4096, 8192]` | FP8 E4M3 + UE8M0 | `model.py:463` |

Per Compressor (`model.py:283-305`), exists iff `compress_ratio != 0`. With
`overlap = (compress_ratio == 4)`, `coff = 1 + overlap`:

| Name (suffix to `model.layers.<i>.attn.compressor.`) | Shape | Disk dtype | Notes |
|---|---|---|---|
| `ape` | `[compress_ratio, coff * head_dim]` (e.g. `[4, 2*512]` for ratio=4; `[128, 1*512]` for ratio=128) | fp32 | `model.py:294` |
| `wkv.weight` (vLLM fuses with `wgate` → `fused_wkv_wgate`) | `[coff * head_dim, dim]` (e.g. `[1024, 4096]` for ratio=4) | fp32 (on disk: bf16 → lifted) | `model.py:297` |
| `wgate.weight` | `[coff * head_dim, dim]` | fp32 (on disk: bf16 → lifted) | `model.py:298` |
| `norm.weight` | `[head_dim] = [512]` | bf16 | `model.py:299` |

Per Indexer (`model.py:384-400`), exists iff `compress_ratio == 4`:

| Name (suffix to `model.layers.<i>.attn.indexer.`) | Shape | Disk dtype |
|---|---|---|
| `wq_b.weight` + scale | `[index_n_heads * index_head_dim, q_lora_rank] = [64 * 128, 1024] = [8192, 1024]` | FP8 E4M3 + UE8M0 |
| `weights_proj.weight` | `[index_n_heads, dim] = [64, 4096]` | bf16 |
| `compressor.ape`, `compressor.wkv`, `compressor.wgate`, `compressor.norm` | same shapes as a `head_dim=128, ratio=4` Compressor — `ape[4, 256]`, `wkv[256, 4096]`, `wgate[256, 4096]`, `norm[128]` | as above |

Per MoE (`model.py:609-628`). 256 experts + 1 shared:

| Name (suffix to `model.layers.<i>.ffn.`) | Shape | Disk dtype | Notes |
|---|---|---|---|
| `gate.weight` | `[n_routed_experts, dim] = [256, 4096]` | fp32 (compute) / bf16 (disk) | `model.py:557, 565`. Used in fp32. |
| `gate.bias` (`e_score_correction_bias`) | `[256]` | fp32 | `model.py:562`. **Only present when `layer_id >= num_hash_layers`** (not hash-routing). |
| `gate.tid2eid` | `[vocab_size, num_experts_per_tok] = [129280, 6]` | int32 | `model.py:559`. **Only present when `layer_id < num_hash_layers`** (hash-routing). vLLM `deepseek_v4.py:761-768`. |
| `experts.<e>.w1.weight` + scale, for `e in [0, 256)` | per-expert: `[moe_intermediate_size, dim] = [2048, 4096]` | FP8 E4M3 + UE8M0 | `model.py:591` |
| `experts.<e>.w3.weight` + scale | `[2048, 4096]` | FP8 E4M3 + UE8M0 | `model.py:593` |
| `experts.<e>.w2.weight` + scale | `[dim, moe_intermediate_size] = [4096, 2048]` | FP8 E4M3 + UE8M0 | `model.py:592` |
| MPK packs the per-expert tensors into grouped `w13[E, 2 * inter, dim]` and `w2[E, dim, inter]` (V3 layout) | `[256, 4096, 4096]` and `[256, 4096, 2048]` | FP8 + UE8M0 | reuses V3 `moe_w13_fp8_layer` / `moe_w2_fp8_layer` (`persistent_kernel.py:1881, 1924`) |
| `shared_experts.w1.weight` + scale | `[moe_intermediate_size, dim] = [2048, 4096]` | FP8 + UE8M0 | `model.py:628` |
| `shared_experts.w3.weight` + scale | `[2048, 4096]` | FP8 + UE8M0 | `model.py:628` |
| `shared_experts.w2.weight` + scale | `[4096, 2048]` | FP8 + UE8M0 | `model.py:628`. **Shared expert has `swiglu_limit=0`**, i.e. no clamp (`model.py:628` comment) |

### 4.3 MTP-specific (layer 43)

Inherits all the Block weights above plus, per `model.py:739-755`:

| Name (suffix to `model.mtp.0.`) | Shape | Disk dtype |
|---|---|---|
| `e_proj.weight` + scale | `[dim, dim] = [4096, 4096]` | FP8 E4M3 + UE8M0 |
| `h_proj.weight` + scale | `[dim, dim] = [4096, 4096]` | FP8 E4M3 + UE8M0 |
| `enorm.weight` | `[4096]` | bf16 |
| `hnorm.weight` | `[4096]` | bf16 |
| `norm.weight` | `[4096]` | bf16 |
| `hc_head_fn` | `[hc_mult, hc_dim] = [4, 16384]` | fp32 |
| `hc_head_base` | `[4]` | fp32 |
| `hc_head_scale` | `[1]` | fp32 |

MTP's `embed` and `head` are aliased to the model-level ones (`model.py:793-794`).

---

## 5. Top-level forward pseudo-code (MPK task names)

The kernel names below are exactly the ones in plan §Phase B (Wave-2 task
list) plus the V3-reuse names from plan §"Reuse from V3". **No new task
names are introduced here**. Each `*_layer` corresponds to one Python method
on `PersistentKernel` (existing or to be added).

```text
# ============================================================
# DeepSeek-V4-Flash forward(input_ids, positions)
# Mirrors model.py:802-810 (Transformer.forward) and
#         model.py:689-701 (Block.forward) and
#         model.py:757-767 (MTPBlock.forward).
# ============================================================

# --- 0. Embedding + HC expand ---
# model.py:804      h = self.embed(input_ids)
# model.py:806      h = h.unsqueeze(2).repeat(1, 1, hc_mult, 1)
h        = embed_layer(input_ids)                                 # [T, dim]
h_hc     = repeat_to_hc(h, hc_mult)                               # [T, hc, dim]  (logical; physical layout = [T, hc*dim])
# `repeat_to_hc` is a degenerate broadcast; in MPK it folds into the first
# mhc_prenorm_gemm consumer (no separate task).

# --- 1. 43 base blocks ---
for layer_idx in range(43):
    residual = h_hc                                              # save for hc_post

    # --- 1a. Attention site: HC reduce → attn_norm folded into prenorm GEMM ---
    # mhc_prenorm_gemm fuses the hc reduce's RMSNorm sq-sum into the linear
    # projection by hc_attn_fn (plan §Wave-2 mHC row 1).
    gemm_mul, gemm_sqrsum = mhc_prenorm_gemm_layer(
        residual=h_hc,                       # [T, hc, dim]
        fn=hc_attn_fn[layer_idx],            # [mix_hc, hc*dim] = [24, 16384] fp32
    )
    post_mix, comb_mix, layer_input = mhc_pre_layer(
        gemm_mul=gemm_mul, gemm_sqrsum=gemm_sqrsum,
        hc_scale=hc_attn_scale[layer_idx],   # [3]    fp32
        hc_base=hc_attn_base[layer_idx],     # [24]   fp32
        residual=residual,
        rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20,
    )
    # layer_input: [T, dim] bf16    — the per-token reduced hidden, ready for attn_norm
    # post_mix:    [T, hc]  fp32    — Sinkhorn post-weights for hc_post
    # comb_mix:    [T, hc, hc] fp32 — Sinkhorn comb-weights for hc_post

    # --- 1b. The attention pipeline (sparse_attn under the hood) ---
    # model.py:484-543. MPK splits this into 5 tasks per plan §attention.md.
    x_attn = attention_pipeline(layer_input, layer_idx, positions)

    # --- 1c. HC post: expand attn output back to hc copies + comb residual ---
    # model.py:684-687
    h_hc = mhc_post_layer(
        x=x_attn,                            # [T, dim]
        residual=residual,                   # [T, hc, dim]
        post=post_mix,                       # [T, hc]
        comb=comb_mix,                       # [T, hc, hc]
    )

    # --- 1d. FFN site: HC reduce → ffn_norm folded into prenorm GEMM ---
    residual = h_hc
    gemm_mul, gemm_sqrsum = mhc_prenorm_gemm_layer(
        residual=h_hc, fn=hc_ffn_fn[layer_idx],
    )
    post_mix, comb_mix, layer_input = mhc_pre_layer(
        gemm_mul=gemm_mul, gemm_sqrsum=gemm_sqrsum,
        hc_scale=hc_ffn_scale[layer_idx],
        hc_base=hc_ffn_base[layer_idx],
        residual=residual,
        rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20,
    )

    # --- 1e. MoE FFN ---
    x_ffn = moe_pipeline(layer_input, layer_idx, input_ids)

    # --- 1f. HC post for FFN ---
    h_hc = mhc_post_layer(
        x=x_ffn, residual=residual, post=post_mix, comb=comb_mix,
    )

# --- 2. MTP block (layer 43) — only in MTP-decoding mode ---
# model.py:757-767. Layer 43's input is the model's previous hidden state h_hc
# plus an injected embedding e of the speculative next token.
if mtp_active:
    e        = embed_layer(mtp_input_ids)                        # [T, dim]
    e_n      = rmsnorm_layer(e,   model.mtp[0].enorm.weight)     # [T, dim]
    h_n_hc   = rmsnorm_layer(h_hc, model.mtp[0].hnorm.weight,
                             along_last=True)                    # [T, hc, dim]
    # mtp_embed_hidden_fuse fuses e_proj(e_n).unsqueeze(2) + h_proj(h_n_hc)
    h_hc = mtp_embed_hidden_fuse_layer(
        e_n=e_n, h_n_hc=h_n_hc,
        e_proj_w=model.mtp[0].e_proj.weight,     # FP8 [dim, dim]
        h_proj_w=model.mtp[0].h_proj.weight,     # FP8 [dim, dim]
    )                                            # [T, hc, dim]
    # Then the MTP block runs the same attn + ffn site pair (layer_idx=43).
    h_hc = base_block_forward(h_hc, layer_idx=43,
                              input_ids=mtp_input_ids,
                              positions=positions)

# --- 3. Final hc_head + RMSNorm + lm_head ---
# model.py:719-727 (ParallelHead.forward + hc_head).
h = mhc_head_layer(
    x=h_hc,                                 # [T, hc, dim]
    fn=model.hc_head_fn,                    # [hc_mult, hc_dim] = [4, 16384] fp32
    scale=model.hc_head_scale,              # [1] fp32
    base=model.hc_head_base,                # [4] fp32
    rms_eps=1e-6, hc_eps=1e-6,
)                                           # [T, dim] bf16
h = rmsnorm_layer(h, model.norm.weight)     # [T, dim] bf16
logits = linear_layer(h, lm_head.weight)    # [T, vocab_size]
next_token = argmax_partial_layer(logits) → argmax_reduce_layer(...)
```

### 5.1 `attention_pipeline(layer_input, layer_idx, positions)`

Mirrors `Attention.forward` (`model.py:484-543`) and vLLM
`DeepseekV4MultiHeadLatentAttentionWrapper.forward`
(`deps/vllm/vllm/model_executor/models/deepseek_v4.py:1079-1085` + the
`mla_attn` modules around `deepseek_v4.py:1045-1077`). Compress_ratio at
this layer determines branches.

```text
ratio = compress_ratios[layer_idx]
has_indexer = (ratio == 4)
has_compressor = (ratio != 0)

# --- A. q-lora + Joint RMSNorm of Q-lora and KV-lora streams ---
# In V3 wq_a and wkv are two separate FP8 linears; vLLM V4 fuses them
# into fused_wqa_wkv (deepseek_v4.py:966-972). For MPK v1 we either reuse
# linear_fp8_layer twice or fuse on the convert side. The downstream rms-norm
# step is owned by mla_v4_q_kv_rmsnorm (plan §attention.md).
q_lora, kv_lora = linear_fp8_layer(layer_input, fused_wqa_wkv.weight)
q_lora_n, kv_lora_n = mla_v4_q_kv_rmsnorm_layer(
    q_lora, kv_lora,
    q_norm_w=attn.q_norm.weight,             # [q_lora_rank=1024]
    kv_norm_w=attn.kv_norm.weight,           # [head_dim=512]
    eps=1e-6,
)

# --- B. Q expand + RoPE on rope-dims ---
# model.py:497-499
q = linear_fp8_layer(q_lora_n, attn.wq_b.weight)  # [T, n_heads*head_dim]
q = unflatten_heads(q)                            # [T, n_heads, head_dim]
# RoPE applied only to last rope_head_dim=64 dims (model.py:499).
q = apply_rotary_emb(q, freqs_cis_for(layer_idx, positions))

# --- C. KV expand + RoPE on rope-dims + FP8-quant non-rope dims ---
# model.py:502-506
kv = kv_lora_n                                    # [T, head_dim=512]
kv = apply_rotary_emb_last_rope_dims(kv)
# act_quant(kv[..., :-rope_dim], block=64, ue8m0=true, in_place=true)
# In MPK v1 this is reused V3 quantize_fp8_layer(scale_ue8m0=True)
#   (persistent_kernel.py:1967-1988).
kv_nope_fp8, kv_nope_scale = quantize_fp8_layer(kv[..., :-rope_dim],
                                                block_size=64,
                                                scale_ue8m0=True)
kv = pack_kv(kv_nope_fp8, kv_nope_scale, kv_rope=kv[..., -rope_dim:])

# --- D. Build per-token top-k index lists ---
# model.py:507-515
window_idxs = paged_swa_window_indices(positions, window_size=128)
if has_indexer:
    # Indexer path (plan §sparse.md)
    q_idx = indexer_q_transform_layer(q_lora_n,
                                      wq_b=indexer.wq_b.weight,    # FP8 [8192, 1024]
                                      head_dim=128, rope_dim=64)
    # indexer's own Compressor runs first (it owns kv_cache for indexer scoring)
    indexer_compressor_step(layer_input, layer_idx,
                            compressor=indexer.compressor,
                            rotate=True)            # writes indexer.kv_cache
    weights_proj_out = linear_layer(layer_input,
                                    indexer.weights_proj.weight)   # [T, 64]
    compress_topk_idxs = indexer_score_topk_layer(
        q_idx=q_idx,
        indexer_kv_cache=indexer.kv_cache,
        weights_proj_out=weights_proj_out,
        softmax_scale=1/sqrt(128),
        n_heads=64, topk=512,
        causal_mask_args=(start_pos, seqlen, ratio, offset),
    )
elif ratio == 128:
    compress_topk_idxs = static_get_compress_topk_idxs(ratio, ...)  # CPU-built tiny tensor; reused index buffer
else:
    compress_topk_idxs = None

# --- E. Main Compressor (writes the attention's compressed KV cache) ---
# model.py:524-526
if has_compressor:
    compressor_layer(
        x=layer_input,                              # [T, dim]
        wkv=attn.compressor.wkv.weight,             # fp32 [coff*512, 4096]
        wgate=attn.compressor.wgate.weight,         # fp32 [coff*512, 4096]
        ape=attn.compressor.ape,                    # fp32 [ratio, coff*512]
        norm_w=attn.compressor.norm.weight,         # bf16 [512]
        rotate=False,                               # main Compressor: no Hadamard, FP8
        block_size=64, scale_ue8m0=True,
        overlap=(ratio == 4),
        compressed_kv_cache=attn.kv_cache[:, window_size:],
        freqs_cis=freqs_cis_compress_for(layer_idx),  # uses compress_rope_theta=160000
    )

# --- F. Sparse attention (decode or prefill) ---
# Decode  : model.py:529-533    (sparse_attn over kv_cache)
# Prefill : model.py:518-528    (sparse_attn over fresh kv ++ kv_compress)
topk_idxs = concat(window_idxs, compress_topk_idxs)    # int32 [B, S_q, win+ktop]

if is_decode_phase:
    o = mla_v4_decode_layer(
        q=q,                                            # [B, 1, H, D]
        swa_cache=attn.kv_cache[:, :window_size],
        compressed_cache=(attn.kv_cache[:, window_size:] if has_compressor else None),
        topk_idxs=topk_idxs,
        attn_sink=attn.attn_sink,                       # [64]
        softmax_scale=1/sqrt(head_dim=512),
    )
else:
    gathered_kv = mla_v4_prefill_gather_layer(           # contiguous workspace
        swa_cache=attn.kv_cache[:, :window_size],
        compressed_cache=(attn.kv_cache[:, window_size:] if has_compressor else None),
        topk_idxs=topk_idxs,
    )
    o = mla_v4_prefill_layer(
        q=q, gathered_kv=gathered_kv, attn_sink=attn.attn_sink,
        softmax_scale=1/sqrt(512),
    )

# --- G. Inverse RoPE on attention output + FP8 quant for wo_a consumption ---
# model.py:534 (apply_rotary_emb with inverse=True)
o_fp8, o_scale = inv_rope_fp8_quant_o_layer(
    o=o, freqs_cis=freqs_cis_for(layer_idx, positions),
    rope_head_dim=64,
    block_size=128,
    scale_ue8m0=True,
)

# --- H. O-projection: grouped wo_a then row-parallel wo_b ---
# model.py:537-542
o_gr = o_fp8.view(T, n_groups=8, head_dim*n_heads/n_groups)
wo_a_out = linear_fp8_layer(o_gr, attn.wo_a.weight)   # per-group bmm, FP8
x_attn   = linear_fp8_with_residual_layer(            # the +residual is the wo_b
    wo_a_out.flatten_groups(),
    attn.wo_b.weight,
    residual=None,                                    # residual happens via mhc_post
)
return x_attn                                         # [T, dim] bf16
```

### 5.2 `moe_pipeline(layer_input, layer_idx, input_ids)`

Mirrors `MoE.forward` (`model.py:630-645`) and vLLM `DeepseekV4MoE.forward`
(`deps/vllm/vllm/model_executor/models/deepseek_v4.py:853-918`). The
`scoring_func == "sqrtsoftplus"` and `swiglu_limit == 10.0` knobs are
config-level constants.

```text
# --- A. Gate (routing) ---
# model.py:564-584
router_logits = linear_layer(                          # gate is BF16 GEMM in fp32 math
    layer_input.float(), gate.weight)                  # [T, 256] fp32

if layer_idx < 3:                                      # hash routing
    indices = hash_route_lookup_layer(
        input_ids=input_ids,                           # [T] int32
        tid2eid=gate.tid2eid,                          # [129280, 6] int32
    )                                                  # [T, 6] int32
    # We still apply sqrtsoftplus to compute weights; bias is None.
    scores  = sqrt(softplus(router_logits))            # fused into sqrtsoftplus_topk path
    weights = gather(scores, indices)
    weights /= weights.sum(-1, keepdim=True)
    weights *= 1.5                                     # routed_scaling_factor
else:                                                  # scored routing
    weights, indices = sqrtsoftplus_topk_layer(
        router_logits=router_logits,
        bias=gate.bias,                                # [256] fp32 (noaux_tc)
        topk=6, n_experts=256,
        renormalize=True,
        scale=1.5,
    )

# --- B. Permute tokens → per-expert contiguous batches (reuse V3 path) ---
# Reused tasks: moe_permute_sm100_layer, then FP8 group GEMMs.
permuted_x, perm_meta = moe_permute_sm100_layer(layer_input, indices)

# Activation FP8 quant with UE8M0 scales (reused V3, persistent_kernel.py:1967-1988)
permuted_x_fp8, permuted_x_scale = quantize_fp8_layer(
    permuted_x, block_size=128, scale_ue8m0=True)

# --- C. Group GEMM w13 → SwiGLU(clamped) → group GEMM w2 ---
w13_out = moe_w13_fp8_layer(permuted_x_fp8, permuted_x_scale,
                            w13=ffn.experts.w13_packed,
                            w13_scale=ffn.experts.w13_scale)
# w13_out: [num_permuted_tokens, 2 * moe_intermediate_size] = [..., 2*2048]

mid = swiglu_clamped_layer(                            # new (extension of silu_mul_layer)
    w13_out,
    swiglu_limit=10.0,                                 # config.json:58
    with_clamp=True,                                   # template flag, plan §moe.md
)
# mid: [num_permuted_tokens, moe_intermediate_size=2048]

mid_fp8, mid_scale = quantize_fp8_layer(
    mid, block_size=128, scale_ue8m0=True)
w2_out = moe_w2_fp8_layer(mid_fp8, mid_scale,
                          w2=ffn.experts.w2_packed,
                          w2_scale=ffn.experts.w2_scale)
# w2_out: [num_permuted_tokens, hidden_size=4096]

# --- D. Combine: weighted accumulate routed experts + shared expert ---
y = moe_mul_sum_add_layer(
    expert_outputs=w2_out,
    perm_meta=perm_meta,
    weights=weights,
    indices=indices,
)
# Shared expert: standard SiLU(gate) * up FFN with NO clamp (swiglu_limit=0
# for the shared expert per model.py:628). Reuses V3 silu_mul_layer +
# linear_fp8_layer.
shared = shared_expert_fn(layer_input)
return y + shared
```

---

## 6. Memory budget for the v1 subset

Per-component sizes for **one** base layer at the Flash-Base shapes, FP8
disk format. UE8M0 scales add 1 byte per 128 FP8 elements (i.e. the scale
tensor is 1/128 the size of the FP8 tensor in bytes).

| Component | Quantity per layer | Per-tensor size | Subtotal |
|---|---|---|---|
| `attn.fused_wqa_wkv` FP8 | 1 | `(1024 + 512) * 4096 * 1B = 6.0 MiB` weight + 1/128 scale = ~6.05 MiB | 6.05 MiB |
| `attn.wq_b` FP8 | 1 | `32768 * 1024 * 1B = 32 MiB` + scale → 32.25 MiB | 32.25 MiB |
| `attn.wo_a` FP8 | 1 | `8192 * 4096 * 1B = 32 MiB` + scale → 32.25 MiB | 32.25 MiB |
| `attn.wo_b` FP8 | 1 | `4096 * 8192 * 1B = 32 MiB` + scale → 32.25 MiB | 32.25 MiB |
| `q_norm`, `kv_norm`, `attn_norm`, `ffn_norm` (bf16) | 4 | `(1024 + 512 + 4096 + 4096) * 2B` ≈ 19 KiB | <0.1 MiB |
| `attn_sink` (fp32) | 1 | `64 * 4B = 256 B` | negligible |
| `hc_attn_fn` + `hc_ffn_fn` (fp32) | 2 | `24 * 16384 * 4B = 1.5 MiB` each | 3.0 MiB |
| `hc_attn_base/scale` + `hc_ffn_base/scale` (fp32) | 4 | tiny | <0.01 MiB |
| `compressor.wkv` + `wgate` (fp32) when ratio=4 | 2 | `1024 * 4096 * 4B = 16 MiB` each | 32 MiB |
| `compressor.wkv` + `wgate` (fp32) when ratio=128 | 2 | `512 * 4096 * 4B = 8 MiB` each | 16 MiB |
| `compressor.ape` (fp32) | 1 | `ratio * coff * 512 * 4B` — 8 KiB (ratio=4) or 256 KiB (ratio=128) | <0.3 MiB |
| `compressor.norm` (bf16) | 1 | `512 * 2B` | negligible |
| `indexer.wq_b` FP8 (ratio=4 only) | 1 | `8192 * 1024 * 1B = 8 MiB` + scale → 8.06 MiB | 8.06 MiB |
| `indexer.weights_proj` (bf16) | 1 | `64 * 4096 * 2B = 0.5 MiB` | 0.5 MiB |
| `indexer.compressor.{wkv,wgate}` (fp32) (ratio=4 only) | 2 | `256 * 4096 * 4B = 4 MiB` each | 8 MiB |
| `ffn.gate.weight` (fp32) | 1 | `256 * 4096 * 4B = 4 MiB` | 4 MiB |
| `ffn.gate.bias` (fp32) or `tid2eid` (int32) | 1 | `256 * 4B = 1 KiB` OR `129280 * 6 * 4B = 3.0 MiB` | up to 3.0 MiB |
| **Routed experts: w13 + w2 packed FP8** | 256 | `w13`: `4096 * 4096 * 1B = 16 MiB` + scale → 16.13 MiB. `w2`: `4096 * 2048 * 1B = 8 MiB` + scale → 8.06 MiB. Per-expert ≈ 24.2 MiB. | **256 × 24.2 ≈ 6.2 GiB** |
| `shared_expert.{w1,w2,w3}` FP8 | 3 | `2048 * 4096 * 1B = 8 MiB` + scale = 8.06 MiB each | 24.2 MiB |

**Per-layer total** (dominated by the 256 routed experts): ~6.35 GiB.

| Subset member | Notable cost adds | Layer GiB |
|---|---|---|
| L0 (ratio=0, hash) | no compressor/indexer; +`tid2eid` 3 MiB | ~6.30 |
| L2 (ratio=4, hash + indexer) | +compressor 32 MiB +indexer 16.5 MiB +`tid2eid` 3 MiB | ~6.36 |
| L3 (ratio=128, scored) | +compressor 16 MiB | ~6.33 |
| L43 (MTP, ratio=0, scored) | inherits Block, +e_proj/h_proj 8 MiB, +mtp hc_head 0.25 MiB | ~6.31 |

| Global | size |
|---|---|
| `embed` bf16 | `129280 * 4096 * 2B = 1.0 GiB` |
| `lm_head` bf16 | 1.0 GiB |
| final `norm`, `hc_head_*` | negligible (~1 MiB) |

**v1 subset weight footprint** (4 layers + global):
4 × 6.33 + 2 × 1.0 ≈ **27.3 GiB**. Activation + KV-cache budget for a
batch of 1 with `max_seq_len` ≤ 4096 is well under 10 GiB
(`max_position_embeddings=1048576` is the model max, not the v1 batch).
**Conclusion: the v1 subset comfortably fits on 1 B200 (96 GiB).**

---

## 7. Linkage to sibling spec files

- `docs/mpk/deepseek_v4/hc.md` — `mhc_prenorm_gemm`, `mhc_pre`, `mhc_post`,
  `mhc_head`. Math from `model.py:674-687, 729-736` and the four tasks in
  plan §Wave-2 mHC table; vLLM impls in
  `deps/vllm/vllm/model_executor/layers/mhc.py`.
- `docs/mpk/deepseek_v4/attention.md` — 5 tasks
  (`mla_v4_q_kv_rmsnorm`, `mla_v4_decode`, `mla_v4_prefill`,
  `mla_v4_prefill_gather`, `inv_rope_fp8_quant_o`). Math from
  `model.py:436-543` and vLLM's `DeepseekV4Attention` /
  `DeepseekV4MultiHeadLatentAttentionWrapper`
  (`deps/vllm/vllm/model_executor/models/deepseek_v4.py:920-1085`).
- `docs/mpk/deepseek_v4/sparse.md` — Compressor + Indexer
  (`compressor_layer`, `indexer_q_transform_layer`,
  `indexer_score_topk_layer`). Math from `model.py:279-433` plus vLLM's
  `deepseek_compressor.py` and `fused_indexer_q.py`. Must clearly document
  the **`compress_rope_theta=160000`** branch and the per-ratio behavior
  matrix (ratio=4 vs ratio=128).
- `docs/mpk/deepseek_v4/moe.md` — `hash_route_lookup`, `swiglu_clamped`
  (extension of `silu_mul_layer`), `sqrtsoftplus_topk` (extension of
  `topk_sigmoid_sm100`). Math from `model.py:546-645`. Reuses V3 group
  GEMMs and `quantize_fp8_layer(scale_ue8m0=True)`.
- `docs/mpk/deepseek_v4/mtp.md` — `mtp_embed_hidden_fuse` (the only new
  task) + reuse of the Block pipeline. Math from `model.py:739-767`.

---

## 8. Open questions

- **OPEN: MTP `compress_ratios[43]` interpretation.** `config.json:66`
  supplies a 44-entry array with entry 43 = 0; the official `model.py`
  passes `args.compress_ratios` to `Attention.__init__` directly
  (`model.py:453`), so MTP attention is dense SWA. vLLM diverges
  (`deps/vllm/vllm/model_executor/models/deepseek_v4.py:951-954` forces
  MTP's `compress_ratio = max(1, ...) = 1`, also dense). The semantics agree
  (dense SWA), but the index handling differs; v1 should follow the
  official `model.py` convention (use the entry directly, value 0 ⇒ no
  Compressor/Indexer/no compressed cache).

- **OPEN: vLLM fuses `wq_a + wkv` into `fused_wqa_wkv`
  (`deps/vllm/vllm/model_executor/models/deepseek_v4.py:1344-1345, 966-972`)
  but the official `model.py` keeps them separate.** v1 spec uses the
  separate-linear semantics for clarity (one `linear_fp8` for `wq_a`, one
  for `wkv`); a follow-up perf pass may fuse them. The convert script must
  either keep them separate or de-fuse the vLLM-fused checkpoint shard
  back into per-projection tensors.

- **OPEN: `mhc_prenorm_gemm` fallback path.** Plan §Wave-2 row 1 says the
  initial implementation may decompose into RMSNorm-then-linear if porting
  the CUTLASS prenorm GEMM (`deps/vllm/vllm/third_party/deep_gemm/include/
  deep_gemm/impls/sm100_tf32_hc_prenorm_gemm.cuh`) is high-risk. This
  overview assumes the fused form for outputs but Wave-2 will need to
  decide whether `mhc_pre_layer` consumes `gemm_mul + gemm_sqrsum` (fused
  path) or `linear_out + rsqrt_per_token` (decomposed path). The Python
  signature should accept both.

- **OPEN: `repeat_to_hc` materialization.** `model.py:806` materializes the
  hc-expanded hidden state by `unsqueeze(-2).repeat(...)`. In MPK we
  intend to fold this broadcast into the first `mhc_prenorm_gemm`
  consumer rather than allocate an `hc_mult`× larger tensor. Confirm with
  the Wave-2 `mhc_prenorm_gemm` agent that its kernel can read a
  non-materialized broadcast (i.e. with the `hc` stride set to 0 for the
  first layer's residual input).

- **OPEN: Paged-KV-cache extensions for v4.** Plan §Phase A includes a
  separate audit task to confirm MPK's `paged_kv_indptr_buffer`,
  `paged_kv_indices_buffer`, and `paged_kv_last_page_len_buffer`
  (`python/mirage/mpk/persistent_kernel.py:1401, 1422`) handle the
  Compressor's **variable-stride compressed-KV pages** and the **Indexer's
  fp4-format pages**. That audit's report should be cross-referenced here
  once it lands.

- **OPEN: Conversion of UE8M0 scales for routed experts.** vLLM stores
  scales as `float8_e8m0fnu` on disk and views them as `uint8` to preserve
  raw exponent bytes (`deps/vllm/vllm/model_executor/models/deepseek_v4.py:1383-1388`).
  MPK's `quantize_fp8_layer(scale_ue8m0=True)` produces packed `uint32`
  scales (`persistent_kernel.py:1974-1988`). The convert script must match
  this packing; confirm endianness and group-size when authoring
  `demo/deepseek_v4/models/convert.py`.

- **OPEN: Hadamard transform implementation in `indexer_q_transform`.**
  Plan §Out-of-scope item 5 explicitly allows v1 to compute the Hadamard
  as a matmul against a precomputed Hadamard matrix instead of a
  fast-transform. The sparse spec (`sparse.md`) must commit to one of
  these for v1 and document the precomputed matrix shape (`[index_head_dim,
  index_head_dim] = [128, 128]` int8).
