# linear_cublas — routine nn.Linear instances in V4-Flash NVIDIA

Every `nn.Linear` call in V4-Flash NVIDIA is a cuBLAS GEMM (or sharded cuBLAS GEMM via vLLM's TP wrappers). They all share the same kernel implementation (cuBLAS `gemmEx`/`hgemm` selected at runtime by `dispatch_unquantized_gemm()` for bf16, or the FP8 quant method's CUTLASS / DeepGEMM kernel for fp8 weights), so we collapse them into one table here rather than one spec file each.

The vLLM TP wrappers (`MergedColumnParallelLinear`, `ColumnParallelLinear`, `RowParallelLinear`, `ReplicatedLinear`, `VocabParallelEmbedding`, `ParallelLMHead`) handle weight sharding + collective ops, but the underlying GEMM kernel is unchanged.

V4-Flash-Base config used below (`deps/deepseek_v4/DeepSeek-V4-Flash/config.json`):
- `hidden_size = 4096`
- `vocab_size = 129280`
- `q_lora_rank = 1024`
- `head_dim = 512` (== `kv_lora_rank` in V4's MLA naming)
- `qk_rope_head_dim = 64`, `nope_head_dim = head_dim - qk_rope_head_dim = 448`
- `num_attention_heads = 64`, `o_groups = 8`, `o_lora_rank = 1024`
- `n_routed_experts = 256`, `n_shared_experts = 1`, `moe_intermediate_size = 2048`
- `hc_mult = 4`
- Indexer: `index_n_heads = 64`, `index_head_dim = 128`, `qk_rope_head_dim = 64`

## Table

| Instance | in_dim | out_dim | dtype (input/output) | Parallelism | Caller |
|---|---|---|---|---|---|
| `fused_wqa_wkv` | 4096 | [1024, 512] (`[q_lora_rank, head_dim]`) | bf16 / bf16 | `MergedColumnParallelLinear` with `disable_tp=True` (replicated; fused W_qa ‖ W_kv) | `vllm/models/deepseek_v4/nvidia/model.py:658-665` (`DeepseekV4Attention.__init__`) |
| `wq_b` | 1024 (`q_lora_rank`) | 32768 (`n_heads * head_dim = 64 * 512`) | bf16 / bf16 | `ColumnParallelLinear` | `vllm/models/deepseek_v4/nvidia/model.py:667-674` |
| `wo_a` | 4096 (`n_heads * head_dim / o_groups = 64*512/8`) | 8192 (`o_groups * o_lora_rank = 8*1024`) | bf16 weight; activation FP8-quantized by `fused_inv_rope_fp8_quant` before this GEMM | `ColumnParallelLinear` with `is_bmm=True`, `bmm_batch_size = n_local_groups` | `vllm/models/deepseek_v4/nvidia/model.py:677-686` — output consumed by `deepseek_v4_fp8_einsum` (`attention.py:338`), so this entry's GEMM is actually an FP8 batched matmul; the `nn.Linear` shape is the abstract view used for weight loading. |
| `wo_b` | 8192 (`o_groups * o_lora_rank`) | 4096 (`hidden_size`) | bf16 / bf16 | `RowParallelLinear` | `vllm/models/deepseek_v4/nvidia/model.py:687-694` |
| `gate_up_proj` (shared expert) | 4096 (`hidden_size`) | [2048, 2048] (`[moe_intermediate_size * n_shared_experts] * 2`) | bf16 / bf16 | `MergedColumnParallelLinear` (or `disable_tp=True` if `is_sequence_parallel`) | `vllm/models/deepseek_v4/nvidia/model.py:82-89` (`DeepseekV4MLP.__init__`) |
| `down_proj` (shared expert) | 2048 | 4096 (`hidden_size`) | bf16 / bf16 | `RowParallelLinear` | `vllm/models/deepseek_v4/nvidia/model.py:90-98` |
| `GateLinear` (router gate) | 4096 (`hidden_size`) | 256 (`n_routed_experts`) | bf16 in / fp32 out (`out_dtype=torch.float32`) | Replicated | `vllm/models/deepseek_v4/nvidia/model.py:437-443` (`DeepseekV4MoE.__init__`) — **NOTE: only when the fallback F.linear path is taken.** Tier-1 (`dsv3_router_gemm`, specialized for E=256/M≤16) and tier-2 (`fp32_router_gemm`, specialized for E=256/M≤32) are NOT routine cuBLAS — they have their own dedicated specs. See R-0 inventory G3. For V4-Flash with `M > 32`, the GateLinear falls back to `F.linear` and IS the cuBLAS GEMM listed here. |
| `embed_tokens` | 129280 (`vocab_size`) | 4096 (`hidden_size`) | int32 lookup → bf16 | `VocabParallelEmbedding` | `vllm/models/deepseek_v4/nvidia/model.py:971-976` (instance); invoked via `DeepseekV4Model.embed_input_ids` at line 1064. **NOT a cuBLAS GEMM** — invokes `F.embedding` (see `vocab_parallel_embedding.md`). Listed here because `VocabParallelEmbedding` is a TP-wrapper sibling of the Linear wrappers and shares the same weight-loading infrastructure; its GPU op is `aten::index_select`, not cuBLAS. |
| `lm_head` | 4096 (`hidden_size`) | 129280 (`vocab_size`) | bf16 / bf16 (logits cast to fp32 in `LogitsProcessor` downstream sampler) | `ParallelLMHead` | `vllm/models/deepseek_v4/nvidia/model.py:1282-1286` (instance); invoked via `DeepseekV4ForCausalLM.compute_logits` at line 1301 → `LogitsProcessor.forward` → `_get_logits` → `lm_head.quant_method.apply` → `dispatch_unquantized_gemm()` → cuBLAS bf16 GEMM. |
| `e_proj` (MTP) | 4096 (`hidden_size`) | 4096 (`hidden_size`) | bf16 / bf16 | `ReplicatedLinear` (no TP) | `vllm/models/deepseek_v4/nvidia/mtp.py:87-93` (`DeepSeekV4MultiTokenPredictorLayer.__init__`) |
| `h_proj` (MTP) | 4096 (`hidden_size`) | 4096 (`hidden_size`) | bf16 / bf16 | `ReplicatedLinear` (no TP) | `vllm/models/deepseek_v4/nvidia/mtp.py:94-100` |
| `shared_head.head` (MTP LM head) | 4096 (`hidden_size`) | 129280 (`vocab_size`) | bf16 / bf16 | `ParallelLMHead` (inside `SharedHead`, instantiated at `mtp.py:118-120`) | invoked via `DeepSeekV4MultiTokenPredictor.compute_logits` at `vllm/models/deepseek_v4/nvidia/mtp.py:252` |
| Compressor `fused_wkv_wgate` | 4096 (`hidden_size`) | [`coff * head_dim`, `coff * head_dim`] = [1024, 1024] for `compress_ratio=4` (`coff=2`); [512, 512] for `compress_ratio=128` (`coff=1`) | bf16 / fp32 (the Linear is constructed with `dtype=torch.float32` at the parameter level, but `quant_config=None`; activations are bf16) | `MergedColumnParallelLinear` with `disable_tp=True` | `vllm/models/deepseek_v4/compressor.py:225-233` (`DeepseekCompressor.__init__`) |
| Indexer `wq_b` | 1024 (`q_lora_rank`) | 8192 (`index_n_heads * index_head_dim = 64 * 128`) | bf16 / bf16 | `ReplicatedLinear` (with `quant_config` — fp8 if active) | `vllm/models/deepseek_v4/attention.py:753-759` (`DeepseekV4Indexer.__init__`) |
| Indexer `weights_proj` | 4096 (`hidden_size`) | 64 (`index_n_heads`) | bf16 / bf16 | `ReplicatedLinear` with `quant_config=None` | `vllm/models/deepseek_v4/attention.py:760-766` |

## Notes

### Dimensions
- All dimensions above are for V4-Flash-Base (config at `deps/deepseek_v4/DeepSeek-V4-Flash/config.json`). The vLLM model code reads these from `config.<attr>`; sharding constants come from the parallel-config (TP/EP/PP) at runtime. The values listed are the **logical, pre-shard** sizes — the cuBLAS GEMM that actually runs on each rank uses the rank-local slice.
- `q_lora_rank=1024` (NOT 1536 as in DeepSeek-V3) — V4-Flash uses a smaller Q-LoRA rank than V3.
- `head_dim=512` (single MLA value; equals `kv_lora_rank` in V4's MLA naming; see `vllm/models/deepseek_v4/nvidia/model.py:768`).
- `hc_mult=4` (NOT 8). The Hyper-Connections expansion factor.

### `wo_a` is FP8 BMM, not cuBLAS GEMM
- `wo_a.is_bmm = True` (`nvidia/model.py:685`), `bmm_batch_size = n_local_groups`. The weight is sharded as a 3D tensor `[n_groups, o_lora_rank, n_heads * head_dim / n_groups]`. The "Linear" forward is never actually called — the GEMM happens via `fp8_einsum("bhr,hdr->bhd", (o_fp8, o_scale), (wo_a_fp8, wo_a_scale), z, recipe=...)` at `vllm/models/deepseek_v4/attention.py:338`. See R-0 inventory entry E2 (`deepseek_v4_fp8_einsum`) for the actual kernel; cuBLAS on SM90 / DeepGEMM on SM100.

### Router gate tier dispatch
- `GateLinear` (router gate) has a **3-tier dispatch** (see R-0 inventory G3 and `dsv3_router_gemm.md`):
  - Tier 1: `dsv3_router_gemm` — specialized CUDA kernel for `E=256, M≤16` (small-batch decode). Best on V4-Flash decode.
  - Tier 2: `fp32_router_gemm` — specialized CUDA kernel for `E=256, M≤32`. Mid-size batch.
  - Tier 3: `F.linear` fallback (cuBLAS) — **this table's entry**. Activated when `M > 32` (large prefill) or when neither specialized path matches the shape.
- For V4-Flash decoding workloads (typically `M ≤ 16`), tier 1 dominates and this cuBLAS row is rarely hit.

### `embed_tokens` is NOT cuBLAS
- Listed for inventory completeness — `VocabParallelEmbedding` is the sibling TP-wrapper but its forward is `F.embedding` → `aten::embedding` → `aten::index_select` (CUDA gather), not a GEMM. See `vocab_parallel_embedding.md`.

### FP8-weighted Linears
- Most V4-Flash Linears (notably `wq_b`, `wo_a`, MoE shared expert) have `quant_config` set to FP8 (per `config.quantization_config.quant_method = "fp8"`, `weight_block_size = [128, 128]`). The forward dispatches to `Fp8LinearMethod.apply`, which performs:
  1. Per-token FP8 quantization of the activation (online).
  2. FP8 block-scaled GEMM (cuBLAS or DeepGEMM).
  3. Dequantization back to bf16 using the per-block scales.
- The "Linear" entry in this table is unchanged — the bf16-equivalent shape is the same, and the GEMM is still cuBLAS-backed (the FP8 codepath inside `dispatch_unquantized_gemm()` / `Fp8LinearMethod.apply` uses cuBLAS `gemmEx` with FP8 inputs on SM100, or CUTLASS on older arches). The FP8 metadata (scales) lives alongside the weight tensor.

### Compressor `fused_wkv_wgate` dtype
- Constructed with `quant_config=None`, `disable_tp=True`. Inputs are bf16 hidden states; outputs are written into bf16 buffers then immediately split and consumed by `save_partial_states` / `compress_norm_rope_store_*` which read them in fp32 (the kernels upcast).
- The `dtype=torch.float32` reference in the V4-Flash inference reference (`deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:297-298`) is the *parameter* dtype, not the activation dtype.

### All entries use cuBLAS-backed GEMM
- Via `torch.nn.functional.linear` (under vLLM's Linear wrappers) → `aten::linear` → cuBLAS `gemmEx`. No custom CUDA kernel for the basic forward.
- Exceptions:
  - `wo_a` (FP8 einsum) — see above.
  - `embed_tokens` (gather, not GEMM) — see above.
  - `GateLinear` tier 1/2 (specialized CUDA) — see above.
  - FP8-quant path is still cuBLAS for the actual matmul (with FP8 inputs).

### Parallelism wrappers
- `MergedColumnParallelLinear` — concatenates multiple column-parallel weights for one fused GEMM with multi-output (`gate_up_proj`, `fused_wqa_wkv`, `fused_wkv_wgate`). The output is split downstream.
- `ColumnParallelLinear` — output dim sharded across TP ranks. No collective on the forward (caller manages gather).
- `RowParallelLinear` — input dim sharded; emits an `all_reduce` after the GEMM unless `reduce_results=False`.
- `ReplicatedLinear` — no sharding; full weight on every rank.
- `VocabParallelEmbedding` / `ParallelLMHead` — vocab dim sharded; gather via `LogitsProcessor` (lm_head) or `all_reduce` after masked embedding lookup (`embed_tokens`).
- `disable_tp=True` on a `MergedColumnParallelLinear` makes it effectively `ReplicatedLinear` with the multi-output split semantics preserved.

### Sources
- `MergedColumnParallelLinear`, `ColumnParallelLinear`, `RowParallelLinear`, `ReplicatedLinear`: `vllm/model_executor/layers/linear.py`.
- `VocabParallelEmbedding`, `ParallelLMHead`: `vllm/model_executor/layers/vocab_parallel_embedding.py` (see `vocab_parallel_embedding.md`).
- `Fp8LinearMethod`: `vllm/model_executor/layers/quantization/fp8.py`.
- `dispatch_unquantized_gemm`: `vllm/model_executor/layers/utils.py` — returns the CUDA bf16 GEMM kernel pointer (`torch.nn.functional.linear` on most platforms).
