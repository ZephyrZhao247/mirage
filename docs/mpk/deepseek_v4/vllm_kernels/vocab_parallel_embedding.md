# vocab_parallel_embedding

## Identity
- Source file (Python wrapper): `vllm/model_executor/layers/vocab_parallel_embedding.py:192-505` (`class VocabParallelEmbedding(PluggableLayer)`).
- Source file (GPU op): `vllm/model_executor/layers/vocab_parallel_embedding.py:67-78` (`UnquantizedEmbeddingMethod.embedding` → `F.embedding`).
- Source file (TP masking): `vllm/model_executor/layers/vocab_parallel_embedding.py:162-187` (`get_masked_input_and_mask`, `@torch.compile`-decorated).
- Language/DSL: **PyTorch native** — `torch.nn.functional.embedding`, which lowers to a CUDA op (`aten::embedding` → `aten::index_select`). No custom kernel, no Triton.
- Third-party dep: none beyond PyTorch.
- Registered as: Python `PluggableLayer` (`@PluggableLayer.register("vocab_parallel_embedding")`, line 191). NOT a `torch.ops` op.
- Mask-fusion: `get_masked_input_and_mask` is wrapped with `@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)` (line 162) — TorchInductor fuses the pointwise mask ops (compare, subtract, multiply, OR) into a single kernel at runtime. The mask fuse is only on the TP-shard path (`tp_size > 1`); for `tp_size=1` only the raw `F.embedding` runs.

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/models/deepseek_v4/nvidia/model.py:1064` | `DeepseekV4Model.embed_input_ids` (called from `DeepseekV4Model.forward` and `DeepseekV4ForCausalLM.embed_input_ids` at line 1294-1295) | `input_ids: [num_tokens]` int32/int64 → `hidden_states: [num_tokens, hidden_size=4096]` | int32 in / bf16 out (weight dtype) | always on (PP first rank); skipped if `inputs_embeds` provided directly |
| `vllm/models/deepseek_v4/nvidia/mtp.py:205` | `DeepSeekV4MultiTokenPredictor.embed_input_ids` (called from `DeepSeekV4MultiTokenPredictor.forward` at line 217) | same shape | int32 in / bf16 out | always on for MTP draft step; skipped if `inputs_embeds` provided |

Both call sites construct an instance with `num_embeddings = config.vocab_size = 129280` (V4-Flash-Base), `embedding_dim = 4096`. The MTP instance is separate (`mtp.py:198-202`) and lives in `model.mtp.embed_tokens`; the main model's instance is `model.embed_tokens` at `nvidia/model.py:971-976`.

Indirectly reachable: `ParallelLMHead` (line 510) is a subclass of `VocabParallelEmbedding` but its `forward()` raises (line 572-574). LM-head GEMM goes through `quant_method.apply` (the `linear`/GEMM path), NOT `quant_method.embedding`. So `ParallelLMHead` does NOT invoke `F.embedding`.

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `input_` | `[num_tokens]` (1D) | int32 (V4 default) or int64 (line 491 calls `.long()` before the embedding) | contiguous | Token ID per request slot. Range `[0, vocab_size)` for in-vocab tokens. |
| `self.weight` | `[num_embeddings_per_partition, embedding_dim]` (`= [129280, 4096]` for `tp_size=1`; `[129280/tp_size, 4096]` for sharded) | bf16 (matches model default dtype) | contiguous, row-major | Embedding table. Shard layout per `_get_indices` (line 260-267); each TP rank holds a contiguous slice of vocab IDs. |
| `tp_size` | scalar int | — | — | Number of TP ranks; controls whether the mask fuse runs. |
| `shard_indices` | dataclass | — | — | Vocab slice owned by this rank: `(org_vocab_start_index, org_vocab_end_index, num_org_vocab_padding, added_vocab_start_index, added_vocab_end_index)`. Computed once in `__init__` at line 260-267. |

For V4-Flash with `padding_size=64`, the vocab is padded from 129280 (already divisible by 64) to itself, so no padding rows are added.

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `output` | `[num_tokens, embedding_dim=4096]` | bf16 (matches `weight.dtype`) | `F.embedding` returns row-major contiguous | Embedded hidden state. For `tp_size > 1`, rows corresponding to out-of-shard token IDs are zero-filled (`masked_fill_` at line 494) before the `all_reduce` collective (line 496). |

## Grid / Block

- This is `F.embedding`, which calls into ATen's `aten::embedding` (which calls `aten::index_select`). **No vLLM-specific grid / block configuration.** PyTorch's CUDA backend launches a generic gather kernel:
  - For `F.embedding(input_, weight)` with `input_.shape = [N]` and `weight.shape = [V, D]`, the underlying CUDA kernel uses a 2D grid of `(ceil(N*D / 128), 1)` blocks of 128 threads (default ATen). Exact specifics are PyTorch-version-dependent; ATen kernel: `at::native::indexSelectSmallIndex` or `indexSelectLargeIndex` depending on weight size.
- The `get_masked_input_and_mask` torch.compile-fused pointwise op (TP path only) launches one Inductor-generated kernel covering: `(input >= start) & (input < end)`, `mask | mask_added`, `input - offset`, `vocab_mask * (...)`. Per `@torch.compile(dynamic=True)`, grid/block sizes are auto-selected by Inductor at first compile.
- The post-embedding `masked_fill_` (line 494) and `tensor_model_parallel_all_reduce` (line 496) are separate PyTorch ops, each launching its own kernel(s) — `masked_fill_` is a pointwise op; the all-reduce dispatches to NCCL.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:96-105` (`class ParallelEmbedding.forward`). The reference uses the same TP-shard pattern: `mask = (x < start) | (x >= end); x = x - start; x[mask] = 0; y = F.embedding(x, weight); y[mask] = 0; all_reduce(y)`. vLLM's `VocabParallelEmbedding.forward` (`vocab_parallel_embedding.py:477-497`) implements the same algorithm — slightly more general (handles LoRA-added vocab range too via `added_vocab_*` indices).

```python
# PyTorch-operator equivalent of the V4-Flash NVIDIA call site (tp_size = 1 case):
#
# Inputs:
#   input_ids:   [T] int32       (T = num_tokens)
#   weight:      [V, H]  bf16    (V = 129280, H = 4096)
#
# tp_size == 1 path: skip masking entirely.
output = F.embedding(input_ids.long(), weight)        # [T, H] bf16

# tp_size > 1 path (V4 production rarely uses TP for embed, but if so):
#   masked_input, input_mask = get_masked_input_and_mask(
#       input_ids, org_vocab_start, org_vocab_end, num_org_vocab_padding,
#       added_vocab_start, added_vocab_end,
#   )
#   = (
#       vocab_mask * (input_ids - valid_offset),     # in-shard rows shifted to local index
#       ~vocab_mask,                                 # True for out-of-shard rows
#     )
#   output_parallel = F.embedding(masked_input.long(), weight)   # [T, H]
#   output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)    # zero out-of-shard
#   output = tensor_model_parallel_all_reduce(output_parallel)   # sum across TP ranks
```

Notes on fusion / quant:
- **No quantization**: weight is bf16, output is bf16. Even when `quant_config` is set (line 270-280), `UnquantizedEmbeddingMethod` is used unconditionally for embedding *forward*. The check at line 279-280 (`is_embedding_layer = type(self) is VocabParallelEmbedding`) ensures only quant methods that implement `embedding()` are kept — in practice for V4 it's always the unquantized path.
- The torch.compile fusion of `get_masked_input_and_mask` (line 162) collapses 6 pointwise ops into one kernel — only on `tp_size > 1`. For `tp_size = 1`, the `else: masked_input = input_` branch (line 488-489) bypasses the fuse entirely.

## Config-dependent dispatch

- Activation condition: always on (V4 PP first rank, MTP draft). Skipped when caller passes `inputs_embeds` (e.g., for multi-modal: at `nvidia/model.py:1061-1062`, when `inputs_embeds is not None`, `embed_input_ids` is bypassed). V4 text-only does not pass `inputs_embeds`.
- Variants:
  - `tp_size == 1` (fast path, no mask): pure `F.embedding`.
  - `tp_size > 1` (sharded path): masked-`F.embedding` + `all_reduce`. Mask is a fused Inductor kernel.
  - `VLLM_BATCH_INVARIANT` — affects `UnquantizedEmbeddingMethod.apply` (linear-style `linear_batch_invariant`, line 73-74) but NOT `embedding()` (line 77-78 unconditional `F.embedding`).
  - `ParallelLMHead` (subclass at line 510-575) — uses the same weight tensor but never calls `embedding()`. Its forward intentionally raises (line 574); the GEMM goes through `quant_method.apply` in `LogitsProcessor._get_logits` (`logits_processor.py:96`).
- Downstream consumer:
  - V4 main model: output is reshaped at `nvidia/model.py:1065` via `unsqueeze(-2).repeat(1, hc_mult, 1)` to shape `[T, hc_mult, hidden_size]` before entering the first decoder layer.
  - V4 MTP: output feeds `fused_mtp_input_rmsnorm` at `mtp.py:144` as `inputs_embeds` (along with `previous_hidden_states`).
- Hard preconditions: `input_.numel()` ≤ kernel limits; `input_.dtype` is castable to int64 (line 491 calls `.long()`).
