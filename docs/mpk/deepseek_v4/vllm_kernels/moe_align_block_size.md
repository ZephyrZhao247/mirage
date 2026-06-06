# moe_align_block_size

## Identity
- Python wrapper: `vllm/model_executor/layers/fused_moe/moe_align_block_size.py:11-103` (`moe_align_block_size`); allocates output tensors and calls `torch.ops._C.moe_align_block_size` at line 90.
- CUDA host launcher: `csrc/moe/moe_align_sum_kernels.cu:495-587` (function `moe_align_block_size`).
- CUDA kernel bodies:
  - "Standard" path: `moe_align_block_size_kernel` (lines 322-336) wrapping the templated `_moe_align_block_size` device function (lines 81-181), launched as **2 thread blocks** (line 560).
  - Followed by `count_and_sort_expert_tokens_kernel` (lines 338-347) wrapping `_count_and_sort_expert_tokens` (lines 291-320), launched as a 2-D grid `(1, actual_blocks)` (line 575-584).
  - "Small batch / few experts" path: `moe_align_block_size_small_batch_expert_kernel` (lines 365-378) wrapping `_moe_align_block_size_small_batch_expert` (lines 183-289), launched as a **single thread block** (line 539-547).
- Language/DSL: **CUDA** (CUB BlockScan + atomic histogram + shared-memory cumsum).
- Third-party dep: CUB (vendored with CUDA toolkit; `cub::BlockScan` at line 55, 144).
- Registered as: `torch.ops._C.moe_align_block_size` (torch binding at `csrc/moe/torch_bindings.cpp`).

## Call sites

| Caller file:line | Module / function | Input shape sketch | Dtype | Config gate |
| --- | --- | --- | --- | --- |
| `vllm/model_executor/layers/fused_moe/fused_moe.py:1465` | `_prepare_expert_assignment` → `fused_experts_impl` | `topk_ids: [T, top_k]`, `expert_map: [E_global]` (or None) | int32 (or int64) ids; int32 expert_map | active when `_prepare_expert_assignment` does NOT take the naive fast path (i.e. `num_tokens * top_k * 4 > global_num_experts` or `expert_map is not None`) — fused_moe.py:1443-1463 |
| `vllm/model_executor/layers/fused_moe/experts/triton_moe.py:~571` | `TritonExperts.apply` (modular FusedMoE entry) | same | same | always when `TritonExperts` is the selected backend |
| `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py:329, 497` | `MarlinExperts.apply` / batched variant | same | same | when Marlin backend is selected |
| `vllm/model_executor/layers/fused_moe/experts/fused_humming_moe.py:481` | `HummingMoE` experts variant | same | same | when Humming backend is selected |
| `vllm/models/deepseek_v4/nvidia/model.py:573` (transitively) | `DeepseekV4MoE._forward_fused_moe → FusedMoE.forward → TritonExperts.apply` | V4-Flash-Base: `[T, 8]` int32 → with `block_size = config["BLOCK_SIZE_M"]` (autotuned, typically 64 or 128) | same | `moe_backend != "deep_gemm_mega_moe"` (so always for V4-Flash-Base) |

Runs once per `fused_experts_impl` call (i.e. once per MoE-layer forward) regardless of L1/L2 — the same `(sorted_token_ids, expert_ids, num_tokens_post_padded)` triple is reused for both GEMM launches (fused_moe.py:1659-1721).

## Inputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `topk_ids` | `[T, top_k]` (V4-Flash: `top_k=8`) | int32, int64, uint32, or uint64 (dispatched by `VLLM_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES`, line 521) | row-major contiguous | Per-token routed expert indices from the gate |
| `num_experts` | scalar | int64 | — | Global expert count (V4-Flash-Base: 128). Padded to `padded_num_experts = ceil(num_experts / 32) * 32` (line 502-503). |
| `block_size` | scalar | int64 | — | M-block size of the downstream MoE GEMM (= `BLOCK_SIZE_M` from the Triton autotune config) |
| `expert_map` (optional) | `[E_global]` | int32 | contiguous | Maps global expert id → local id for EP shards (`-1` ⇒ not on this rank). Passed only when `ignore_invalid_experts=True` (moe_align_block_size.py:97); otherwise applied as a post-pass in Python (line 100-101). |
| `pad_sorted_ids` | bool (Python-only) | — | — | If True, round `max_num_tokens_padded` up to a multiple of `block_size` (moe_align_block_size.py:75-76) |
| `ignore_invalid_experts` | bool (Python-only) | — | — | If True, `expert_map` filters topk_ids during counting; if False, all experts are counted and `expert_map` is applied to the output `expert_ids` post-hoc (moe_align_block_size.py:39-45) |

The Python wrapper computes:
- `max_num_tokens_padded = topk_ids.numel() + num_experts * (block_size - 1)` (line 74), optionally `min(topk_ids.numel() * block_size, ...)` when `topk_ids.numel() < num_experts` (line 77-80).
- `max_num_m_blocks = ceil(max_num_tokens_padded, block_size)`.

## Outputs

| Name | Shape | Dtype | Layout | Meaning |
| --- | --- | --- | --- | --- |
| `sorted_token_ids` | `[max_num_tokens_padded]` | int32 | contiguous | Token-topk-flat indices sorted by their assigned expert and padded to block boundaries. Padding slots contain `numel` (the SENTINEL = `topk_ids.numel()`, line 105). Downstream consumers treat any index `>= num_valid_tokens` as a padding row to skip. |
| `expert_ids` | `[max_num_m_blocks]` | int32 | contiguous | Expert id for each `block_size`-wide M-block of `sorted_token_ids`. Tail blocks (past the last valid block) are filled with `inactive_expert_id = -1` (line 175-180). When `expert_map` is applied in Python post-hoc (line 100-101), `-1` propagates for experts outside the local EP shard. |
| `num_tokens_post_pad` | `[1]` | int32 | — | Total padded token count = sum over experts of `ceil(expert_count, block_size) * block_size`. Used by the downstream GEMM kernel to early-exit M-blocks past the fence (`fused_moe_kernel` line 397). |

Example (from the docstring, moe_align_block_size.py:59-72):
```
topk_ids = [[2,3,4],[1,2,4],[1,3,4],[1,2,3]], block_size=4, num_experts=4
flatten:                    [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3]   # length 12
sorted_token_ids:           [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12]
                            #  └ expert 1 ┘  └ expert 2  ┘ └ expert 3 ┘ └ expert 4 ┘
expert_ids:                 [1, 2, 3, 4]                            # one per block of 4
num_tokens_post_pad:        16
```

## Grid / Block

### Standard path (`topk_ids.numel() >= 1024` OR `num_experts > 64`, line 524)

- Histogram + cumsum + tail-fill kernel: `align_kernel<<<2, 1024, shared_mem_size, stream>>>` (line 560).
  - `blockIdx.x == 0`: warp-partitioned atomicAdd histogram in shared memory (lines 110-139), then `cub::BlockScan` exclusive-sum across `num_experts` (lines 144-160), then fill `expert_ids` with block-replicated expert ids (lines 168-180).
  - `blockIdx.x == 1`: parallel fill of `sorted_token_ids` with `numel` SENTINEL (lines 101-108) — runs concurrently with block 0 because they touch disjoint outputs.
  - `threads = 1024` (must be 1024 for the `cub::BlockScan<int32_t, 1024>` template, see `padded_num_experts < 1024` assert at line 509).
  - Shared mem: `shared_mem_size = num_warps * 32 * sizeof(int32_t)` (line 554-555) for the per-warp histogram counts (`shared_counts`).
- Sort kernel: `sort_kernel<<<(1, actual_blocks), block_threads=256, 0, stream>>>` (line 575-584).
  - `gridDim = (1, min(ceil(numel/256), 65535))`. Each thread atomically increments `cumsum_buffer[expert_id]` to get a rank, then writes `sorted_token_ids[rank] = i` (lines 313-318). This is a parallel bucket sort using the cumsum offsets from the align kernel.

### Small batch / few experts path (`topk_ids.numel() < 1024` AND `num_experts <= 64`, line 524)

- `small_batch_expert_kernel<<<1, fill_threads + threads, shared_mem, stream>>>` (line 539-547).
  - `threads = max(num_experts, 32)`, `fill_threads = 256` (constexpr).
  - Single block: lanes `0..255` fill `sorted_token_ids` with SENTINEL while lanes `256..255+num_experts` build per-thread histograms, do prefix-sum across stride lanes, then write `sorted_token_ids[cumsum[expert] + offset] = i` directly (lines 274-288). No second kernel needed.
  - Shared mem: `(threads+1)*num_experts + (num_experts+1)` int32 (line 530-531).

Per-CTA work breakdown (standard path, block 0):
1. **Histogram (lines 110-139)**: each warp owns `WARP_SIZE = 32` experts (`my_expert_start = warp_id * 32`). Threads stride through `topk_ids[0..numel)` and atomicAdd into `shared_counts[expert_id]` (or skip if `expert_map[expert_id] == -1` when `has_expert_map`).
2. **Round-up + ExclusiveSum (lines 142-164)**: each thread reads its expert's count, rounds up to a multiple of `block_size`, `cub::BlockScan` produces the exclusive prefix sum; result lands in `cumsum[expert_id]` (gmem). Last thread writes `num_tokens_post_pad = cumsum_last + ceil_count_last`.
3. **Fill expert_ids (lines 168-180)**: each thread `i < num_experts` writes `expert_ids[cumsum[i]/block_size .. cumsum[i+1]/block_size]` with its expert id. Then tail-fill with `-1` from `cumsum[num_experts]/block_size` to `max_num_m_blocks`.

## Math

Reference: `deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:633-641` (`MoE.forward`):

```python
weights, indices = self.gate(x, input_ids.flatten())   # indices: [T, top_k]
y = torch.zeros_like(x, dtype=torch.float32)
counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
for i in range(self.experts_start_idx, self.experts_end_idx):
    if counts[i] == 0:
        continue
    expert = self.experts[i]
    idx, top = torch.where(indices == i)                # tokens routed to expert i
    y[idx] += expert(x[idx], weights[idx, top, None])
```

The reference iterates over experts in Python, materializes `idx, top = where(indices == i)`, and feeds the row-subset `x[idx]` to each expert. This kernel produces the same routing but in a GEMM-friendly form: instead of per-expert variable-length lists, it produces a single `sorted_token_ids` array padded to `block_size` boundaries so that the downstream Triton GEMM (`fused_moe_kernel`) can iterate `[BLOCK_SIZE_M, BLOCK_SIZE_N]` tiles uniformly with `expert_ids[pid_m]` picking the per-block expert.

```python
# PyTorch-operator equivalent of the full op:
#
# Inputs:
#   topk_ids:    [T, top_k] int32      (per-token expert assignment, e.g. from fused_topk_bias)
#   num_experts: int                    (global expert count)
#   block_size:  int                    (= BLOCK_SIZE_M of downstream GEMM)
#   expert_map:  [E_global] int32 | None (EP rank-local mapping; -1 = not on this rank)
#
ids_flat = topk_ids.reshape(-1)                                              # [T * top_k]
if expert_map is not None and ignore_invalid_experts:
    valid = expert_map[ids_flat] != -1
    mapped = expert_map[ids_flat]
    counts = torch.bincount(mapped[valid], minlength=num_experts)
else:
    counts = torch.bincount(ids_flat, minlength=num_experts)

# Round-up per-expert counts to block_size boundary:
padded_counts = ((counts + block_size - 1) // block_size) * block_size       # [num_experts]
cumsum = torch.cumsum(torch.cat([torch.tensor([0]), padded_counts]), dim=0)  # [num_experts + 1]
num_tokens_post_pad = cumsum[-1].item()                                      # scalar

# Initialize outputs:
SENTINEL = ids_flat.numel()                                                  # = T * top_k
sorted_token_ids = torch.full((max_num_tokens_padded,), SENTINEL, dtype=torch.int32)
expert_ids = torch.full((max_num_m_blocks,), -1, dtype=torch.int32)          # tail = -1

# Bucket sort (atomicAdd-based, parallel over T*top_k):
per_expert_offset = cumsum.clone()                                           # mutable rank cursor
for i in range(ids_flat.numel()):
    e = ids_flat[i].item()
    if has_expert_map:
        e = expert_map[e].item()
        if e == -1: continue
    r = per_expert_offset[e].item()
    sorted_token_ids[r] = i                                                  # i = token_idx * top_k + k
    per_expert_offset[e] += 1

# Fill expert_ids per block (one entry per block_size-wide chunk):
for e in range(num_experts):
    lo, hi = cumsum[e].item(), cumsum[e+1].item()
    for b in range(lo // block_size, hi // block_size):
        expert_ids[b] = e

# If expert_map applied post-hoc (ignore_invalid_experts=False), remap expert_ids:
if expert_map is not None and not ignore_invalid_experts:
    expert_ids = expert_map[expert_ids]                                       # -1 propagates
```

Notes on fusion / quant:
- **Two-block parallelism (standard path)**: blocks 0 and 1 of the align kernel write disjoint outputs (`expert_ids` + `cumsum` vs `sorted_token_ids`), so launching `gridDim.x = 2` lets the GPU overlap the histogram/scan work with the SENTINEL-fill memset. The check `if blockIdx.x % 2: ... return;` (line 101) is an early-exit for the fill block.
- **CUB BlockScan vs naive prefix sum**: the small-batch path uses a manual O(num_experts) cumsum loop in shared memory (line 248-255) because `num_experts <= 64` makes BlockScan overkill, while the standard path uses `cub::BlockScan<int32_t, 1024>` which requires exactly 1024 threads (hence the `padded_num_experts < 1024` assert at line 509).
- **Two-pass vs one-pass**: the standard path does histogram in pass 1 (`align_kernel`) and bucket-sort in pass 2 (`sort_kernel`) so that the second pass can read the finalized cumsum offsets via gmem. The small-batch path fuses both into a single block by keeping all state in shared memory.
- **`expert_map` two semantics**:
  - `ignore_invalid_experts=True`: passes `expert_map` to CUDA; invalid experts are dropped during counting, never appear in `sorted_token_ids`, and `expert_ids` has no `-1`s except in the tail. Used by `fused_experts_impl` (fused_moe.py:1669).
  - `ignore_invalid_experts=False`: CUDA runs with `has_expert_map=False`; all experts count, and Python applies `expert_ids = expert_map[expert_ids]` post-hoc (moe_align_block_size.py:101). Yields `-1` slots for tokens routed to off-rank experts → triggers `write_zeros_to_output` in the downstream GEMM. Used by some non-fused-experts callers.
- **Why pad to `block_size`?**: the Triton GEMM tiles M-blocks of `BLOCK_SIZE_M` and routes the whole block to a single expert via `expert_ids[pid_m]`. Padding tokens (filled with SENTINEL) within the last block of each expert keep the block aligned; the GEMM kernel ignores them via `token_mask = offs_token < num_valid_tokens` (`fused_moe_kernel` line 412).

## Config-dependent dispatch

- Activation condition: **active when `moe_backend != "deep_gemm_mega_moe"`** — the MegaMoE path does its own dispatch on the GPU via NVLink and does not use this op. Within the FusedMoE path, called for every layer's expert assignment (potentially skipped only by the `naive_block_assignment` fast path at `_prepare_expert_assignment` lines 1443-1463 when `num_tokens * top_k * 4 <= global_num_experts and expert_map is None`).
- Host-side branch (line 524):
  - `small_batch_expert_mode = (topk_ids.numel() < 1024) and (num_experts <= 64)` → single-block kernel.
  - Else → 2-block align + 2-D sort grid.
  V4-Flash-Base has `num_experts = 128 > 64`, so it ALWAYS takes the standard 2-kernel path regardless of batch size.
- Hardware: requires CUDA compute capability supporting `cub::BlockScan` with 1024 threads (SM35+), and atomic add to shared memory (universal on SM50+). No SM100-specific features used.
- Preconditions:
  - `padded_num_experts < 1024` (`TORCH_CHECK` at line 509) — V4-Flash-Base's 128 experts pad to 128, well under the limit.
  - Output tensors allocated by Python wrapper at lines 81-88 with sizes derived from `max_num_tokens_padded` and `max_num_m_blocks`.
  - `topk_ids` integral dtype (dispatched by `VLLM_DISPATCH_INTEGRAL_AND_UNSIGNED_TYPES` at line 521).
- Downstream consumer constraints:
  - `sorted_token_ids` is consumed by `fused_moe_kernel` (line 157) and `fused_moe_kernel_gptq_awq` (line 157) via `tl.load(sorted_token_ids_ptr + offs_token_id)`; SENTINEL values (`= topk_ids.numel()`) are filtered by `token_mask = offs_token < num_valid_tokens`.
  - `expert_ids` is consumed via `tl.load(expert_ids_ptr + pid_m)`; `-1` triggers `write_zeros_to_output` (see `write_zeros_to_output.md`).
  - `num_tokens_post_padded` is loaded once per CTA at the head of the GEMM kernel to gate `if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded: return` (`fused_moe_kernel` line 397).
  - Any spec change must preserve these three output semantics — they are baked into both the FusedMoE Triton kernels and the MarlinExperts / HummingMoE backends.
