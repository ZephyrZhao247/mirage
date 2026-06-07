"""V4-Flash ``quantize_and_insert_k_kernel`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/quantize_and_insert_k_kernel.md``.

Decision: **NEW**.

Rationale
---------
The vLLM Triton kernel packs the SWA K cache in a specific 656-byte
per-token layout (448 FP8 NoPE + 128 bf16 RoPE + 8 UE8M0 scale bytes
including 1 padding slot). No existing MPK task writes this exact
layout, and the layout is read verbatim by:
- the FlashMLA sparse-prefill kernel (`flash_mla_sparse_fwd`),
- the FlashMLA decode kernel (`flash_mla_with_kvcache`), and
- the prefill-path dequantizer (`dequantize_and_gather_k_kernel`).

Naive design (correctness only)
-------------------------------
* One CTA per input token (``grid = (num_tokens, 1, 1)``,
  ``block = (256, 1, 1)``).
* Single CTA; threads stripe across the 64 lanes of each quant block.
* fp32 reductions; UE8M0 scale extracted from the IEEE 754 exponent.
* RoPE bf16 lanes copied verbatim as 128 bytes (one bf16-per-lane copy).
* Tokens with ``slot_mapping == -1`` are silently skipped.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor


__all__ = ["V4QuantizeAndInsertK"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]

# Locked structural constants per the V4-Flash SWA cache layout.
CACHE_BLOCK_SIZE = 64
FP8_DIM = 448
BF16_DIM = 64
SCALE_DIM = 8           # 7 real + 1 padding
TOKEN_DATA_SIZE = FP8_DIM + 2 * BF16_DIM   # 576 bytes per token
BLOCK_STRIDE_MIN = CACHE_BLOCK_SIZE * TOKEN_DATA_SIZE + CACHE_BLOCK_SIZE * SCALE_DIM
# 64 * 576 + 64 * 8 = 36864 + 512 = 37376


class V4QuantizeAndInsertK(MPKModule):
    """Naive SWA K-cache writer.

    ``compile()`` registers a single task that, per token:
      * decomposes ``slot_mapping[t]`` into (block_idx, pos_in_block),
      * writes 7 FP8-quantized NoPE blocks (UE8M0 scales) + 1 padding
        scale slot,
      * copies the 64-lane bf16 RoPE verbatim.

    The kernel mutates ``k_cache`` in place; the catalog exposes
    ``k_cache`` as the single output for graph-book-keeping.

    Constructor args:
        block_stride: bytes per paged block. Must be >= 37376
            (64*576 + 64*8). Caller-provided so the layer matches the
            paged allocator's per-block padding.
    """

    def __init__(self, block_stride: int = BLOCK_STRIDE_MIN,
                 *, prefix: str = "") -> None:
        super().__init__(prefix=prefix)
        if block_stride < BLOCK_STRIDE_MIN:
            raise ValueError(
                f"V4QuantizeAndInsertK: block_stride={block_stride} too "
                f"small; need >= {BLOCK_STRIDE_MIN}"
            )
        self.block_stride = int(block_stride)

    # ------------------------------------------------------------------
    # PyTorch reference (eager, fp32 reductions)
    # ------------------------------------------------------------------
    def forward(
        self,
        k: torch.Tensor,
        slot_mapping: torch.Tensor,
        k_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Eager reference matching the vLLM Triton kernel.

        Mutates ``k_cache`` in place and returns it.
        """
        assert k.dim() == 2 and k.shape[1] == FP8_DIM + BF16_DIM, (
            f"k shape: {tuple(k.shape)} (expected [T, 512])"
        )
        assert slot_mapping.shape == (k.shape[0],), (
            f"slot_mapping shape: {tuple(slot_mapping.shape)}"
        )
        assert slot_mapping.dtype in (torch.int64, torch.long), (
            f"slot_mapping dtype: {slot_mapping.dtype}"
        )
        assert k_cache.dtype == torch.uint8, (
            f"k_cache dtype: {k_cache.dtype}"
        )
        assert k_cache.dim() == 2 and k_cache.shape[1] == self.block_stride, (
            f"k_cache shape: {tuple(k_cache.shape)} expected [_, {self.block_stride}]"
        )

        T = k.shape[0]
        FP8_MAX = 448.0
        AMAX_FLOOR = 1e-4

        for t in range(T):
            slot = int(slot_mapping[t].item())
            if slot < 0:
                continue
            block_idx = slot // CACHE_BLOCK_SIZE
            pos_in_block = slot % CACHE_BLOCK_SIZE
            token_data_off = pos_in_block * TOKEN_DATA_SIZE
            token_scale_off = CACHE_BLOCK_SIZE * TOKEN_DATA_SIZE + pos_in_block * SCALE_DIM

            x = k[t].to(torch.float32)

            # FP8 quant of 7 NoPE blocks.
            for qb in range(FP8_DIM // CACHE_BLOCK_SIZE):
                chunk = x[qb * CACHE_BLOCK_SIZE:(qb + 1) * CACHE_BLOCK_SIZE]
                amax = chunk.abs().max().clamp(min=AMAX_FLOOR)
                raw = amax / FP8_MAX
                exponent = int(torch.ceil(torch.log2(raw)).item())
                scale = 2.0 ** exponent
                fp8 = (chunk / scale).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
                fp8_bytes = fp8.view(torch.uint8)
                k_cache[block_idx, token_data_off + qb * CACHE_BLOCK_SIZE:
                        token_data_off + (qb + 1) * CACHE_BLOCK_SIZE] = fp8_bytes

                enc = max(0, min(255, exponent + 127))
                k_cache[block_idx, token_scale_off + qb] = enc

            # Padding scale.
            k_cache[block_idx, token_scale_off + FP8_DIM // CACHE_BLOCK_SIZE] = 0

            # RoPE bf16: 64 lanes = 128 bytes verbatim.
            rope_bf16 = k[t, FP8_DIM:FP8_DIM + BF16_DIM].contiguous()
            rope_bytes = rope_bf16.view(torch.uint8)
            k_cache[block_idx, token_data_off + FP8_DIM:
                    token_data_off + FP8_DIM + 2 * BF16_DIM] = rope_bytes

        return k_cache

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, k_dt: DTensor) -> GridDim:
        pk = current_pk()
        num_tokens = k_dt.dim(0)
        return (max(1, min(num_tokens, pk.num_workers)), 1, 1)

    # ------------------------------------------------------------------
    # MPK registration
    # ------------------------------------------------------------------
    def compile(
        self,
        k: DTensor,
        slot_mapping: DTensor,
        k_cache: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``quantize_and_insert_k_v4_sm100`` task.

        Tensor contract:
          k            : (T, 512) bf16, row-major.
          slot_mapping : (T,)     int64.
          k_cache      : (num_blocks, block_stride) uint8 -- in-place mutated.
        """
        from .....core import CyTBGraph
        from .....kernel import TBGraph

        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(k)
        if block_dim is None:
            block_dim = self.default_block_dim()

        if k.num_dims != 2 or k.dim(1) != FP8_DIM + BF16_DIM:
            raise ValueError(f"k must be [T, 512]; got dims={k.num_dims}, "
                             f"last={k.dim(1) if k.num_dims >= 2 else None}")
        if slot_mapping.num_dims != 1:
            raise ValueError(f"slot_mapping must be 1-D; got {slot_mapping.num_dims}-D")
        if k_cache.num_dims != 2 or k_cache.dim(1) != self.block_stride:
            raise ValueError(
                f"k_cache must be [num_blocks, {self.block_stride}]; got shape "
                f"{(k_cache.dim(0), k_cache.dim(1))}"
            )

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # k partitions on dim 0 (token); broadcast everywhere else.
        tb_graph.new_input(k, (0, -1, -1), 1, True)
        tb_graph.new_input(slot_mapping, (0, -1, -1), 1, True)
        # k_cache broadcast (no partition); the kernel decomposes the slot
        # and writes through the byte buffer.
        tb_graph.new_input(k_cache, (-1, -1, -1), 0, True)
        pk.kn_graph.customized([k, slot_mapping, k_cache], tb_graph)
        pk.kn_graph.register_task(
            tb_graph, "quantize_and_insert_k_v4_sm100",
            [self.block_stride],
        )
        return k_cache
