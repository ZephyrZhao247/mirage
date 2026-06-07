"""V4-Flash ``dequantize_and_gather_k_kernel`` -- NEW Blackwell naive kernel.

Spec (this implementation tracks the **Triton** variant per D2):
``docs/mpk/deepseek_v4/vllm_kernels/dequantize_and_gather_k_kernel.md``.

Locked alternative (NOT spec'd here): the CuteDSL sibling
``DequantGatherKCacheKernel`` at
``vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py``.
The vLLM runtime dispatcher picks CuteDSL when available; MPK locks the
Triton math as the reference.

Decision: **NEW**.

Rationale
---------
Inverse of :class:`V4QuantizeAndInsertK`. No existing MPK task reads the
656-byte packed K cache layout (448 FP8 + 128 bf16 + 8 UE8M0 scales);
this is the dequantizer for the prefill `flash_mla_sparse_fwd` path.

Naive design
------------
* One CTA per request (``grid = (num_reqs, 1, 1)``).
* Serial loop over the per-request gather range; 256-thread CTA strides
  across the 64-lane FP8 / bf16 chunks.
* Optional ``gather_lens``: when omitted (passed as a flag), gather all
  ``seq_len`` tokens (compressed-K call site); when provided, gather
  the last ``gather_len[i]`` tokens (SWA-K call site).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor


__all__ = ["V4DequantizeAndGatherK"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]

FP8_DIM = 448
BF16_DIM = 64
SCALE_DIM = 8
TOKEN_DATA_SIZE = FP8_DIM + 2 * BF16_DIM   # 576 bytes
OUTPUT_DIM = 512   # lanes the kernel writes (448 dequant + 64 bf16 copy)
HEAD_SIZE = 576    # but the `out` tensor has last-dim 576 (caller-provided)


class V4DequantizeAndGatherK(MPKModule):
    """Naive paged-K dequantize + gather.

    Constructor args:
        block_stride: bytes per paged block. Must match the producer.
        cache_block_size: tokens per paged block (64 for SWA, 256 for
            compressed). Same kernel handles both.
    """

    def __init__(
        self,
        block_stride: int,
        cache_block_size: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if cache_block_size not in (64, 256):
            raise ValueError(
                f"cache_block_size must be 64 (SWA) or 256 (compressed); "
                f"got {cache_block_size}"
            )
        self.block_stride = int(block_stride)
        self.cache_block_size = int(cache_block_size)

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        out: torch.Tensor,
        k_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        block_table: torch.Tensor,
        gather_lens: Optional[torch.Tensor],
        *,
        offset: int = 0,
    ) -> torch.Tensor:
        """Eager reference matching the Triton kernel.

        Writes into ``out[:, offset:offset+gather_len, :512]`` per batch.
        Lanes ``[512, 576)`` are left untouched by the kernel.
        """
        assert out.dim() == 3 and out.shape[2] >= OUTPUT_DIM
        assert k_cache.dtype == torch.uint8 and k_cache.dim() == 2
        assert seq_lens.dtype == torch.int32
        assert block_table.dtype == torch.int32 and block_table.dim() == 2

        num_reqs = out.shape[0]
        for b in range(num_reqs):
            seq_len = int(seq_lens[b].item())
            gather_len = (
                int(gather_lens[b].item()) if gather_lens is not None else seq_len
            )
            start_pos = seq_len - gather_len

            for i in range(gather_len):
                pos = start_pos + i
                block_in_seq = pos // self.cache_block_size
                pos_in_block = pos % self.cache_block_size
                phys = int(block_table[b, block_in_seq].item())

                token_data_off = pos_in_block * TOKEN_DATA_SIZE
                token_scale_off = self.cache_block_size * TOKEN_DATA_SIZE + pos_in_block * SCALE_DIM

                # Dequant 7 FP8 blocks of 64 lanes.
                for qb in range(7):
                    enc = int(k_cache[phys, token_scale_off + qb].item())
                    scale = 2.0 ** (enc - 127)
                    fp8_bytes = k_cache[phys, token_data_off + qb * 64:
                                        token_data_off + (qb + 1) * 64]
                    fp8 = fp8_bytes.view(torch.float8_e4m3fn)
                    out[b, offset + i, qb * 64:(qb + 1) * 64] = (
                        fp8.to(torch.float32) * scale
                    ).to(out.dtype)

                # Copy bf16 RoPE.
                rope_bytes = k_cache[phys, token_data_off + FP8_DIM:
                                     token_data_off + FP8_DIM + 2 * BF16_DIM]
                rope_bf16 = rope_bytes.contiguous().view(torch.bfloat16)
                out[b, offset + i, FP8_DIM:FP8_DIM + BF16_DIM] = rope_bf16.to(out.dtype)

        return out

    # ------------------------------------------------------------------
    def auto_grid_dim(self, out_dt: DTensor) -> GridDim:
        pk = current_pk()
        num_reqs = out_dt.dim(0)
        return (max(1, min(num_reqs, pk.num_workers)), 1, 1)

    # ------------------------------------------------------------------
    def compile(
        self,
        k_cache: DTensor,
        seq_lens: DTensor,
        gather_lens: Optional[DTensor],
        block_table: DTensor,
        out: DTensor,
        *,
        offset: int = 0,
        max_blocks_per_seq: Optional[int] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``dequantize_and_gather_k_v4_sm100`` task.

        Tensor contract:
          k_cache       : (num_blocks, block_stride) uint8.
          seq_lens      : (num_reqs,)                int32.
          gather_lens   : (num_reqs,)                int32 or None.
          block_table   : (num_reqs, max_blocks)     int32.
          out           : (num_reqs, M, 576)         bf16 -- in-place written.
        """
        from .....core import CyTBGraph
        from .....kernel import TBGraph

        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(out)
        if block_dim is None:
            block_dim = self.default_block_dim()

        if max_blocks_per_seq is None:
            max_blocks_per_seq = block_table.dim(1)

        # `out`'s per-token byte stride is HEAD_SIZE * 2 (bf16); the
        # codegen receives the byte stride to convert positions in
        # tokens -> raw byte offsets.
        out_stride1_bytes = out.dim(2) * 2  # bf16 = 2 bytes

        use_gather_lens = 1 if gather_lens is not None else 0
        if gather_lens is None:
            # The TBGraph still needs *some* DTensor in slot 2 to keep the
            # operator count = 5. Reuse seq_lens as a harmless placeholder;
            # the kernel ignores input 2 because use_gather_lens == 0.
            gather_lens_dt = seq_lens
        else:
            gather_lens_dt = gather_lens

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # k_cache broadcast (no partition).
        tb_graph.new_input(k_cache, (-1, -1, -1), 0, True)
        # Per-request scalars / rows partition on dim 0.
        tb_graph.new_input(seq_lens, (0, -1, -1), 1, True)
        tb_graph.new_input(gather_lens_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(block_table, (0, -1, -1), 1, True)
        # out partitions on dim 0 (batch).
        tb_graph.new_input(out, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [k_cache, seq_lens, gather_lens_dt, block_table, out], tb_graph
        )
        pk.kn_graph.register_task(
            tb_graph, "dequantize_and_gather_k_v4_sm100",
            [
                out_stride1_bytes,
                self.block_stride,
                self.cache_block_size,
                int(offset),
                int(max_blocks_per_seq),
                use_gather_lens,
            ],
        )
        return out
