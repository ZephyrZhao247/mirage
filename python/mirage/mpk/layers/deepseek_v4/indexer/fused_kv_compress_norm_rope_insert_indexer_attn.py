"""V4-Flash ``fused_kv_compress_norm_rope_insert_indexer_attn`` (FP8 K-side)
-- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/
fused_kv_compress_norm_rope_insert_indexer_attn.md``.

Decision: **NEW**.

Coupling
--------
Class-B sibling under the ``use_fp4_cache`` flag (FP8 vs MXFP4 paired
with the Q-side variant). This FP8 K-side variant is selected when
``attention_config.use_fp4_indexer_cache=False``; the companion Q-side
kernel is ``fused_indexer_q_rope_quant`` (FP8). Switching to ``True``
requires BOTH this K-side and the FP8 Q-side to be replaced by their
MXFP4 siblings.

Rationale
---------
Per-token indexer compressor (head_dim=128). Same pipeline as the
sparse-attn compressor (gather window + softmax + weighted sum +
RMSNorm + RoPE) but with:

* HEAD_SIZE=128, NOPE=ROPE=64;
* SINGLE quant block (QUANT_BLOCK == HEAD_SIZE) -> one fp32 scale per
  token (NOT a UE8M0 byte; the indexer cache inherits V3.2's fp32
  scale layout);
* the whole head_size vector is FP8-quantized after RoPE.

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/fused_kv_compress_norm_rope_insert_indexer_attn_v4_sm100.cuh``
* Task name:   ``fused_kv_compress_norm_rope_insert_indexer_attn_v4_sm100``
* Enum slot:   ``TASK_FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_ATTN_V4_SM100 = 374``

Compressor RoPE base = ``compress_rope_theta=160000`` (the Q-side
counterpart uses the SAME cos_sin_cache).

Audit
-----
* dtype: state_cache fp32; bf16 rms_norm_weight; fp32 cos_sin_cache;
  uint8 paged k_cache with 128 FP8 bytes + 4 fp32-scale bytes per
  token. Scale is **fp32** (not UE8M0).
* layout: per-token (positions, slot_mapping, kv_slot_mapping,
  token_to_req_indices) partitioned on dim 0; broadcast on cache and
  table.
* multi-batch: by construction. Tests use ``max_num_batched_requests >= 2``.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

import mirage as mi

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor


__all__ = ["V4FusedKVCompressNormRopeInsertIndexerAttn"]

GridDim = tuple
BlockDim = tuple


class V4FusedKVCompressNormRopeInsertIndexerAttn(MPKModule):
    """V4-Flash indexer compressor FP8 K-side naive Blackwell kernel.

    Constructor args:
      * ``block_size``           -- state-cache tokens/block.
      * ``kv_cache_block_size``  -- paged K-cache tokens/block.
      * ``rms_norm_eps``         -- forwarded to ``forward()``; kernel
                                     hard-codes ``1e-6f``.

    Owns one bf16 ``rms_norm_weight`` ``nn.Parameter`` of shape [128].
    """

    HEAD_SIZE = 128
    COMPRESS_RATIO = 4  # spec-locked for indexer
    ROPE_HEAD_DIM = 64
    NOPE_HEAD_DIM = 64
    TOKEN_STRIDE = 128  # FP8 bytes/token in cache
    SCALE_DIM = 4       # 1 fp32 scale = 4 bytes/token
    FP8_MAX = 448.0

    def __init__(
        self,
        block_size: int,
        kv_cache_block_size: int,
        rms_norm_eps: float = 1e-6,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if block_size <= 0:
            raise ValueError(
                f"V4FusedKVCompressNormRopeInsertIndexerAttn: block_size must "
                f"be positive; got {block_size}"
            )
        if kv_cache_block_size <= 0:
            raise ValueError(
                f"V4FusedKVCompressNormRopeInsertIndexerAttn: "
                f"kv_cache_block_size must be positive; got {kv_cache_block_size}"
            )
        self.block_size = block_size
        self.kv_cache_block_size = kv_cache_block_size
        self.rms_norm_eps = rms_norm_eps
        self.rms_norm_weight = nn.Parameter(
            torch.ones(self.HEAD_SIZE, dtype=torch.bfloat16)
        )

    # ------------------------------------------------------------------
    # PyTorch reference.
    # ------------------------------------------------------------------
    def forward(
        self,
        state_cache: torch.Tensor,
        token_to_req_indices: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        block_table: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        kv_slot_mapping: torch.Tensor,
        k_cache: torch.Tensor,
    ) -> torch.Tensor:
        T = positions.shape[0]
        H = self.HEAD_SIZE
        NOPE = self.NOPE_HEAD_DIM
        ROPE = self.ROPE_HEAD_DIM
        ratio = self.COMPRESS_RATIO
        W = 2 * ratio  # OVERLAP=True
        STATE_WIDTH = 2 * H
        FP8_MAX = self.FP8_MAX
        weight_fp32 = self.rms_norm_weight.detach().to(torch.float32)

        for t in range(T):
            slot = int(slot_mapping[t].item())
            position = int(positions[t].item())
            if slot < 0 or (position + 1) % ratio != 0:
                continue
            kv_slot = int(kv_slot_mapping[t].item())
            if kv_slot < 0:
                continue
            req_idx = int(token_to_req_indices[t].item())
            kv_rows = torch.zeros((W, H), dtype=torch.float32,
                                  device=state_cache.device)
            score_rows = torch.full((W, H), float("-inf"),
                                    dtype=torch.float32,
                                    device=state_cache.device)
            for w in range(W):
                pos_w = position - (W - 1) + w
                if pos_w < 0:
                    continue
                head_off = H if w >= ratio else 0
                blk_idx = pos_w // self.block_size
                blk_off = pos_w % self.block_size
                blk_no = int(block_table[req_idx, blk_idx].item())
                row = state_cache[blk_no, blk_off]
                kv_rows[w]    = row[head_off : head_off + H].to(torch.float32)
                score_rows[w] = row[head_off + STATE_WIDTH :
                                    head_off + STATE_WIDTH + H].to(torch.float32)
            sm = torch.softmax(score_rows, dim=0)
            compressed = (kv_rows * sm).sum(dim=0)
            var = compressed.pow(2).mean()
            normed = compressed * torch.rsqrt(var + self.rms_norm_eps) * weight_fp32

            # RoPE on rope tail (window-aligned).
            even = normed[NOPE::2]
            odd  = normed[NOPE + 1::2]
            compressed_pos = (position // ratio) * ratio
            cs_row = cos_sin_cache[compressed_pos]
            cos = cs_row[: ROPE // 2]
            sin = cs_row[ROPE // 2 :]
            new_even = even * cos - odd * sin
            new_odd  = odd  * cos + even * sin
            normed = normed.clone()
            normed[NOPE::2] = new_even
            normed[NOPE + 1::2] = new_odd

            # Single-block FP8 quant. bf16 roundtrip.
            full = normed.to(torch.bfloat16).to(torch.float32)
            absmax = full.abs().amax().clamp_min(1e-4)
            exponent = torch.ceil(torch.log2(absmax / FP8_MAX))
            inv_scale = torch.exp2(-exponent)
            scale_fp32 = torch.exp2(exponent)
            fp8 = (full * inv_scale).clamp(-FP8_MAX, FP8_MAX)
            fp8 = fp8.to(torch.float8_e4m3fn)

            blk = kv_slot // self.kv_cache_block_size
            off = kv_slot % self.kv_cache_block_size
            # data: bytes [0, 128) = FP8 values
            k_cache[blk, off, 0, : H] = fp8.view(torch.uint8)
            # scale region: per-block bytes [block_size*128 + off*4, +4)
            block_view = k_cache[blk].view(-1)  # uint8
            scale_off = self.kv_cache_block_size * self.TOKEN_STRIDE + off * self.SCALE_DIM
            block_view[scale_off : scale_off + 4] = (
                scale_fp32.to(torch.float32).view(torch.uint8).flatten()
            )
        return k_cache

    def auto_grid_dim(self, positions_dt: Any) -> GridDim:
        pk = current_pk()
        num_tokens = positions_dt.dim(0)
        return (max(1, min(num_tokens, int(pk.num_workers))), 1, 1)

    def compile(
        self,
        state_cache_dt: Any,
        token_to_req_indices_dt: Any,
        positions_dt: Any,
        slot_mapping_dt: Any,
        block_table_dt: Any,
        cos_sin_cache_dt: Any,
        kv_slot_mapping_dt: Any,
        k_cache_dt: Any,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        from .....core import CyTBGraph
        from .....kernel import TBGraph

        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(positions_dt)
        if block_dim is None:
            block_dim = self.default_block_dim()

        rms_w_dt = pk.attach_input(
            self.rms_norm_weight.data, name=f"{self.prefix}rms_norm_weight"
        )

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(state_cache_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(token_to_req_indices_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(positions_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(slot_mapping_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(block_table_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(rms_w_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(cos_sin_cache_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(kv_slot_mapping_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(k_cache_dt, (-1, -1, -1), 0, True)
        pk.kn_graph.customized(
            [
                state_cache_dt,
                token_to_req_indices_dt,
                positions_dt,
                slot_mapping_dt,
                block_table_dt,
                rms_w_dt,
                cos_sin_cache_dt,
                kv_slot_mapping_dt,
                k_cache_dt,
            ],
            tb_graph,
        )
        pk.kn_graph.register_task(
            tb_graph,
            "fused_kv_compress_norm_rope_insert_indexer_attn_v4_sm100",
        )
        return k_cache_dt
