"""V4-Flash ``fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn``
(MXFP4 K-side) -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/
fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn.md``.

Decision: **NEW**.

Coupling
--------
Class-B sibling under ``use_fp4_cache``. This MXFP4 K-side variant is
selected when ``attention_config.use_fp4_indexer_cache=True``; the
companion Q-side kernel is ``fused_indexer_q_rope_mxfp4``. When
``use_fp4_cache=True``, BOTH this K-side and the MXFP4 Q-side must be
used; mixing with FP8 siblings is forbidden (the DeepGEMM MQA-logits
boundary distinguishes the two paths via the
``(q_values, q_scale)`` tuple).

Rationale
---------
Per-token indexer compressor (head_dim=128). Same compress + RMSNorm
+ RoPE pipeline as the FP8 sibling, but the quant tail is MXFP4:

* per-32-element quant block (HEAD_SIZE / QUANT_BLOCK = 4 blocks);
* per-block UE8M0 byte scale (NOT a single fp32 scale);
* values packed two E2M1 nibbles per byte (64 bytes total).

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn_v4_sm100.cuh``
* Task name:   ``fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn_v4_sm100``
* Enum slot:   ``TASK_FUSED_KV_COMPRESS_NORM_ROPE_INSERT_INDEXER_MXFP4_ATTN_V4_SM100 = 375``

The MXFP4 path uses the same paged-cache *physical* allocation size
as the FP8 path (132 bytes/token), but only 68 bytes are used. The
TBGraph cache tensor matches the FP8 sibling's layout.

Audit
-----
* dtype: state_cache fp32; bf16 rms_norm_weight; fp32 cos_sin_cache.
  Output cache uint8 (MXFP4 nibbles + UE8M0 bytes).
* layout: per-token (positions, slot_mapping, kv_slot_mapping,
  token_to_req_indices) partitioned on dim 0; broadcast on cache and
  table.
* scale format: UE8M0 byte per 32-element block (NOT fp32). This
  matches the spec; downstream consumers (``fp8_fp4_paged_mqa_logits``
  with ``use_fp4_cache=True``) read 4 UE8M0 bytes per token,
  reinterpreted as one int32 lane.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

import mirage as mi

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor


__all__ = ["V4FusedKVCompressNormRopeInsertIndexerMxfp4Attn"]

GridDim = tuple
BlockDim = tuple


def _fp32_to_e2m1(v: float) -> int:
    """Quantize fp32 to a 4-bit E2M1 (sign + 3-bit magnitude code)."""
    sign = 0x8 if v < 0.0 else 0
    a = -v if v < 0.0 else v
    if a < 0.25:
        mag = 0
    elif a < 0.75:
        mag = 1
    elif a < 1.25:
        mag = 2
    elif a < 1.75:
        mag = 3
    elif a < 2.5:
        mag = 4
    elif a < 3.5:
        mag = 5
    elif a < 5.0:
        mag = 6
    else:
        mag = 7
    return sign | mag


class V4FusedKVCompressNormRopeInsertIndexerMxfp4Attn(MPKModule):
    """V4-Flash indexer compressor MXFP4 K-side naive Blackwell kernel.

    Constructor args:
      * ``block_size``           -- state-cache tokens/block.
      * ``kv_cache_block_size``  -- paged K-cache tokens/block.
      * ``rms_norm_eps``         -- forwarded to ``forward()``; kernel
                                     hard-codes ``1e-6f``.

    Owns one bf16 ``rms_norm_weight`` ``nn.Parameter`` of shape [128].
    """

    HEAD_SIZE = 128
    COMPRESS_RATIO = 4
    ROPE_HEAD_DIM = 64
    NOPE_HEAD_DIM = 64
    QUANT_BLOCK = 32                              # MXFP4 block
    N_BLOCKS = HEAD_SIZE // QUANT_BLOCK           # 4
    TOKEN_STRIDE = HEAD_SIZE // 2                 # 64 packed bytes
    SCALE_DIM = N_BLOCKS                          # 4 UE8M0 bytes
    # Same physical cache as FP8 sibling.
    FP8_TOKEN_STRIDE = 128
    FP8_SCALE_DIM = 4
    E2M1_MAX = 6.0

    def __init__(
        self,
        block_size: int,
        kv_cache_block_size: int,
        rms_norm_eps: float = 1e-6,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
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
        W = 2 * ratio
        STATE_WIDTH = 2 * H
        E2M1_MAX = self.E2M1_MAX
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

            # RoPE on rope tail (window-aligned). Stay in pair-split form.
            even_idx = slice(NOPE, None, 2)
            odd_idx  = slice(NOPE + 1, None, 2)
            even = normed[even_idx]
            odd  = normed[odd_idx]
            compressed_pos = (position // ratio) * ratio
            cs_row = cos_sin_cache[compressed_pos]
            cos = cs_row[: ROPE // 2]
            sin = cs_row[ROPE // 2 :]
            new_even = even * cos - odd * sin
            new_odd  = odd  * cos + even * sin
            normed = normed.clone()
            normed[even_idx] = new_even
            normed[odd_idx]  = new_odd

            # bf16 roundtrip.
            full = normed.to(torch.bfloat16).to(torch.float32)

            blk = kv_slot // self.kv_cache_block_size
            off = kv_slot % self.kv_cache_block_size

            scale_bytes = torch.zeros(self.N_BLOCKS, dtype=torch.uint8,
                                      device=state_cache.device)
            packed_bytes = torch.zeros(self.TOKEN_STRIDE, dtype=torch.uint8,
                                       device=state_cache.device)

            for b in range(self.N_BLOCKS):
                base = b * self.QUANT_BLOCK
                chunk = full[base : base + self.QUANT_BLOCK]
                amax = chunk.abs().amax().item()
                amax = max(amax, E2M1_MAX * 2 ** -126)
                log2_ratio = max(min(
                    float(torch.ceil(torch.log2(torch.tensor(amax) / E2M1_MAX)).item()),
                    127.0), -127.0)
                inv_scale = 2.0 ** (-log2_ratio)
                byte = int(log2_ratio) + 127
                if byte < 0: byte = 0
                if byte > 254: byte = 254
                scale_bytes[b] = byte
                # Pack 32 nibbles into 16 bytes:
                # byte i in [0, 16): low = E2M1(chunk[2i]*inv), high = E2M1(chunk[2i+1]*inv).
                pack_base = base // 2
                for i in range(self.QUANT_BLOCK // 2):
                    lo = _fp32_to_e2m1(float(chunk[2 * i].item()) * inv_scale)
                    hi = _fp32_to_e2m1(float(chunk[2 * i + 1].item()) * inv_scale)
                    packed_bytes[pack_base + i] = (hi << 4) | (lo & 0x0F)

            # Write into the physical (FP8-sized) cache. Only the first
            # 64 bytes of the 128-byte data slot are used; the remaining
            # 64 are padding.
            k_cache[blk, off, 0, : self.TOKEN_STRIDE] = packed_bytes
            block_view = k_cache[blk].view(-1)  # uint8
            scale_off = self.kv_cache_block_size * self.FP8_TOKEN_STRIDE + \
                        off * self.FP8_SCALE_DIM
            block_view[scale_off : scale_off + self.SCALE_DIM] = scale_bytes

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
            "fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn_v4_sm100",
        )
        return k_cache_dt
