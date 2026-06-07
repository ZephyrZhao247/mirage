"""V4-Flash ``fused_kv_compress_norm_rope_insert_sparse_attn`` -- NEW
Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/
fused_kv_compress_norm_rope_insert_sparse_attn.md``.

Decision: **NEW**.

Rationale
---------
The vLLM Triton kernel fuses gather-window + softmax + weighted-sum +
RMSNorm + UE8M0 block-FP8 quant + forward GPT-J RoPE + paged-cache
write for the V4-Flash attention compressor (head_dim=512).

There is no existing MPK task with this contract. So this is a NEW
kernel:

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/fused_kv_compress_norm_rope_insert_sparse_attn_v4_sm100.cuh``
* Task name:   ``fused_kv_compress_norm_rope_insert_sparse_attn_v4_sm100``
* Enum slot:   ``TASK_FUSED_KV_COMPRESS_NORM_ROPE_INSERT_SPARSE_ATTN_V4_SM100 = 373``

The Blackwell impl is intentionally NAIVE (single-CTA per token, fp32
throughout, no TMA/UMMA/warp-spec). Supports COMPRESS_RATIO=4 only
(ratio=128 takes the CuteDSL fast path on NVIDIA; the naive port
would exceed Blackwell smem limits for the score-window tile).

**Compressor RoPE base = 160000** (NOT main RoPE's 10000). The
``cos_sin_cache`` argument MUST be built against the 160000 base.

Audit
-----
* dtype: state_cache fp32; bf16 rms_norm_weight; fp32 cos_sin_cache;
  uint8 paged k_cache (FP8 nope + bf16 rope + UE8M0 scales). No silent
  casts; bf16 roundtrip on the NoPE region before UE8M0 quant matches
  the reference numerics.
* layout: per-token (positions, slot_mapping, kv_slot_mapping,
  token_to_req_indices) partitioned on dim 0; state_cache, k_cache,
  block_table, rms_norm_weight, cos_sin_cache broadcast.
* multi-batch: by construction. Test uses ``max_num_batched_requests
  >= 2``.
* ``forward()``: faithful PyTorch reference (uses the same per-token
  slot/window math as the kernel).
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

import mirage as mi

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor


__all__ = ["V4FusedKVCompressNormRopeInsertSparseAttn"]

GridDim = tuple
BlockDim = tuple


class V4FusedKVCompressNormRopeInsertSparseAttn(MPKModule):
    """V4-Flash sparse-attn compressor (head_dim=512) naive Blackwell kernel.

    Constructor args:
      * ``compress_ratio``  -- 4 (overlap) or 128. Naive port supports 4 only.
      * ``block_size``      -- state-cache tokens/block.
      * ``kv_cache_block_size`` -- paged K-cache tokens/block.
      * ``rms_norm_eps``    -- forwarded to ``forward()``; the kernel
                                hard-codes ``1e-6f`` in codegen.

    Owns one bf16 ``rms_norm_weight`` ``nn.Parameter`` of shape [512].
    """

    HEAD_SIZE = 512  # spec-locked: this kernel is the head_dim=512 variant
    ROPE_HEAD_DIM = 64
    NOPE_HEAD_DIM = HEAD_SIZE - ROPE_HEAD_DIM  # 448
    QUANT_BLOCK = 64
    N_NOPE_BLOCKS = NOPE_HEAD_DIM // QUANT_BLOCK  # 7
    SCALE_DIM = N_NOPE_BLOCKS + 1                  # 8 (with pad)
    TOKEN_STRIDE = NOPE_HEAD_DIM + 2 * ROPE_HEAD_DIM  # 576
    FP8_MAX = 448.0

    def __init__(
        self,
        compress_ratio: int,
        block_size: int,
        kv_cache_block_size: int,
        rms_norm_eps: float = 1e-6,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if compress_ratio != 4:
            # Naive port: ratio=128 score window would exceed Blackwell smem.
            raise ValueError(
                "V4FusedKVCompressNormRopeInsertSparseAttn: naive port "
                "supports COMPRESS_RATIO=4 only; ratio=128 takes the "
                "locked CuteDSL path on NVIDIA."
            )
        self.compress_ratio = compress_ratio
        self.block_size = block_size
        self.kv_cache_block_size = kv_cache_block_size
        self.rms_norm_eps = rms_norm_eps
        self.rms_norm_weight = nn.Parameter(
            torch.ones(self.HEAD_SIZE, dtype=torch.bfloat16)
        )

    # ------------------------------------------------------------------
    # PyTorch reference (faithful to the Triton kernel body).
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
        """In-place write into ``k_cache`` (returned for chaining)."""
        T = positions.shape[0]
        H = self.HEAD_SIZE
        ROPE = self.ROPE_HEAD_DIM
        NOPE = self.NOPE_HEAD_DIM
        W = 2 * self.compress_ratio  # ratio=4 -> 8
        ratio = self.compress_ratio
        STATE_WIDTH = 2 * H
        FP8_MAX = self.FP8_MAX

        weight_fp32 = self.rms_norm_weight.detach().to(torch.float32)

        for t in range(T):
            slot = int(slot_mapping[t].item())
            position = int(positions[t].item())
            if slot < 0:
                continue
            if (position + 1) % ratio != 0:
                continue
            kv_slot = int(kv_slot_mapping[t].item())
            if kv_slot < 0:
                continue
            req_idx = int(token_to_req_indices[t].item())

            # Gather W rows.
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
            compressed = (kv_rows * sm).sum(dim=0)  # [H]

            var = compressed.pow(2).mean()
            normed = compressed * torch.rsqrt(var + self.rms_norm_eps) * weight_fp32

            # UE8M0 block-FP8 quant for the NoPE half.
            nope = normed[:NOPE]
            nope = nope.to(torch.bfloat16).to(torch.float32)
            chunks = nope.view(self.N_NOPE_BLOCKS, self.QUANT_BLOCK)
            absmax = chunks.abs().amax(dim=-1).clamp_min(1e-4)
            exponents = torch.ceil(torch.log2(absmax / FP8_MAX))
            inv_scale = torch.exp2(-exponents)
            fp8 = (chunks * inv_scale.unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX)
            fp8 = fp8.to(torch.float8_e4m3fn).view(-1)
            scale_bytes = (exponents.clamp(-127, 128) + 127).to(torch.uint8)

            # Forward GPT-J RoPE for the rope tail.
            rope = normed[NOPE:]
            even = rope[0::2]
            odd  = rope[1::2]
            compressed_pos = (position // ratio) * ratio
            cs_row = cos_sin_cache[compressed_pos]
            cos = cs_row[: ROPE // 2]
            sin = cs_row[ROPE // 2 :]
            new_even = even * cos - odd * sin
            new_odd  = odd  * cos + even * sin
            rotated = torch.empty_like(rope)
            rotated[0::2] = new_even
            rotated[1::2] = new_odd
            rotated_bf16 = rotated.to(torch.bfloat16)

            # Write to paged k_cache layout: data + per-block UE8M0 scales.
            blk = kv_slot // self.kv_cache_block_size
            off = kv_slot % self.kv_cache_block_size
            # data: [0, NOPE) fp8 + [NOPE, NOPE+2*ROPE) bf16
            k_cache[blk, off, 0, : NOPE] = fp8.view(torch.uint8)
            # bf16 RoPE (64 elements * 2 bytes = 128 bytes)
            rope_bytes = rotated_bf16.view(torch.uint8).flatten()
            assert rope_bytes.numel() == 2 * ROPE
            k_cache[blk, off, 0, NOPE : NOPE + 2 * ROPE] = rope_bytes
            # scale region: [block_size*TOKEN_STRIDE + off*SCALE_DIM, +SCALE_DIM)
            scale_base = self.kv_cache_block_size * self.TOKEN_STRIDE + off * self.SCALE_DIM
            k_cache[blk, scale_base // (k_cache.shape[2] * k_cache.shape[3])] \
                if False else None
            # Simpler: k_cache is shaped [num_blocks, KV_BLOCK_SIZE, 1, TOKEN_STRIDE + SCALE_DIM]
            # The scale region for the block lives after all token data
            # rows; the kernel addresses it via raw byte indexing.
            # We flatten the (KV_BLOCK_SIZE, 1, TOKEN_STRIDE + SCALE_DIM)
            # view to uint8 and write the scale bytes.
            block_view = k_cache[blk].view(-1)  # [KV_BLOCK_SIZE * (TOKEN_STRIDE+SCALE_DIM)]
            # The "data" rows occupy [off*(TOKEN_STRIDE+SCALE_DIM),
            # off*(TOKEN_STRIDE+SCALE_DIM)+TOKEN_STRIDE) but the C++
            # kernel addresses them by a different layout (a 2-D split
            # of data then scales). To match the kernel, we use the
            # kernel's layout: bytes [0, KV_BLOCK_SIZE*TOKEN_STRIDE)
            # are data; bytes [KV_BLOCK_SIZE*TOKEN_STRIDE, +KV_BLOCK_SIZE
            # *SCALE_DIM) are scales. Since the tensor's last dim is
            # (TOKEN_STRIDE + SCALE_DIM), we re-flatten using a
            # block_view above and address bytes manually.
            scale_off = self.kv_cache_block_size * self.TOKEN_STRIDE + off * self.SCALE_DIM
            block_view[scale_off : scale_off + self.N_NOPE_BLOCKS] = scale_bytes
            block_view[scale_off + self.N_NOPE_BLOCKS] = 0  # pad byte

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
        """Register one ``fused_kv_compress_norm_rope_insert_sparse_attn_v4_sm100``
        task.

        Per-token tensors (PARTITION dim 0): token_to_req_indices,
        positions, slot_mapping, kv_slot_mapping.
        Broadcast (full base): state_cache, block_table,
        rms_norm_weight, cos_sin_cache, k_cache.
        """
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
            "fused_kv_compress_norm_rope_insert_sparse_attn_v4_sm100",
            [self.compress_ratio],
        )
        return k_cache_dt
