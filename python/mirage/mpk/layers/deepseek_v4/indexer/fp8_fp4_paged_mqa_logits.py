"""V4-Flash ``fp8_fp4_paged_mqa_logits`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fp8_fp4_paged_mqa_logits.md``.

Decision: **NEW**.

FP8 path only in the naive port (matches `use_fp4_cache=False`). The
MXFP4 dispatch in the upstream DeepGEMM wrapper would require block-
scaled MMA dequant; we leave that as a follow-up. The MPK contract is
straightforward: produce per-(query atom, kv_pos) MQA logits.

Rationale
---------
Per-(q, kv_pos) logit:

    logits[q, kv_pos] = sum_h weights[q, h] * (q_fp32[h] . k_fp32(kv_pos))

with paged-KV addressing via ``block_tables``. No existing MPK kernel
matches this contract.

So this is a NEW kernel:

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/fp8_fp4_paged_mqa_logits_v4_sm100.cuh``
* Task name:   ``fp8_fp4_paged_mqa_logits_v4_sm100``
* Enum slot:   ``TASK_FP8_FP4_PAGED_MQA_LOGITS_V4_SM100 = 378``

Naive design:

* One CTA per query atom (B * NEXT_N). grid = (B*NEXT_N, 1, 1).
  block = (256, 1, 1).
* Each CTA loops kv_pos in [0, MAX_MODEL_LEN), checking the mask
  `kv_pos < context_len`. clean_logits=False (masked slots untouched).
* No TMA / no UMMA / no warp-spec / no scheduler.

Audit
-----
* dtype: ``q`` fp8 e4m3, ``kv_cache`` uint8, ``weights`` fp32,
  ``block_table`` int32, ``context_lens`` int32, ``logits`` fp32.
  Per-token K scale is fp32 (matches FP8 sibling).
* layout: paged KV cache `[num_blocks, BLOCK_SIZE, 1, KV_HEAD_WIDTH]`
  with KV_HEAD_WIDTH = HEAD_DIM + 4 (fp8 128B + fp32 scale 4B).
* multi-batch: by construction (CTAs cover B*NEXT_N).
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

import mirage as mi

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor

__all__ = ["V4Fp8Fp4PagedMqaLogits"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class V4Fp8Fp4PagedMqaLogits(MPKModule):
    """V4-Flash paged-MQA logits naive Blackwell kernel (FP8 path).

    Constructor args:

      * ``n_heads``       -- V4-Flash: 64 (`index_n_heads`).
      * ``head_dim``      -- V4-Flash: 128 (`index_head_dim`).
      * ``block_size``    -- paged KV cache block size (V4-Flash typical: 64).
      * ``kv_head_width`` -- per-token KV bytes (HEAD_DIM + 4).
      * ``max_model_len`` -- maximum kv_pos (output last-dim size).
    """

    def __init__(
        self,
        n_heads: int = 64,
        head_dim: int = 128,
        block_size: int = 64,
        kv_head_width: Optional[int] = None,
        max_model_len: int = 1024,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if n_heads <= 0 or head_dim <= 0 or block_size <= 0 or max_model_len <= 0:
            raise ValueError(
                f"V4Fp8Fp4PagedMqaLogits: all sizes must be positive; "
                f"got n_heads={n_heads}, head_dim={head_dim}, "
                f"block_size={block_size}, max_model_len={max_model_len}"
            )
        if kv_head_width is None:
            kv_head_width = head_dim + 4
        if kv_head_width < head_dim + 4:
            raise ValueError(
                f"V4Fp8Fp4PagedMqaLogits: kv_head_width ({kv_head_width}) "
                f"must accommodate head_dim ({head_dim}) + 4 fp32-scale bytes"
            )
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.kv_head_width = kv_head_width
        self.max_model_len = max_model_len

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,            # fp8  [Q, n_heads, head_dim] (Q = B*NEXT_N)
        kv_cache: torch.Tensor,     # uint8 [num_blocks, block_size, 1, kv_head_width]
        weights: torch.Tensor,      # fp32 [Q, n_heads]
        block_tables: torch.Tensor, # int32 [Q, max_blocks]
        context_lens: torch.Tensor, # int32 [Q]
    ) -> torch.Tensor:
        """Eager reference matching the kernel I/O contract."""
        Q = q.shape[0]
        device = q.device
        logits = torch.zeros(
            Q, self.max_model_len, dtype=torch.float32, device=device
        )
        q_fp32 = q.to(torch.float32)            # [Q, H, D]

        for qi in range(Q):
            ctx = int(context_lens[qi].item())
            for kv_pos in range(min(ctx, self.max_model_len)):
                block_logical = kv_pos // self.block_size
                slot = kv_pos % self.block_size
                block_phys = int(block_tables[qi, block_logical].item())
                row_bytes = kv_cache[block_phys, slot, 0]  # [kv_head_width] uint8
                # K bytes [0:head_dim] = fp8; bytes [head_dim:head_dim+4] = fp32 scale.
                k_fp8 = row_bytes[: self.head_dim].view(torch.float8_e4m3fn)
                k_scale = row_bytes[self.head_dim : self.head_dim + 4].view(torch.float32)[0]
                k_fp32 = k_fp8.to(torch.float32) * k_scale
                # MQA logit.
                contrib = (q_fp32[qi] * k_fp32.unsqueeze(0)).sum(dim=-1)  # [H]
                logits[qi, kv_pos] = (weights[qi] * contrib).sum()
        return logits

    def auto_grid_dim(self, q_dt: Any) -> GridDim:
        pk = current_pk()
        Q = q_dt.dim(0)
        return (max(1, min(Q, int(pk.num_workers))), 1, 1)

    def compile(
        self,
        q_dt: Any,
        kv_cache_dt: Any,
        weights_dt: Any,
        block_tables_dt: Any,
        context_lens_dt: Any,
        *,
        logits_out: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``fp8_fp4_paged_mqa_logits_v4_sm100`` task.

        Tensor contract:
          q_dt            : (Q, n_heads, head_dim)         fp8
          kv_cache_dt     : (num_blocks, block_size, 1, kv_head_width) uint8
          weights_dt      : (Q, n_heads)                   fp32
          block_tables_dt : (Q, max_blocks)                int32
          context_lens_dt : (Q,) or (Q, 1)                 int32
          logits_out      : (Q, max_model_len)             fp32, alloc if None
        """
        from .....core import CyTBGraph
        from .....kernel import TBGraph

        pk = current_pk()

        if q_dt.num_dims != 3 or q_dt.dim(1) != self.n_heads or q_dt.dim(2) != self.head_dim:
            raise ValueError(
                f"V4Fp8Fp4PagedMqaLogits: q_dt must be 3-D "
                f"(Q, n_heads={self.n_heads}, head_dim={self.head_dim}); "
                f"got shape ({q_dt.dim(0)}, {q_dt.dim(1)}, {q_dt.dim(2)})"
            )
        if kv_cache_dt.num_dims != 4 or kv_cache_dt.dim(3) != self.kv_head_width:
            raise ValueError(
                f"V4Fp8Fp4PagedMqaLogits: kv_cache_dt must be 4-D with "
                f"dim(3) == kv_head_width={self.kv_head_width}"
            )
        if weights_dt.num_dims != 2 or weights_dt.dim(1) != self.n_heads:
            raise ValueError(
                f"V4Fp8Fp4PagedMqaLogits: weights_dt must be 2-D "
                f"(Q, n_heads={self.n_heads})"
            )
        if block_tables_dt.num_dims != 2:
            raise ValueError(
                f"V4Fp8Fp4PagedMqaLogits: block_tables_dt must be 2-D "
                f"(Q, max_blocks)"
            )

        Q = q_dt.dim(0)
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(q_dt)
        if block_dim is None:
            block_dim = self.default_block_dim()

        if logits_out is None:
            logits_dt = pk.new_tensor(
                dims=(Q, self.max_model_len),
                dtype=mi.float32,
                name=f"{self.prefix}logits",
            )
        elif isinstance(logits_out, torch.Tensor):
            logits_dt = pk.attach_input(
                logits_out, name=f"{self.prefix}logits"
            )
        else:
            logits_dt = logits_out

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # q, weights, block_tables, context_lens, logits: per-row (dim 0).
        # kv_cache: broadcast (kernel does the block-table lookup).
        tb_graph.new_input(q_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(kv_cache_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(weights_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(block_tables_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(context_lens_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(logits_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [
                q_dt, kv_cache_dt, weights_dt,
                block_tables_dt, context_lens_dt, logits_dt,
            ],
            tb_graph,
        )
        pk.kn_graph.register_task(
            tb_graph,
            "fp8_fp4_paged_mqa_logits_v4_sm100",
            [
                self.n_heads,
                self.head_dim,
                self.block_size,
                self.kv_head_width,
                self.max_model_len,
            ],
        )
        return logits_dt
