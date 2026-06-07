"""V4-Flash ``fp8_fp4_mqa_logits`` -- NEW Blackwell naive kernel (prefill).

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fp8_fp4_mqa_logits.md``.

Decision: **NEW**.

FP8 path only in the naive port. Non-paged: K is contiguous
`[N, head_dim]` with companion `[N]` fp32 per-token scale. Per-Q-row
masking uses `[cu_seqlen_ks[q], cu_seqlen_ke[q])` ranges.

Rationale
---------
Prefill companion of :class:`V4Fp8Fp4PagedMqaLogits`. No existing MPK
kernel matches the contract.

So this is a NEW kernel:

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/fp8_fp4_mqa_logits_v4_sm100.cuh``
* Task name:   ``fp8_fp4_mqa_logits_v4_sm100``
* Enum slot:   ``TASK_FP8_FP4_MQA_LOGITS_V4_SM100 = 379``

Naive design:

* One CTA per Q-row. grid = (T_chunk, 1, 1); block = (256, 1, 1).
* Each CTA loops kv in [0, N), checking the mask `cu_seqlen_ks <= kv <
  cu_seqlen_ke`. clean_logits=False (masked slots untouched).
* No TMA / no UMMA / no warp-spec / no scheduler.

Audit
-----
* dtype: ``q`` fp8, ``k_packed`` fp8, ``k_scales`` fp32, ``weights`` fp32,
  ``cu_seqlen_ks/ke`` int32, ``logits`` fp32.
* layout: contiguous K (no paging).
* multi-batch: by construction (CTAs cover T_chunk).
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn

import mirage as mi

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor

__all__ = ["V4Fp8Fp4MqaLogits"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class V4Fp8Fp4MqaLogits(MPKModule):
    """V4-Flash non-paged MQA logits naive Blackwell kernel (FP8 path).

    Constructor args:

      * ``n_heads``  -- V4-Flash: 64.
      * ``head_dim`` -- V4-Flash: 128.
      * ``n_kv``    -- ``chunk.total_seq_lens``; total KV positions in this
                       chunk. Sets the kernel's kv-loop bound and the
                       logits last-dim.
    """

    def __init__(
        self,
        n_heads: int = 64,
        head_dim: int = 128,
        n_kv: int = 1024,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if n_heads <= 0 or head_dim <= 0 or n_kv <= 0:
            raise ValueError(
                f"V4Fp8Fp4MqaLogits: all sizes must be positive; "
                f"got n_heads={n_heads}, head_dim={head_dim}, n_kv={n_kv}"
            )
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_kv = n_kv

    def forward(
        self,
        q: torch.Tensor,         # fp8 [T_chunk, n_heads, head_dim]
        k_packed: torch.Tensor,  # fp8 [N, head_dim]
        k_scales: torch.Tensor,  # fp32 [N]
        weights: torch.Tensor,   # fp32 [T_chunk, n_heads]
        cu_seqlen_ks: torch.Tensor, # int32 [T_chunk]
        cu_seqlen_ke: torch.Tensor, # int32 [T_chunk]
    ) -> torch.Tensor:
        T = q.shape[0]
        N = k_packed.shape[0]
        device = q.device
        logits = torch.zeros(T, N, dtype=torch.float32, device=device)
        q_fp32 = q.to(torch.float32)            # [T, H, D]
        k_fp32 = k_packed.to(torch.float32) * k_scales.unsqueeze(-1)  # [N, D]
        for qi in range(T):
            ks = int(cu_seqlen_ks[qi].item())
            ke = int(cu_seqlen_ke[qi].item())
            for kv in range(ks, min(ke, N)):
                contrib = (q_fp32[qi] * k_fp32[kv].unsqueeze(0)).sum(dim=-1)  # [H]
                logits[qi, kv] = (weights[qi] * contrib).sum()
        return logits

    def auto_grid_dim(self, q_dt: Any) -> GridDim:
        pk = current_pk()
        T = q_dt.dim(0)
        return (max(1, min(T, int(pk.num_workers))), 1, 1)

    def compile(
        self,
        q_dt: Any,
        k_packed_dt: Any,
        k_scales_dt: Any,
        weights_dt: Any,
        cu_seqlen_ks_dt: Any,
        cu_seqlen_ke_dt: Any,
        *,
        logits_out: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``fp8_fp4_mqa_logits_v4_sm100`` task.

        Tensor contract:
          q_dt            : (T_chunk, n_heads, head_dim) fp8
          k_packed_dt     : (N, head_dim)                fp8
          k_scales_dt     : (N,) or (N, 1)               fp32
          weights_dt      : (T_chunk, n_heads)           fp32
          cu_seqlen_ks_dt : (T_chunk,) or (T_chunk, 1)   int32
          cu_seqlen_ke_dt : (T_chunk,) or (T_chunk, 1)   int32
          logits_out      : (T_chunk, N)                 fp32, alloc if None
        """
        from .....core import CyTBGraph
        from .....kernel import TBGraph

        pk = current_pk()

        if q_dt.num_dims != 3 or q_dt.dim(1) != self.n_heads or q_dt.dim(2) != self.head_dim:
            raise ValueError(
                f"V4Fp8Fp4MqaLogits: q_dt must be 3-D "
                f"(T, n_heads={self.n_heads}, head_dim={self.head_dim}); "
                f"got shape ({q_dt.dim(0)}, {q_dt.dim(1)}, {q_dt.dim(2)})"
            )
        if k_packed_dt.num_dims != 2 or k_packed_dt.dim(1) != self.head_dim:
            raise ValueError(
                f"V4Fp8Fp4MqaLogits: k_packed_dt must be 2-D "
                f"(N, head_dim={self.head_dim})"
            )
        if weights_dt.num_dims != 2 or weights_dt.dim(1) != self.n_heads:
            raise ValueError(
                f"V4Fp8Fp4MqaLogits: weights_dt must be 2-D (T, n_heads)"
            )

        T = q_dt.dim(0)
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(q_dt)
        if block_dim is None:
            block_dim = self.default_block_dim()

        if logits_out is None:
            logits_dt = pk.new_tensor(
                dims=(T, self.n_kv),
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
        # q, weights, cu_seqlen_ks, cu_seqlen_ke, logits: per-row (dim 0).
        # k_packed, k_scales: broadcast.
        tb_graph.new_input(q_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(k_packed_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(k_scales_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(weights_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(cu_seqlen_ks_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(cu_seqlen_ke_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(logits_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [
                q_dt, k_packed_dt, k_scales_dt, weights_dt,
                cu_seqlen_ks_dt, cu_seqlen_ke_dt, logits_dt,
            ],
            tb_graph,
        )
        pk.kn_graph.register_task(
            tb_graph,
            "fp8_fp4_mqa_logits_v4_sm100",
            [
                self.n_heads,
                self.head_dim,
                self.n_kv,
            ],
        )
        return logits_dt
