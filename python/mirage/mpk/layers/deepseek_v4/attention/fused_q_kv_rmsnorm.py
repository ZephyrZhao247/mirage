"""V4-Flash ``fused_q_kv_rmsnorm`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fused_q_kv_rmsnorm.md``.

Joint per-token RMSNorm of Q-LoRA + KV-LoRA streams in one CTA.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor


__all__ = ["V4FusedQKVRMSNorm"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class V4FusedQKVRMSNorm(MPKModule):
    """V4-Flash joint Q-LoRA + KV-LoRA RMSNorm.

    Constructor args:

      * ``q_size``  -- ``q_lora_rank`` (V4-Flash: 1024)
      * ``kv_size`` -- ``head_dim`` (V4-Flash: 512)
      * ``eps``     -- affects ``forward()``; kernel hard-codes
                       ``1e-6f``.
    """

    def __init__(
        self,
        q_size: int = 1024,
        kv_size: int = 512,
        eps: float = 1e-6,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.q_size = q_size
        self.kv_size = kv_size
        self.eps = eps
        self.q_weight = nn.Parameter(torch.ones(q_size))
        self.kv_weight = nn.Parameter(torch.ones(kv_size))

    def forward(
        self,
        qr: torch.Tensor,
        kv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """fp32 reduction, bf16 store."""
        in_dtype = qr.dtype
        qr_f = qr.to(torch.float32)
        kv_f = kv.to(torch.float32)
        qr_var = qr_f.pow(2).mean(-1, keepdim=True)
        kv_var = kv_f.pow(2).mean(-1, keepdim=True)
        qr_out = (qr_f * torch.rsqrt(qr_var + self.eps) *
                  self.q_weight.to(torch.float32)).to(in_dtype)
        kv_out = (kv_f * torch.rsqrt(kv_var + self.eps) *
                  self.kv_weight.to(torch.float32)).to(in_dtype)
        return qr_out, kv_out

    def auto_grid_dim(self, qr_dt: DTensor) -> GridDim:
        """One CTA per token (capped at ``num_workers``)."""
        pk = current_pk()
        return (max(1, min(qr_dt.dim(0), pk.num_workers)), 1, 1)

    def compile(
        self,
        qr: DTensor,
        kv: DTensor,
        *,
        qr_out: Optional[torch.Tensor] = None,
        kv_out: Optional[torch.Tensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register one ``fused_q_kv_rmsnorm_v4_sm100`` task.

        Tensor contract:
          qr        : (T, Q_SIZE)  bf16, row-major contiguous.
          q_weight  : (Q_SIZE,)    bf16 nn.Parameter (auto-attached).
          kv        : (T, KV_SIZE) bf16, row-major contiguous.
          kv_weight : (KV_SIZE,)   bf16 nn.Parameter (auto-attached).
          qr_out    : (T, Q_SIZE)  bf16, allocated if None.
          kv_out    : (T, KV_SIZE) bf16, allocated if None.

        Notes: eps hard-coded to 1e-6f in codegen.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(qr)
        if block_dim is None:
            block_dim = self.default_block_dim()

        T = qr.dim(0)
        qw_dt = pk.attach_input(
            self.q_weight.data, name=f"{self.prefix}q_weight"
        )
        kvw_dt = pk.attach_input(
            self.kv_weight.data, name=f"{self.prefix}kv_weight"
        )

        def _resolve_out(buf, default_shape, name):
            if buf is None:
                return pk.new_tensor(dims=default_shape, dtype=qr.dtype,
                                     name=name)
            if isinstance(buf, torch.Tensor):
                return pk.attach_input(buf, name=name)
            if isinstance(buf, DTensor):
                return buf
            raise TypeError(
                f"{name} must be None, Tensor, or DTensor"
            )

        qr_out_dt = _resolve_out(qr_out, (T, self.q_size),
                                 f"{self.prefix}qr_out")
        kv_out_dt = _resolve_out(kv_out, (T, self.kv_size),
                                 f"{self.prefix}kv_out")

        from .....core import CyTBGraph
        from .....kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(qr, (0, -1, -1), 1, True)
        tb_graph.new_input(qw_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(kv, (0, -1, -1), 1, True)
        tb_graph.new_input(kvw_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(qr_out_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(kv_out_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [qr, qw_dt, kv, kvw_dt, qr_out_dt, kv_out_dt], tb_graph
        )
        pk.kn_graph.register_task(tb_graph, "fused_q_kv_rmsnorm_v4_sm100")
        return qr_out_dt, kv_out_dt
