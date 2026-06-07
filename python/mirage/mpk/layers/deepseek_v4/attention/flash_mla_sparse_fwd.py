"""V4-Flash ``flash_mla_sparse_fwd`` (sparse prefill) -- NEW Blackwell
naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/flash_mla_sparse_fwd.md``.

KV is bf16 pre-gathered upstream; one CTA per (token, head) does
online-softmax sparse attention.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor


__all__ = ["V4FlashMLASparsePrefill"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class V4FlashMLASparsePrefill(MPKModule):
    """V4-Flash sparse MLA prefill (one task per (token, head))."""

    def __init__(
        self,
        num_heads_q: int = 64,
        head_dim: int = 576,
        head_v: int = 512,
        softmax_scale: Optional[float] = None,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if head_dim != 576 or head_v != 512:
            raise ValueError(
                "V4FlashMLASparsePrefill locks head_dim=576, head_v=512"
            )
        if num_heads_q not in (64, 128):
            raise ValueError(
                f"num_heads_q must be in {{64, 128}}; got {num_heads_q}"
            )
        self.num_heads_q = num_heads_q
        self.head_dim = head_dim
        self.head_v = head_v
        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(head_dim)
        self.softmax_scale = float(softmax_scale)

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        indices: torch.Tensor,
        topk_length: torch.Tensor,
        attn_sink: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Eager naive sparse attention reference."""
        s_q, h, _ = q.shape
        s_kv = kv.shape[0]
        out = torch.zeros(s_q, h, self.head_v, dtype=q.dtype, device=q.device)
        q_f = q.to(torch.float32)
        kv_f = kv.to(torch.float32)
        for t in range(s_q):
            L = int(topk_length[t].item())
            if L <= 0:
                continue
            idx = indices[t, :L]
            mask = (idx >= 0) & (idx < s_kv)
            if not mask.any():
                continue
            idx_clamped = idx.clamp(min=0, max=s_kv - 1)
            K = kv_f[idx_clamped]
            for hi in range(h):
                logits = (q_f[t, hi] @ K.T) * self.softmax_scale
                logits = torch.where(mask, logits,
                                     torch.full_like(logits, -1e9))
                lse = torch.logsumexp(logits, dim=-1)
                p = torch.softmax(logits, dim=-1)
                V = K[:, :self.head_v]
                out_th = p @ V
                if attn_sink is not None:
                    sink = attn_sink[hi]
                    out_th = out_th * (
                        torch.exp(lse) / (torch.exp(lse) + torch.exp(sink))
                    )
                out[t, hi] = out_th.to(q.dtype)
        return out

    def auto_grid_dim(self, q_dt: DTensor) -> GridDim:
        """One CTA per (token, head)."""
        return (q_dt.dim(0), self.num_heads_q, 1)

    def compile(
        self,
        q: DTensor,
        kv: DTensor,
        indices: DTensor,
        topk_length: DTensor,
        attn_sink: DTensor,
        *,
        head_id: int = 0,
        out: Optional[torch.Tensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``flash_mla_sparse_prefill_v4_sm100`` task per head.

        Tensor contract:
          q            : (s_q, NUM_HEADS_Q, 576) bf16.
          kv           : (s_kv, 576) bf16 broadcast.
          indices      : (s_q, topk) int32.
          topk_length  : (s_q,) int32.
          attn_sink    : (NUM_HEADS_Q,) fp32.
          out          : (s_q, NUM_HEADS_Q, 512) bf16.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim(q)
        if block_dim is None:
            block_dim = self.default_block_dim()

        s_q = q.dim(0)
        s_kv = kv.dim(0)
        if out is None:
            out_dt = pk.new_tensor(
                dims=(s_q, self.num_heads_q, self.head_v),
                dtype=q.dtype,
                name=f"{self.prefix}flash_mla_prefill_out",
            )
        elif isinstance(out, torch.Tensor):
            out_dt = pk.attach_input(out,
                                     name=f"{self.prefix}flash_mla_prefill_out")
        elif isinstance(out, DTensor):
            out_dt = out
        else:
            raise TypeError("out must be None, Tensor, or DTensor")

        from .....core import CyTBGraph
        from .....kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q, (0, -1, -1), 1, True)
        tb_graph.new_input(kv, (-1, -1, -1), 0, True)
        tb_graph.new_input(indices, (0, -1, -1), 1, True)
        tb_graph.new_input(topk_length, (0, -1, -1), 1, True)
        tb_graph.new_input(attn_sink, (-1, -1, -1), 0, True)
        tb_graph.new_input(out_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [q, kv, indices, topk_length, attn_sink, out_dt],
            tb_graph,
        )
        softmax_scale_x1e6 = int(round(self.softmax_scale * 1.0e6))
        pk.kn_graph.register_task(
            tb_graph,
            "flash_mla_sparse_prefill_v4_sm100",
            [s_kv, head_id, softmax_scale_x1e6],
        )
        return out_dt
