"""V4-Flash ``fused_moe_kernel_gptq_awq`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fused_moe_kernel_gptq_awq.md``.

Decision: **NEW**.

Same dispatch as ``fused_moe_kernel`` but with on-the-fly INT8 W8A16
dequant (B is int8, scales are fp32 per K-group; zero-points are int8
or implicit 128).  Naive port handles the W8A16 path; W4A16 (nibble
packing) is not exposed in this version -- callers requesting W4A16
should construct an upstream INT8 stand-in or fall back to the
bf16 ``fused_moe_kernel`` after host-side dequant.

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/fused_moe_kernel_gptq_awq_v4_sm100.cuh``
* Task name: ``fused_moe_kernel_gptq_awq_v4_sm100``
* Enum slot: ``TASK_FUSED_MOE_KERNEL_GPTQ_AWQ_V4_SM100 = 386``
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..._base import BlockDim, GridDim, MPKModule
from ....context import current_pk
from .....core import DTensor

__all__ = ["V4FusedMoeKernelGptqAwq"]


class V4FusedMoeKernelGptqAwq(MPKModule):
    """Per-(pid_m, pid_n) MoE GEMM tile with INT8 W8A16 dequant.

    Constructor args:
      * ``num_tokens``    -- T.
      * ``top_k``         -- TOP_K.
      * ``k_dim``         -- K.
      * ``n_dim``         -- N.
      * ``num_experts``   -- E.
      * ``block_m``       -- BLOCK_SIZE_M.
      * ``block_n``       -- BLOCK_SIZE_N.
      * ``group_size``    -- K-axis scale group size (GROUP_SIZE).
      * ``has_zp``        -- if False the kernel uses the symmetric
                              default zp = 128 (W8 midpoint).
      * ``mul_routed_weight`` -- if True multiply by topk_weights.
    """

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        k_dim: int,
        n_dim: int,
        num_experts: int,
        *,
        block_m: int,
        block_n: int,
        group_size: int,
        has_zp: bool = False,
        mul_routed_weight: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if k_dim % group_size != 0:
            raise ValueError(
                f"V4FusedMoeKernelGptqAwq: K must be divisible by group_size; "
                f"got K={k_dim}, group_size={group_size}."
            )
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.k_dim = k_dim
        self.n_dim = n_dim
        self.num_experts = num_experts
        self.block_m = block_m
        self.block_n = block_n
        self.group_size = group_size
        self.num_k_groups = k_dim // group_size
        self.has_zp = bool(has_zp)
        self.mul_routed_weight = bool(mul_routed_weight)
        em = num_tokens * top_k + num_experts * (block_m - 1)
        em = ((em + block_m - 1) // block_m) * block_m
        self.em = em
        self.num_m_blocks = em // block_m
        self.num_n_blocks = (n_dim + block_n - 1) // block_n

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        b_scale: torch.Tensor,
        b_zp: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_pad: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        T = self.num_tokens
        top_k = self.top_k
        N = self.n_dim
        K = self.k_dim
        E = self.num_experts
        bs = self.block_m
        gs = self.group_size
        num_valid = T * top_k
        num_post = int(num_tokens_post_pad.item())

        c = torch.zeros(T, top_k, N, dtype=torch.bfloat16, device=a.device)
        # Dequant B element by element (reference): bq is int8, scale fp32,
        # zp int8 (or default 128).
        b_f = b.to(torch.float32)
        b_s = b_scale.to(torch.float32)
        if self.has_zp:
            zp_f = b_zp.to(torch.float32)
        else:
            zp_f = torch.full_like(b_s, 128.0)
        # Broadcast scales over K group: shape [E, N, K]
        zp_b = zp_f.repeat_interleave(gs, dim=-1)
        sc_b = b_s.repeat_interleave(gs, dim=-1)
        b_dq = (b_f - zp_b) * sc_b   # fp32 [E, N, K]

        for pid_m in range(self.num_m_blocks):
            if pid_m * bs >= num_post:
                break
            e = int(expert_ids[pid_m].item())
            if e < 0 or e >= E:
                continue
            ot_block = sorted_token_ids[pid_m * bs:(pid_m + 1) * bs]
            for m in range(min(bs, ot_block.numel())):
                ot = int(ot_block[m].item())
                if ot >= num_valid:
                    continue
                t = ot // top_k
                k_slot = ot % top_k
                acc = a[t].float() @ b_dq[e].T
                if self.mul_routed_weight:
                    acc = acc * topk_weights.flatten()[ot].float()
                c[t, k_slot] = acc.to(torch.bfloat16)
        return c

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        pk = current_pk()
        gx = self.num_m_blocks
        gy = self.num_n_blocks
        total = gx * gy
        if total > pk.num_workers:
            gy = max(1, min(gy, pk.num_workers // max(gx, 1)))
        return (max(1, gx), max(1, gy), 1)

    def compile(
        self,
        a: DTensor,
        b: DTensor,
        b_scale: DTensor,
        b_zp: DTensor,
        sorted_token_ids: DTensor,
        expert_ids: DTensor,
        num_tokens_post_pad: DTensor,
        topk_weights: DTensor,
        c: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        from .....core import CyTBGraph
        from .....kernel import TBGraph

        pk = current_pk()
        grid_dim = grid_dim or self.auto_grid_dim(a)
        block_dim = block_dim or self.default_block_dim()

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(a, (-1, -1, -1), -1, True)
        tb_graph.new_input(b, (-1, -1, -1), -1, True)
        tb_graph.new_input(b_scale, (-1, -1, -1), -1, True)
        tb_graph.new_input(b_zp, (-1, -1, -1), -1, True)
        tb_graph.new_input(sorted_token_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(expert_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(num_tokens_post_pad, (-1, -1, -1), -1, True)
        tb_graph.new_input(topk_weights, (-1, -1, -1), -1, True)
        tb_graph.new_input(c, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [a, b, b_scale, b_zp, sorted_token_ids, expert_ids,
             num_tokens_post_pad, topk_weights, c],
            tb_graph,
        )
        pk.kn_graph.register_task(
            tb_graph,
            "fused_moe_kernel_gptq_awq_v4_sm100",
            [self.block_m, self.block_n, self.group_size,
             1 if self.has_zp else 0,
             1 if self.mul_routed_weight else 0],
        )
        return c
