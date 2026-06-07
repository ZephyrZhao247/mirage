"""V4-Flash ``fused_moe_kernel`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fused_moe_kernel.md``.

Decision: **NEW**.

The vLLM Triton ``fused_moe_kernel`` runs a per-(M-block, N-block) GEMM
tile against a stacked weight ``[E, N, K]``, dispatched via
``sorted_token_ids`` + ``expert_ids``.  None of the existing MPK MoE
catalog entries (``MoEW13`` / ``MoEW2`` etc.) use this dispatch -- they
take per-expert routing-indices matrices instead.  Hence NEW.

Naive port supports the BF16 W/A path (the simplest correct contract
preserving the spec's `[T, top_k, N]` output layout).  Quantised paths
(FP8/INT8 W8A8, INT8 W8A16) lift to the GPTQ/AWQ sibling kernel.

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/fused_moe_kernel_v4_sm100.cuh``
* Task name: ``fused_moe_kernel_v4_sm100``
* Enum slot: ``TASK_FUSED_MOE_KERNEL_V4_SM100 = 385``
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..._base import BlockDim, GridDim, MPKModule
from ....context import current_pk
from .....core import DTensor

__all__ = ["V4FusedMoeKernel"]


class V4FusedMoeKernel(MPKModule):
    """Per-(pid_m, pid_n) MoE GEMM tile (naive bf16).

    Constructor args:
      * ``num_tokens``    -- T.
      * ``top_k``         -- TOP_K.
      * ``k_dim``         -- inner GEMM dim (H for L1, I for L2).
      * ``n_dim``         -- output dim (2*I for L1, H for L2).
      * ``num_experts``   -- E.
      * ``block_m``       -- BLOCK_SIZE_M.
      * ``block_n``       -- BLOCK_SIZE_N.
      * ``mul_routed_weight`` -- if True, multiply by topk_weights inside
                                  the kernel (used for the L2 launch in
                                  the spec).
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
        mul_routed_weight: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.k_dim = k_dim
        self.n_dim = n_dim
        self.num_experts = num_experts
        self.block_m = block_m
        self.block_n = block_n
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
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_pad: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Faithful reference: per-block GEMM with -1 skip and SENTINEL mask."""
        T = self.num_tokens
        top_k = self.top_k
        N = self.n_dim
        K = self.k_dim
        E = self.num_experts
        bs = self.block_m
        num_valid = T * top_k
        num_post = int(num_tokens_post_pad.item())

        c = torch.zeros(T, top_k, N, dtype=torch.bfloat16, device=a.device)

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
                acc = a[t].float() @ b[e].float().T  # [N]
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
        tb_graph.new_input(sorted_token_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(expert_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(num_tokens_post_pad, (-1, -1, -1), -1, True)
        tb_graph.new_input(topk_weights, (-1, -1, -1), -1, True)
        tb_graph.new_input(c, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [a, b, sorted_token_ids, expert_ids, num_tokens_post_pad,
             topk_weights, c],
            tb_graph,
        )
        pk.kn_graph.register_task(
            tb_graph,
            "fused_moe_kernel_v4_sm100",
            [self.block_m, self.block_n,
             1 if self.mul_routed_weight else 0],
        )
        return c
