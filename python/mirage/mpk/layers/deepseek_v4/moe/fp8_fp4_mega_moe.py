"""V4-Flash ``fp8_fp4_mega_moe`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fp8_fp4_mega_moe.md``.

Decision: **NEW**.

The vLLM kernel is a heavyweight persistent-grid cluster-MMA kernel
that fuses MoE dispatch + L1 GEMM + SwiGLU + L2 GEMM + combine +
allreduce.  Our naive port preserves the per-token output contract
``y: [T, H]`` bf16 but uses bf16 weights and a single CTA per token:
L1 matmul -> SwiGLU (with optional ``swiglu_limit`` clamp) -> L2
matmul -> per-topk combine.  No FP4/FP8 packing, no multicast, no
all-reduce.  FP4/FP8 + cluster-MMA + NVLink is a follow-up; the I/O
contract is preserved at the bf16 level so the surrounding model
code can swap in the fused kernel later.

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/fp8_fp4_mega_moe_v4_sm100.cuh``
* Task name: ``fp8_fp4_mega_moe_v4_sm100``
* Enum slot: ``TASK_FP8_FP4_MEGA_MOE_V4_SM100 = 384``
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ..._base import BlockDim, GridDim, MPKModule
from ....context import current_pk
from .....core import DTensor

__all__ = ["V4Fp8Fp4MegaMoe"]


class V4Fp8Fp4MegaMoe(MPKModule):
    """Naive per-token MoE: L1 GEMM -> SwiGLU -> L2 GEMM -> combine.

    Constructor args:
      * ``num_tokens``        -- T.
      * ``top_k``             -- TOP_K.
      * ``hidden_size``       -- H.
      * ``intermediate_size`` -- I.
      * ``num_experts``       -- E.
      * ``activation_clamp``  -- optional float; passed to the kernel as
                                  milli-units; ``None`` or ``<=0`` disables
                                  the clamp.
    """

    def __init__(
        self,
        num_tokens: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        *,
        activation_clamp: Optional[float] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.num_tokens = num_tokens
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.activation_clamp = activation_clamp

    def forward(
        self,
        a: torch.Tensor,
        w13: torch.Tensor,
        w2: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_w: torch.Tensor,
    ) -> torch.Tensor:
        T = self.num_tokens
        TOP_K = self.top_k
        H = self.hidden_size
        I = self.intermediate_size
        E = self.num_experts

        a_f = a.float()
        w13_f = w13.float()
        w2_f = w2.float()
        clamp = self.activation_clamp

        y = torch.zeros(T, H, dtype=torch.float32, device=a.device)
        for t in range(T):
            for k in range(TOP_K):
                e = int(topk_idx[t, k].item())
                w = float(topk_w[t, k].item())
                if e < 0 or e >= E:
                    continue
                l1 = a_f[t] @ w13_f[e].T  # [2*I]
                gate = l1[:I]
                up = l1[I:2 * I]
                if clamp is not None and clamp > 0:
                    gate = gate.clamp(max=clamp)
                    up = up.clamp(min=-clamp, max=clamp)
                h = F.silu(gate) * up  # [I]
                l2 = h @ w2_f[e].T     # [H]
                y[t] += w * l2
        return y.to(torch.bfloat16)

    def auto_grid_dim(self, a_dt: DTensor) -> GridDim:
        pk = current_pk()
        return (max(1, min(self.num_tokens, pk.num_workers)), 1, 1)

    def compile(
        self,
        a: DTensor,
        w13: DTensor,
        w2: DTensor,
        topk_idx: DTensor,
        topk_w: DTensor,
        y: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        from .....core import CyTBGraph
        from .....kernel import TBGraph

        pk = current_pk()
        grid_dim = grid_dim or self.auto_grid_dim(a)
        block_dim = block_dim or self.default_block_dim()

        clamp = self.activation_clamp or 0.0
        clamp_milli = int(round(max(0.0, clamp) * 1000.0))

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # Partition `a` and `y` on the token axis so each CTA gets its
        # per-token slice via runtime pointer pre-offset.
        tb_graph.new_input(a, (0, -1, -1), 1, True)
        tb_graph.new_input(w13, (-1, -1, -1), -1, True)
        tb_graph.new_input(w2, (-1, -1, -1), -1, True)
        tb_graph.new_input(topk_idx, (0, -1, -1), 1, True)
        tb_graph.new_input(topk_w, (0, -1, -1), 1, True)
        tb_graph.new_input(y, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [a, w13, w2, topk_idx, topk_w, y], tb_graph
        )
        pk.kn_graph.register_task(
            tb_graph,
            "fp8_fp4_mega_moe_v4_sm100",
            [clamp_milli],
        )
        return y
