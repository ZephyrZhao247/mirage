"""V4-Flash ``dsv3_router_gemm`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/dsv3_router_gemm.md``.

Decision: **NEW** (Tier-3 / Tier-4 plain matmul).

Rationale
---------
The vLLM ``GateLinear.forward`` dispatch picks ONE of FOUR tiers based on
shape / dtype:

* **Tier 1** -- ``dsv3_router_gemm`` (CUDA, DSV3-specialized): H=7168,
  E in {256, 384}, M<=16. DeepSeek V3 / Kimi K2 only. NOT V4-Flash.
  (Locked alternative pointer; source:
  ``csrc/moe/dsv3_router_gemm_*.cu``.)
* **Tier 2** -- ``fp32_router_gemm`` (CUDA, fp32-specialized): H=3072,
  E=256, M<=32, fp32 weight. NOT V4-Flash. (Locked alternative pointer;
  source: ``csrc/libtorch_stable/fp32_router_gemm*.cu``.)
* **Tier 3** -- cuBLASLt ``torch.mm(x, weight.T, out_dtype=fp32)``.
* **Tier 4** -- ``F.linear`` fallback.

For V4-Flash hidden_size = 4096, n_routed_experts = 256 -- neither
Tier-1 nor Tier-2 fires, so the runtime hot path is Tier-3 (cuBLASLt
bf16x bf16 -> fp32). Tier-4 is the bf16->fp32 cast path used if
``params_dtype == fp32`` is forced.

This naive port implements the Tier-3 / Tier-4 path only: a plain
matmul ``hidden_states @ weight.T`` with bf16 inputs and fp32 output.
Tier-1 and Tier-2 are documented as locked alternative pointers; future
divergence (e.g., to wire in the actual DSV3 specialized kernel) can
sit alongside this one without breaking the V4 catalog.

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/dsv3_router_gemm_v4_sm100.cuh``
* Task name: ``dsv3_router_gemm_v4_sm100``
* Enum slot: ``TASK_DSV3_ROUTER_GEMM_V4_SM100 = 381``

Blackwell impl is intentionally NAIVE:

* One CTA per token. ``grid = (T, 1, 1)``, ``block = (256, 1, 1)``.
* For each expert ``e in [0, NUM_EXPERTS)``: thread-strided fp32
  dot-product over K = HIDDEN_SIZE, warp + cross-warp reduce, thread 0
  writes ``out[t, e]``.
* fp32 accumulator; bf16 inputs; fp32 output.
* No TMA / no UMMA / no warp-spec / no split-K.

Audit
-----
* dtype: bf16 in, bf16 weight, fp32 out. Matches Tier 3 / Tier 4
  contract (``topk_softplus_sqrt`` consumes fp32 logits).
* layout: ``hidden_states: [T, H]``, ``weight: [E, H]``, ``out: [T, E]``.
  All row-major contiguous.
* multi-batch: partitioned on dim 0 (tokens). Test uses
  ``max_num_batched_requests = 4``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor

__all__ = ["V4Dsv3RouterGemm"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class V4Dsv3RouterGemm(MPKModule):
    """V4-Flash dsv3_router_gemm naive Blackwell kernel (Tier-3 / Tier-4).

    Owns one ``nn.Parameter``:

      * ``weight: [num_experts, hidden_size]`` bf16 -- the gate weight.

    Constructor args:

      * ``hidden_size`` -- input feature dim (V4-Flash: 4096).
      * ``num_experts`` -- output dim / expert count (V4-Flash: 256).
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if hidden_size <= 0:
            raise ValueError(
                f"V4Dsv3RouterGemm hidden_size must be positive; "
                f"got {hidden_size}"
            )
        if num_experts <= 0:
            raise ValueError(
                f"V4Dsv3RouterGemm num_experts must be positive; "
                f"got {num_experts}"
            )
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.weight = nn.Parameter(
            torch.empty(num_experts, hidden_size, dtype=torch.bfloat16)
        )

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Eager reference: ``F.linear(hidden_states, weight)`` with fp32 out.

        Mirrors Tier-3 (cuBLASLt) and Tier-4 (F.linear with explicit fp32
        out) -- both produce a fp32 ``[T, E]`` logits tensor.
        """
        if hidden_states.dim() != 2:
            raise ValueError(
                "hidden_states must be 2-D [T, H]; "
                f"got shape {tuple(hidden_states.shape)}"
            )
        if hidden_states.shape[1] != self.hidden_size:
            raise ValueError(
                f"hidden_states last dim ({hidden_states.shape[1]}) != "
                f"hidden_size ({self.hidden_size})"
            )
        # fp32 accumulate to match the kernel.
        x_f = hidden_states.to(torch.float32)
        w_f = self.weight.to(torch.float32)
        return F.linear(x_f, w_f)  # [T, E] fp32

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, hidden_states_dt: DTensor) -> GridDim:
        """One CTA per token (capped at ``num_workers``)."""
        pk = current_pk()
        num_tokens = hidden_states_dt.dim(0)
        return (max(1, min(num_tokens, pk.num_workers)), 1, 1)

    # ------------------------------------------------------------------
    # MPK registration
    # ------------------------------------------------------------------
    def compile(
        self,
        hidden_states: DTensor,
        *,
        output: Optional[torch.Tensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``dsv3_router_gemm_v4_sm100`` task.

        Tensor contract:
          hidden_states : (T, H) bf16, row-major contiguous.
          weight (auto) : (E, H) bf16 nn.Parameter.
          out           : (T, E) fp32, allocated if None.
        """
        from .....core import CyTBGraph, float32 as _mi_f32
        from .....kernel import TBGraph

        pk = current_pk()

        if hidden_states.num_dims != 2:
            raise ValueError(
                "V4Dsv3RouterGemm: hidden_states must be 2-D; got "
                f"num_dims={hidden_states.num_dims}"
            )
        if hidden_states.dim(1) != self.hidden_size:
            raise ValueError(
                f"V4Dsv3RouterGemm: hidden_states.dim(1)={hidden_states.dim(1)} "
                f"!= hidden_size ({self.hidden_size})"
            )

        num_tokens = hidden_states.dim(0)

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(hidden_states)
        if block_dim is None:
            block_dim = self.default_block_dim()

        weight_dt = pk.attach_input(
            self.weight.data, name=f"{self.prefix}weight"
        )

        if output is None:
            out_dt = pk.new_tensor(
                dims=(num_tokens, self.num_experts),
                dtype=_mi_f32,
                name=f"{self.prefix}router_logits",
            )
        elif isinstance(output, torch.Tensor):
            out_dt = pk.attach_input(
                output, name=f"{self.prefix}router_logits"
            )
        elif isinstance(output, DTensor):
            out_dt = output
        else:
            raise TypeError(
                "V4Dsv3RouterGemm.compile output must be None, "
                f"torch.Tensor, or DTensor; got {type(output).__name__}"
            )

        # ----- TBGraph construction --------------------------------------
        # hidden_states + out partition on dim 0 (tokens); weight is
        # broadcast across all tasks. Runtime preoffsets the per-token
        # slices for the partitioned tensors.
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(hidden_states, (0, -1, -1), 1, True)
        tb_graph.new_input(weight_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(out_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [hidden_states, weight_dt, out_dt], tb_graph
        )
        pk.kn_graph.register_task(tb_graph, "dsv3_router_gemm_v4_sm100")
        return out_dt
