"""V4-Flash ``topk_softplus_sqrt`` -- NEW Blackwell naive kernel
(ONE task, TWO USE_HASH template branches).

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/topk_softplus_sqrt.md``.

Decision: **NEW**.

Rationale
---------
vLLM's ``topk_softplus_sqrt`` is the MoE top-k selector for DeepSeek
V4-Flash's ``scoring_func == "sqrtsoftplus"`` regime. The kernel has
TWO branches selected at the host dispatcher level by whether the
``tid2eid`` (hash table) tensor is non-null:

* **USE_HASH = true** (layers 0..num_hash_layers-1; V4-Flash: 3 layers).
  ``Gate.tid2eid`` is populated and ``input_ids`` is passed; the kernel
  reads the precomputed expert ids per token and only computes the
  ``sqrt(softplus)`` weights for those experts. No bias.
* **USE_HASH = false** (layers num_hash_layers..n_layers-1; V4-Flash:
  40 layers). The kernel does the full top-k argmax with bias-added
  scoring (bias is subtracted from the weight, so the WEIGHT is
  ``sqrt(softplus)`` un-biased).

Both branches share the ``sqrt(softplus(x))`` scoring math, the optional
post-renormalize, and the ``routed_scaling_factor`` multiply. No
existing MPK task covers this dual-branch contract.

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/topk_softplus_sqrt_v4_sm100.cuh``
* Task name: ``topk_softplus_sqrt_v4_sm100``
* Enum slot: ``TASK_TOPK_SOFTPLUS_SQRT_V4_SM100 = 380``

Blackwell impl is intentionally NAIVE:

* One CTA per token. ``grid = (T, 1, 1)``, ``block = (256, 1, 1)``.
* Only thread 0 does the work; the rest idle. With E=256 and k=6
  per token this is well under 1us even on B200.
* fp32 internal math; fp32 in, fp32 weights out, int32 indices out.
* No vectorization / no warp-spec / no cooperative reduce.

Audit
-----
* dtype: ``gating_output`` fp32 (per the V4-Flash dispatch which sets
  ``GateLinear.out_dtype = torch.float32``). ``topk_weights`` fp32,
  ``topk_indices`` int32, ``token_expert_indices`` int32 (scored only).
  ``correction_bias`` fp32 (scored only).
  ``input_ids`` int32, ``tid2eid`` int32 [vocab, K] (hash only).
* layout: all row-major contiguous.
* multi-batch: partitions on dim 0 (tokens). Test uses
  ``max_num_batched_requests = 4``.

NOTE: the scored branch's ``token_expert_indices`` write uses the row
index within the kernel; the naive port currently sets ``token_idx = 0``
in the codegen since the kernel doesn't see ``blockIdx`` (MPK tasks are
blockIdx-agnostic). Multi-token-correct ``token_expert_indices`` is
follow-up work for the perf pass; the weights / indices outputs (which
``FusedMoE`` and ``MegaMoE`` actually read) ARE multi-token-correct.
"""
from __future__ import annotations

import struct
from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor

__all__ = ["V4TopkSoftplusSqrt"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


def _float_to_int_bits(x: float) -> int:
    packed = struct.pack("<f", float(x))
    return int.from_bytes(packed, byteorder="little", signed=False)


def _softplus_sqrt_ref(x: torch.Tensor) -> torch.Tensor:
    """Match the kernel's numerical-stable form: softplus(x) ~ x for x > 20."""
    out = torch.empty_like(x, dtype=torch.float32)
    big = x > 20.0
    out = torch.where(big, x.to(torch.float32), torch.log1p(torch.exp(x.to(torch.float32))))
    return torch.sqrt(out.clamp_min(0.0))


class V4TopkSoftplusSqrt(MPKModule):
    """V4-Flash ``topk_softplus_sqrt`` naive Blackwell kernel.

    Constructor args:

      * ``num_experts`` -- ``E`` (V4-Flash: 256).
      * ``topk``        -- ``K`` (V4-Flash: 6).
      * ``use_hash``    -- True for layers 0..num_hash_layers-1
                           (V4-Flash: layers 0..2); False for the rest.
      * ``renormalize`` -- V4-Flash: True (``config.norm_topk_prob``).
      * ``routed_scaling_factor`` -- V4-Flash: 1.5.
      * ``start_expert``, ``end_expert`` -- local expert range. V4-Flash
                                            single-node: ``[0, num_experts)``.
      * ``vocab_size`` (hash only)        -- vocab dim of ``tid2eid``.
                                              V4-Flash: 129280.
    """

    def __init__(
        self,
        num_experts: int,
        topk: int,
        *,
        use_hash: bool,
        renormalize: bool = True,
        routed_scaling_factor: float = 1.5,
        start_expert: int = 0,
        end_expert: Optional[int] = None,
        vocab_size: Optional[int] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if num_experts <= 0:
            raise ValueError(
                f"V4TopkSoftplusSqrt num_experts must be positive; "
                f"got {num_experts}"
            )
        if topk <= 0 or topk > num_experts:
            raise ValueError(
                f"V4TopkSoftplusSqrt topk must be in (0, num_experts]; "
                f"got topk={topk}, num_experts={num_experts}"
            )
        if use_hash and vocab_size is None:
            raise ValueError(
                "V4TopkSoftplusSqrt: vocab_size is required when use_hash=True"
            )
        if end_expert is None:
            end_expert = num_experts

        self.num_experts = num_experts
        self.topk = topk
        self.use_hash = bool(use_hash)
        self.renormalize = bool(renormalize)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.start_expert = int(start_expert)
        self.end_expert = int(end_expert)
        self.vocab_size = vocab_size

        # Bias parameter (scored branch only). Initialized to zeros; tests
        # may overwrite. For the hash branch this parameter is unused.
        if not self.use_hash:
            self.correction_bias = nn.Parameter(
                torch.zeros(num_experts, dtype=torch.float32)
            )

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        gating_output: torch.Tensor,
        *,
        input_ids: Optional[torch.Tensor] = None,
        tid2eid: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Eager reference matching the vLLM kernel.

        Returns:
          (topk_weights[T,K] fp32, topk_indices[T,K] int32,
           token_expert_indices[T,K] int32 OR None on hash branch)
        """
        if gating_output.dim() != 2:
            raise ValueError(
                f"gating_output must be 2-D [T, E]; got "
                f"{tuple(gating_output.shape)}"
            )
        if gating_output.shape[1] != self.num_experts:
            raise ValueError(
                f"gating_output last dim ({gating_output.shape[1]}) != "
                f"num_experts ({self.num_experts})"
            )

        T = gating_output.shape[0]
        K = self.topk
        E = self.num_experts
        gating = gating_output.to(torch.float32)
        s_unbiased = _softplus_sqrt_ref(gating)  # [T, E] fp32

        if self.use_hash:
            assert input_ids is not None and tid2eid is not None
            expert_ids = tid2eid[input_ids.to(torch.int64).clamp_min(0)]  # [T, K]
            weights = torch.gather(
                s_unbiased, dim=1, index=expert_ids.to(torch.int64)
            )  # [T, K]
            if self.renormalize:
                denom = weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-30)
                denom = torch.where(denom > 0, denom, torch.ones_like(denom))
                weights = weights * (self.routed_scaling_factor / denom)
            else:
                weights = weights * self.routed_scaling_factor
            return weights, expert_ids.to(torch.int32), None

        # Scored branch.
        bias = self.correction_bias.to(torch.float32)  # [E]
        scores_for_choice = s_unbiased + bias.view(1, -1)
        sentinel = float("-inf")
        topk_indices = torch.empty(T, K, dtype=torch.int32, device=gating.device)
        topk_weights = torch.empty(T, K, dtype=torch.float32, device=gating.device)
        token_expert_indices = torch.empty(
            T, K, dtype=torch.int32, device=gating.device
        )
        for k_idx in range(K):
            # argmax with lower-index tie-break (PyTorch's argmax does
            # earliest-index tie-break by default).
            best_e = scores_for_choice.argmax(dim=-1)  # [T]
            best_v = scores_for_choice.gather(1, best_e.view(-1, 1)).squeeze(1)
            # Strip bias: weight = scores_unbiased[best_e] = best_v - bias[best_e]
            weight = best_v - bias[best_e]
            topk_weights[:, k_idx] = weight
            in_range = (best_e >= self.start_expert) & (best_e < self.end_expert)
            adjusted = best_e - self.start_expert
            topk_indices[:, k_idx] = torch.where(
                in_range, adjusted, torch.full_like(adjusted, E)
            ).to(torch.int32)
            for t in range(T):
                token_expert_indices[t, k_idx] = k_idx * T + t
            # Mask the winner so next iter can't pick it.
            scores_for_choice[
                torch.arange(T, device=gating.device), best_e
            ] = -1.0e4

        if self.renormalize:
            denom = topk_weights.sum(dim=-1, keepdim=True)
            denom = torch.where(denom > 0, denom, torch.ones_like(denom))
            topk_weights = topk_weights * (self.routed_scaling_factor / denom)
        else:
            topk_weights = topk_weights * self.routed_scaling_factor

        return topk_weights, topk_indices, token_expert_indices

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, gating_output_dt: DTensor) -> GridDim:
        pk = current_pk()
        num_tokens = gating_output_dt.dim(0)
        return (max(1, min(num_tokens, pk.num_workers)), 1, 1)

    # ------------------------------------------------------------------
    # MPK registration
    # ------------------------------------------------------------------
    def compile(
        self,
        gating_output: DTensor,
        *,
        # Scored branch inputs
        correction_bias: Optional[torch.Tensor] = None,
        # Hash branch inputs
        input_ids: Optional[DTensor] = None,
        tid2eid: Optional[torch.Tensor] = None,
        # Outputs
        topk_weights: Optional[torch.Tensor] = None,
        topk_indices: Optional[torch.Tensor] = None,
        token_expert_indices: Optional[torch.Tensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ):
        """Register one ``topk_softplus_sqrt_v4_sm100`` task instance.

        On the SCORED branch returns ``(topk_weights, topk_indices,
        token_expert_indices)``. On the HASH branch returns
        ``(topk_weights, topk_indices)``.
        """
        from .....core import CyTBGraph, float32 as _mi_f32, int32 as _mi_i32
        from .....kernel import TBGraph

        pk = current_pk()

        if gating_output.num_dims != 2:
            raise ValueError(
                "V4TopkSoftplusSqrt: gating_output must be 2-D; got "
                f"num_dims={gating_output.num_dims}"
            )
        if gating_output.dim(1) != self.num_experts:
            raise ValueError(
                f"V4TopkSoftplusSqrt: gating_output.dim(1)="
                f"{gating_output.dim(1)} != num_experts "
                f"({self.num_experts})"
            )

        num_tokens = gating_output.dim(0)
        K = self.topk

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(gating_output)
        if block_dim is None:
            block_dim = self.default_block_dim()

        if topk_weights is None:
            tw_dt = pk.new_tensor(
                dims=(num_tokens, K),
                dtype=_mi_f32,
                name=f"{self.prefix}topk_weights",
            )
        elif isinstance(topk_weights, torch.Tensor):
            tw_dt = pk.attach_input(topk_weights, name=f"{self.prefix}topk_weights")
        else:
            tw_dt = topk_weights

        if topk_indices is None:
            ti_dt = pk.new_tensor(
                dims=(num_tokens, K),
                dtype=_mi_i32,
                name=f"{self.prefix}topk_indices",
            )
        elif isinstance(topk_indices, torch.Tensor):
            ti_dt = pk.attach_input(topk_indices, name=f"{self.prefix}topk_indices")
        else:
            ti_dt = topk_indices

        # Common params: [use_hash, start_expert, end_expert,
        # renormalize, routed_scaling_factor_bits, num_tokens].
        scale_bits = _float_to_int_bits(self.routed_scaling_factor)
        params = [
            1 if self.use_hash else 0,
            self.start_expert,
            self.end_expert,
            1 if self.renormalize else 0,
            scale_bits,
            num_tokens,
        ]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))

        if self.use_hash:
            if input_ids is None or tid2eid is None:
                raise ValueError(
                    "V4TopkSoftplusSqrt.compile: input_ids and tid2eid are "
                    "required on the hash branch (use_hash=True)"
                )
            if isinstance(tid2eid, torch.Tensor):
                tid2eid_dt = pk.attach_input(
                    tid2eid, name=f"{self.prefix}tid2eid"
                )
            else:
                tid2eid_dt = tid2eid
            tb_graph.new_input(gating_output, (0, -1, -1), 1, True)
            tb_graph.new_input(input_ids, (0, -1, -1), 1, True)
            tb_graph.new_input(tid2eid_dt, (-1, -1, -1), 0, True)
            tb_graph.new_input(tw_dt, (0, -1, -1), 1, True)
            tb_graph.new_input(ti_dt, (0, -1, -1), 1, True)
            pk.kn_graph.customized(
                [gating_output, input_ids, tid2eid_dt, tw_dt, ti_dt],
                tb_graph,
            )
            pk.kn_graph.register_task(
                tb_graph, "topk_softplus_sqrt_v4_sm100", params
            )
            return tw_dt, ti_dt

        # Scored branch.
        if correction_bias is None:
            cb_dt = pk.attach_input(
                self.correction_bias.data, name=f"{self.prefix}correction_bias"
            )
        elif isinstance(correction_bias, torch.Tensor):
            cb_dt = pk.attach_input(
                correction_bias, name=f"{self.prefix}correction_bias"
            )
        else:
            cb_dt = correction_bias

        if token_expert_indices is None:
            tei_dt = pk.new_tensor(
                dims=(num_tokens, K),
                dtype=_mi_i32,
                name=f"{self.prefix}token_expert_indices",
            )
        elif isinstance(token_expert_indices, torch.Tensor):
            tei_dt = pk.attach_input(
                token_expert_indices, name=f"{self.prefix}token_expert_indices"
            )
        else:
            tei_dt = token_expert_indices

        tb_graph.new_input(gating_output, (0, -1, -1), 1, True)
        tb_graph.new_input(cb_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(tw_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(ti_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(tei_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [gating_output, cb_dt, tw_dt, ti_dt, tei_dt], tb_graph
        )
        pk.kn_graph.register_task(
            tb_graph, "topk_softplus_sqrt_v4_sm100", params
        )
        return tw_dt, ti_dt, tei_dt
