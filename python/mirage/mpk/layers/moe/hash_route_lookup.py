"""DeepSeek V4-Flash hash-based MoE expert routing for early layers.

Backed by ``tasks/blackwell/hash_route_lookup_sm100.cuh`` (task name
``"hash_route_lookup_sm100"``).

For ``layer_idx < num_hash_layers`` (3 in V4-Flash) the MoE expert
selection bypasses the gate GEMM + topk score path. Expert indices come
from a precomputed table shipped in the checkpoint at
``layers.{layer_idx}.ffn.gate.tid2eid`` (shape ``[vocab_size,
num_experts_per_tok]``, int32):

    expert_ids[t, k]   = tid2eid[input_ids[t], k]
    topk_weights[t, k] = 1.0 / num_experts_per_tok    # uniform

The uniform weight is the naive reading of the official ``Gate.forward``
hash branch (deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py, the
``self.hash`` branch). See ``docs/mpk/deepseek_v4/moe.md`` §1.2 OPEN
note — vLLM's hash path *also* recomputes scores via the
``topk_softplus_sqrt`` kernel and gathers at the hash-provided indices,
producing non-uniform weights. This module implements the simpler
uniform-weight path; the score-gather composite can be built on top by
combining :class:`HashRouteLookup` with the score-gather kernel.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["HashRouteLookup"]


class HashRouteLookup(MPKModule):
    """Hash-based expert routing via precomputed ``tid2eid`` lookup.

    Constructor args:
      vocab_size           : ``V`` — first dim of ``tid2eid`` (typically 129280).
      num_experts_per_tok  : ``K`` — second dim of ``tid2eid`` (typically 6).
      prefix               : MPK kernel-tensor name prefix.

    State:
      tid2eid (nn.Parameter): ``[V, K]`` int32, populated by
        ``load_state_dict`` from the checkpoint key ``{prefix}tid2eid``.
    """

    def __init__(
        self,
        vocab_size: int,
        num_experts_per_tok: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0; got {vocab_size}")
        if num_experts_per_tok <= 0:
            raise ValueError(
                f"num_experts_per_tok must be > 0; got {num_experts_per_tok}"
            )
        self.vocab_size = vocab_size
        self.num_experts_per_tok = num_experts_per_tok
        # int32 to match the kernel signature and the checkpoint dtype.
        # ``requires_grad=False`` because this is a lookup table, not a
        # learnable weight.
        self.tid2eid = nn.Parameter(
            torch.zeros(vocab_size, num_experts_per_tok, dtype=torch.int32),
            requires_grad=False,
        )

    def forward(
        self, input_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch reference: gather + uniform weights.

        Returns ``(expert_ids [N, K] int32, topk_weights [N, K] fp32)``.
        """
        if input_ids.dim() != 1:
            raise ValueError(
                f"input_ids must be 1-D; got shape {tuple(input_ids.shape)}"
            )
        # Gather; cast to int32 to match the compiled kernel's output dtype.
        ids = input_ids.to(torch.long)
        expert_ids = self.tid2eid.index_select(0, ids).to(torch.int32)
        n = input_ids.size(0)
        k = self.num_experts_per_tok
        topk_weights = torch.full(
            (n, k),
            1.0 / float(k),
            dtype=torch.float32,
            device=input_ids.device,
        )
        return expert_ids, topk_weights

    def auto_grid_dim(self, input_ids_dt: DTensor) -> GridDim:
        """One CTA per token; CTAs derive ``t`` from ``token_offset``."""
        from ... import context as _ctx
        pk = _ctx.current_pk()
        n = input_ids_dt.dim(0)
        return (max(1, min(n, pk.num_workers)), 1, 1)

    def default_block_dim(self) -> BlockDim:
        """The kernel only needs K_TOPK active lanes; pin a single warp
        regardless of arch (the kernel gates higher threadIdx.x internally).
        """
        return (32, 1, 1)

    def compile(
        self,
        input_ids: DTensor,
        *,
        expert_ids: Optional[Union[torch.Tensor, DTensor]] = None,
        topk_weights: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register one ``hash_route_lookup_sm100`` task.

        Tensor contract:
          input_ids    : (N,) int32, the token-id buffer.
          tid2eid (self.tid2eid, attached as ``{prefix}tid2eid``):
                         (vocab_size, K) int32.
          expert_ids   : (N, K) int32 OUT — auto-allocated if None.
          topk_weights : (N, K) fp32 OUT — auto-allocated if None.

        Notes: grid is ``(N, 1, 1)`` with one CTA per token; each CTA
        addresses its row via ``task_metadata.token_offset``. No params
        are passed; ``K`` is read from ``tid2eid.dim(1)`` by the
        registration function.
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        assert input_ids.num_dims == 1
        n = input_ids.dim(0)
        k = self.num_experts_per_tok
        prefix = self.prefix or "hash_route_lookup_"

        # Attach the lookup table. ``attach_input`` accepts an int32 tensor
        # (see convert_torch_type_to_dtype in core.pyx).
        tid2eid_dt = pk.attach_input(self.tid2eid, name=f"{prefix}tid2eid")
        assert tid2eid_dt.num_dims == 2
        assert tid2eid_dt.dim(0) == self.vocab_size
        assert tid2eid_dt.dim(1) == k

        def _attach_out(buf, default_dims, default_dtype_torch,
                        default_dtype_mi, name):
            if buf is None:
                return pk.new_tensor(
                    dims=default_dims, dtype=default_dtype_mi, name=name
                )
            if isinstance(buf, torch.Tensor):
                if buf.dtype != default_dtype_torch:
                    raise ValueError(
                        f"{name} must have dtype {default_dtype_torch}; "
                        f"got {buf.dtype}"
                    )
                return pk.attach_input(buf, name=name)
            if isinstance(buf, DTensor):
                return buf
            raise TypeError(
                f"{name} must be None, torch.Tensor, or DTensor; got "
                f"{type(buf).__name__}"
            )

        expert_ids_dt = _attach_out(
            expert_ids, (n, k), torch.int32, mi.int32,
            f"{prefix}expert_ids",
        )
        topk_weights_dt = _attach_out(
            topk_weights, (n, k), torch.float32, mi.float32,
            f"{prefix}topk_weights",
        )
        assert expert_ids_dt.num_dims == 2
        assert expert_ids_dt.dim(0) == n
        assert expert_ids_dt.dim(1) == k
        assert topk_weights_dt.num_dims == 2
        assert topk_weights_dt.dim(0) == n
        assert topk_weights_dt.dim(1) == k

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(input_ids)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # All inputs/outputs are addressed via ``task_metadata.token_offset``
        # (not via TBGraph partitioning), so all map dims are (-1, -1, -1).
        # This matches the mhc_pre_sm100 convention.
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(input_ids, (-1, -1, -1), -1, True)
        tb_graph.new_input(tid2eid_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(expert_ids_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(topk_weights_dt, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [input_ids, tid2eid_dt, expert_ids_dt, topk_weights_dt], tb_graph
        )
        pk.kn_graph.register_task(tb_graph, "hash_route_lookup_sm100")
        return expert_ids_dt, topk_weights_dt
