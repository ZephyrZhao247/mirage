"""Adapter: convert hash-routing ``[T, K]`` indices to ``MoEPermute``'s
``[E_LOCAL, MBT]`` expert-major slot layout.

Wave 3.5 follow-up to commit 83c38ecc (Gap 3): the V3 ``MoEPermute``
kernel expects ``routing_indices`` to be expert-major and 1-indexed:

    routing_indices[e, t] = k + 1   if token t routes to expert e at slot k
                          = 0       otherwise   (token not routed locally)

``HashRouteLookup`` produces ``expert_ids[t, k]`` -- token-major, raw
expert id. This module is a thin reshape adapter: a pure-PyTorch
reference for now (used by the layer-level test), and a placeholder
``compile()`` that documents what kernel signature would be needed if
this work cannot be folded into ``MoEPermute`` itself.

OPEN question for Wave 4 / next agent: the cheapest landing for this is
likely an extension of the existing ``moe_permute_sm100`` kernel that
accepts a token-major routing format directly (one new kernel template
parameter ``ROUTING_LAYOUT``). That would avoid the scatter pass
entirely. The current scaffold uses a separate adapter to keep the
existing kernel untouched; revisit before un-skipping the L0 compile
test.
"""
from __future__ import annotations

from typing import Optional, Union

import torch

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["HashRouteToExpertMajor", "hash_route_to_expert_major"]


def hash_route_to_expert_major(
    expert_ids: torch.Tensor,
    num_local_experts: int,
    mbt: Optional[int] = None,
) -> torch.Tensor:
    """Pure-PyTorch reference scatter -- ``[T, K]`` -> ``[E_LOCAL, MBT]``.

    Args:
      expert_ids        : ``[T, K]`` int32 -- hash-routed expert ids.
      num_local_experts : ``E_LOCAL`` -- range of valid expert ids.
      mbt               : output column count ``MBT`` (defaults to ``T``).

    Returns:
      ``routing_indices [E_LOCAL, MBT]`` int32. Entry ``[e, t]`` is
      ``k + 1`` if ``expert_ids[t, k] == e`` for some ``k``, else ``0``.
    """
    T, K = expert_ids.shape
    if mbt is None:
        mbt = T
    out = torch.zeros(
        num_local_experts, mbt, dtype=torch.int32, device=expert_ids.device
    )
    for t in range(T):
        for k in range(K):
            e = int(expert_ids[t, k].item())
            if 0 <= e < num_local_experts:
                out[e, t] = k + 1
    return out


class HashRouteToExpertMajor(MPKModule):
    """Catalog wrapper -- scatter ``[T, K]`` -> ``[E_LOCAL, MBT]``.

    Wave 3.5 status: this is a scaffold. The MPK compile path could
    either:

      (a) bake this scatter into ``moe_permute_sm100`` as a new template
          parameter ``ROUTING_LAYOUT={EXPERT_MAJOR, TOKEN_MAJOR}`` -- the
          recommended path because the scatter is tiny (T*K iterations,
          T<=MBT~=128) and merging it avoids a kernel boundary; OR
      (b) ship a new ``hash_route_to_expert_major_sm100`` task with one
          CTA per expert. This module's ``compile()`` reserves the latter
          signature.

    Both options are deferred until the next wave. ``forward()`` works
    today and is used by the PyTorch reference in
    :class:`DeepseekV4Block.forward`.
    """

    def __init__(
        self,
        num_local_experts: int,
        num_experts_per_tok: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if num_local_experts <= 0:
            raise ValueError(
                f"num_local_experts must be > 0; got {num_local_experts}"
            )
        if num_experts_per_tok <= 0:
            raise ValueError(
                f"num_experts_per_tok must be > 0; got {num_experts_per_tok}"
            )
        self.num_local_experts = num_local_experts
        self.num_experts_per_tok = num_experts_per_tok

    def forward(
        self,
        expert_ids: torch.Tensor,
        mbt: Optional[int] = None,
    ) -> torch.Tensor:
        return hash_route_to_expert_major(
            expert_ids, self.num_local_experts, mbt
        )

    def auto_grid_dim(self, *_) -> GridDim:
        return (self.num_local_experts, 1, 1)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)

    def compile(
        self,
        expert_ids: DTensor,
        routing_indices: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        raise NotImplementedError(
            "HashRouteToExpertMajor.compile() is a Wave-3.5 scaffold. "
            "Wave 4 recommends folding the scatter into "
            "moe_permute_sm100 (new ROUTING_LAYOUT template parameter) "
            "rather than landing a standalone kernel. See module "
            "docstring for the trade-offs."
        )
