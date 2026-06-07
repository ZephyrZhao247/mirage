"""V4-Flash ``save_partial_states`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/save_partial_states.md``.

Decision: **NEW**.

Rationale
---------
The vLLM Triton kernel is per-token:

* slot_id = slot_mapping[t]; if slot_id < 0 return.
* state_cache[slot, :HEAD_SIZE]  = kv[t]
* state_cache[slot, HEAD_SIZE:]  = score[t] + ape[positions[t] % COMPRESS_RATIO]

There is no existing MPK task with this signature (5 inputs + 1 paged
output indexed by data-dependent slot_mapping).

So this is a NEW kernel:

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/save_partial_states_v4_sm100.cuh``
* Task name:   ``save_partial_states_v4_sm100``
* Enum slot:   ``TASK_SAVE_PARTIAL_STATES_V4_SM100 = 372``

The Blackwell impl is intentionally NAIVE:

* One CTA per token. ``grid = (num_tokens, 1, 1)``. ``block = (256,1,1)``.
* Thread-strided element loop over HEAD_SIZE; no smem staging.
* All fp32 math.

Audit
-----
* dtype: kv/score/ape all fp32 (matches the spec's up-cast inside
  ``DeepseekCompressor.forward``); positions/slot_mapping int64;
  state_cache fp32 in-place.
* layout: row-major contiguous. Per-token tensors PARTITION on dim 0;
  ``ape`` and ``state_cache`` are BROADCAST (the kernel addresses
  them by ``positions[t] % COMPRESS_RATIO`` and ``slot_mapping[t]``).
* multi-batch: by construction (token-axis partition); test uses
  ``max_num_batched_requests >= 2``.
* ``forward()``: faithful PyTorch reference matching the spec.
"""
from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

import mirage as mi

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor


__all__ = ["V4SavePartialStates"]

GridDim = tuple
BlockDim = tuple


class V4SavePartialStates(MPKModule):
    """V4-Flash compressor save_partial_states naive Blackwell kernel.

    Stateless wrapper (no learned parameters). The APE table lives in
    the upstream ``DeepseekCompressor`` as a buffer; the catalog
    accepts it as an attached input.

    Constructor args:

      * ``head_size``      -- last-dim of kv/score; equals
                              ``coff * head_dim`` of the spec.
                              V4-Flash attention compressor: 1024 (or
                              512 for ratio=128). Indexer compressor:
                              256.
      * ``compress_ratio`` -- 4 (overlap) or 128 (non-overlap).
                              Determines the APE-table row count.
      * ``block_size``     -- tokens per state-cache block.
    """

    def __init__(
        self,
        head_size: int,
        compress_ratio: int,
        block_size: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if head_size <= 0:
            raise ValueError(
                f"V4SavePartialStates: head_size must be positive; got {head_size}"
            )
        if compress_ratio not in (4, 128):
            raise ValueError(
                f"V4SavePartialStates: compress_ratio must be 4 or 128 (V4-Flash); "
                f"got {compress_ratio}"
            )
        if block_size <= 0:
            raise ValueError(
                f"V4SavePartialStates: block_size must be positive; got {block_size}"
            )
        self.head_size = head_size
        self.compress_ratio = compress_ratio
        self.block_size = block_size

    # ------------------------------------------------------------------
    # PyTorch reference (faithful to the Triton kernel).
    # ------------------------------------------------------------------
    def forward(
        self,
        kv: torch.Tensor,             # fp32 [T, head_size]
        score: torch.Tensor,          # fp32 [T, head_size]
        ape: torch.Tensor,            # fp32 [compress_ratio, head_size]
        positions: torch.Tensor,      # int64 [T]
        slot_mapping: torch.Tensor,   # int64 [T]
        state_cache: torch.Tensor,    # fp32 [num_blocks, block_size, 2*head_size]
    ) -> torch.Tensor:
        """Returns the updated ``state_cache`` (in-place semantics)."""
        assert kv.dtype == torch.float32
        assert score.dtype == torch.float32
        assert ape.dtype == torch.float32
        assert state_cache.dtype == torch.float32
        assert positions.dtype == torch.int64
        assert slot_mapping.dtype == torch.int64
        assert kv.shape == score.shape
        assert kv.shape[1] == self.head_size
        assert ape.shape == (self.compress_ratio, self.head_size)
        assert state_cache.shape[1] == self.block_size
        assert state_cache.shape[2] == 2 * self.head_size

        T = kv.shape[0]
        for t in range(T):
            slot = int(slot_mapping[t].item())
            if slot < 0:
                continue
            blk = slot // self.block_size
            off = slot % self.block_size
            row = int(positions[t].item()) % self.compress_ratio
            state_cache[blk, off, : self.head_size] = kv[t]
            state_cache[blk, off, self.head_size :] = score[t] + ape[row]
        return state_cache

    def auto_grid_dim(self, kv_dt: Any) -> GridDim:
        pk = current_pk()
        num_tokens = kv_dt.dim(0)
        return (max(1, min(num_tokens, int(pk.num_workers))), 1, 1)

    def compile(
        self,
        kv_dt: Any,
        score_dt: Any,
        ape_dt: Any,
        positions_dt: Any,
        slot_mapping_dt: Any,
        state_cache_dt: Any,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``save_partial_states_v4_sm100`` task.

        Tensor contract (all contiguous row-major):
          kv             : (T, head_size)                         fp32  (dim-0 part)
          score          : (T, head_size)                         fp32  (dim-0 part)
          ape            : (compress_ratio, head_size)            fp32  (broadcast)
          positions      : (T,)                                   int64 (dim-0 part)
          slot_mapping   : (T,)                                   int64 (dim-0 part)
          state_cache    : (num_blocks, block_size, 2*head_size)  fp32  (broadcast, in-place)
        """
        from .....core import CyTBGraph
        from .....kernel import TBGraph

        pk = current_pk()

        if kv_dt.num_dims != 2 or kv_dt.dim(1) != self.head_size:
            raise ValueError(
                f"V4SavePartialStates: kv_dt must be 2-D (T, {self.head_size}); "
                f"got num_dims={kv_dt.num_dims}, dim(1)="
                f"{kv_dt.dim(1) if kv_dt.num_dims > 1 else 'NA'}"
            )
        if score_dt.num_dims != 2 or score_dt.dim(1) != self.head_size:
            raise ValueError(
                f"V4SavePartialStates: score_dt must be 2-D (T, {self.head_size})"
            )
        if ape_dt.num_dims != 2 or ape_dt.dim(0) != self.compress_ratio:
            raise ValueError(
                f"V4SavePartialStates: ape_dt must be 2-D (compress_ratio="
                f"{self.compress_ratio}, head_size={self.head_size})"
            )
        if positions_dt.num_dims != 1:
            raise ValueError(
                f"V4SavePartialStates: positions_dt must be 1-D (T,)"
            )
        if slot_mapping_dt.num_dims != 1:
            raise ValueError(
                f"V4SavePartialStates: slot_mapping_dt must be 1-D (T,)"
            )
        if state_cache_dt.num_dims != 3 or state_cache_dt.dim(1) != self.block_size:
            raise ValueError(
                f"V4SavePartialStates: state_cache_dt must be 3-D "
                f"(num_blocks, block_size={self.block_size}, "
                f"2*head_size={2*self.head_size})"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(kv_dt)
        if block_dim is None:
            block_dim = self.default_block_dim()  # (256,1,1) on Blackwell

        # TBGraph partition layout:
        #   per-token tensors: PARTITION on dim 0.
        #   ape, state_cache:  BROADCAST.
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(kv_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(score_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(ape_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(positions_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(slot_mapping_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(state_cache_dt, (-1, -1, -1), 0, True)
        pk.kn_graph.customized(
            [
                kv_dt,
                score_dt,
                ape_dt,
                positions_dt,
                slot_mapping_dt,
                state_cache_dt,
            ],
            tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "save_partial_states_v4_sm100")
        return state_cache_dt
