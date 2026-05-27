"""DeepSeek V4-Flash Compressor state-update step (sub-batch C2).

Backed by ``tasks/blackwell/compressor_state_update_sm100.cuh`` (task name
``"compressor_state_update_sm100"``).

This module owns the *state-write* half of the V4-Flash ``Compressor``;
the matching *compress* half is its sibling task
``compressor_compress_sm100``.  The pair together replaces the vLLM
two-kernel design (`_save_partial_states_kernel` +
`_fused_compress_quant_cache`) from
``deps/vllm/vllm/model_executor/layers/deepseek_compressor.py``.

For each token ``t`` (one CTA per token) we write a single row of width
``2 * head_dim`` into a ring buffer of "partial state" rows::

    slot_id = slot_mapping[t]                  # skip if slot_id < 0
    row     = state_cache[slot_id]             # [2 * head_dim] bf16
    row[              :head_dim] = kv[t]
    row[head_dim:2 * head_dim]   = score[t] + ape[positions[t] % R]

where ``ape : [R, head_dim]`` is a learned per-token-in-window positional
offset, and ``R = compress_ratio``.  ``overlap`` is forwarded as a
constexpr to the kernel for parity with the spec, but the slot_id is
pre-computed by the caller (matching the Triton reference at
``deepseek_compressor.py:401-416``); the kernel does NOT recompute it.

Reference Triton kernel:
``deps/vllm/vllm/model_executor/layers/deepseek_compressor.py:380-433``.

Reference PyTorch oracle:
``deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`` —
``Compressor.forward`` state-write logic (lines 316-360).

OPEN questions about the state-cache layout (deferred to integration
with sibling task / Python ``compressor_layer`` method):

  * ``coff`` (per-head expansion factor): the spec's full layout is
    ``state_cache : [num_blocks, block_size_state, 2 * coff * head_dim]``
    where ``coff ∈ {1, 2}`` depending on ``compress_ratio``.  v1 collapses
    the kernel surface to ``coff = 1`` by folding any per-head expansion
    into ``head_dim`` at the Python catalog layer; the downstream
    ``compressor_compress_sm100`` sibling task is expected to do the
    same.  If the integration uncovers a need for the kernel to see the
    raw ``coff > 1`` shape, the kernel template already supports it via
    a larger ``HEAD_DIM`` (the row width is ``2 * HEAD_DIM``).

  * ``overlap`` semantics: the spec's overlap layout is `state_row[:D_h] =
    "incoming"`, `state_row[D_h:2*D_h] = "outgoing"` — i.e. two copies of
    kv across the window boundary.  v1 assumes the caller's
    ``slot_mapping`` already accounts for this; the kernel just writes
    one (kv, score+ape) pair per token at the slot_mapping-indicated row.
    The OVERLAP template flag is reserved for a future fused variant
    that derives both slots inside the kernel.

  * ``block_size_state`` paging: the kernel sees a flat
    ``[num_slots, 2*head_dim]`` layout (i.e. ``block_size_state == 1``)
    which matches the Triton reference's behaviour for that paging
    parameter.  Production code with ``block_size_state ∈ {4, 8}`` packs
    the slot index the same way (``slot = block_idx * block_size + pos_in_block``)
    so the call site is identical from the kernel's POV.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["CompressorStateUpdate"]


class CompressorStateUpdate(MPKModule):
    """Compressor state-update task wrapper.

    Constructor args:
      head_dim        : ``D_h`` — per-token feature width of one half of the
                         state row.  In real V4-Flash this is 512 (attention
                         compressor) or 128 (indexer compressor) when
                         ``coff = 1``.
      compress_ratio  : ``R`` — number of input tokens collapsed into one
                         compressed-KV entry (4 or 128 in V4-Flash).
      overlap         : whether the upstream window uses overlapping halves
                         (``True`` for ratio=4, ``False`` for ratio=128).
                         Forwarded to the kernel as a constexpr template
                         flag; see module docstring for caveats.
      prefix          : ``state_dict`` key prefix.

    State:
      ape (nn.Parameter): ``[compress_ratio, head_dim]`` bf16 — learned
        per-position-in-window offset added to ``score`` before it is
        written into the ring buffer.
    """

    def __init__(
        self,
        head_dim: int,
        compress_ratio: int,
        overlap: bool = True,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0; got {head_dim}")
        if compress_ratio <= 0:
            raise ValueError(
                f"compress_ratio must be > 0; got {compress_ratio}"
            )
        self.head_dim = head_dim
        self.compress_ratio = compress_ratio
        self.overlap = bool(overlap)
        # Learned per-position offset; bf16 to match the rest of the
        # Compressor's bf16-only state-cache row.
        self.ape = nn.Parameter(
            torch.zeros(compress_ratio, head_dim, dtype=torch.bfloat16)
        )

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        kv: torch.Tensor,
        score: torch.Tensor,
        positions: torch.Tensor,
        slot_mapping: torch.Tensor,
        state_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Functional reference — returns the updated ``state_cache``.

        The PyTorch reference does an in-place update of the supplied
        ``state_cache`` and returns the same tensor, matching the
        kernel's in-place semantics.

        Args:
          kv           : ``[T, head_dim]`` bf16 — per-token KV slot value.
          score        : ``[T, head_dim]`` bf16 — per-token gated score
                          (post-gate-proj).
          positions    : ``[T]`` int32 — absolute token positions.
          slot_mapping : ``[T]`` int32 — destination slot for each token
                          in ``state_cache``; ``-1`` means "skip".
          state_cache  : ``[num_slots, 2 * head_dim]`` bf16 — in/out
                          ring buffer.
        """
        if kv.shape != (kv.size(0), self.head_dim):
            raise ValueError(
                f"kv must be [T, head_dim={self.head_dim}]; "
                f"got {tuple(kv.shape)}"
            )
        if score.shape != kv.shape:
            raise ValueError(
                f"score must match kv shape; got {tuple(score.shape)} "
                f"vs {tuple(kv.shape)}"
            )
        T = kv.size(0)
        if positions.shape != (T,):
            raise ValueError(
                f"positions must be [T={T}]; got {tuple(positions.shape)}"
            )
        if slot_mapping.shape != (T,):
            raise ValueError(
                f"slot_mapping must be [T={T}]; "
                f"got {tuple(slot_mapping.shape)}"
            )
        if state_cache.dim() != 2 or state_cache.size(1) != 2 * self.head_dim:
            raise ValueError(
                f"state_cache must be [num_slots, 2*head_dim="
                f"{2 * self.head_dim}]; got {tuple(state_cache.shape)}"
            )

        ape = self.ape.to(device=kv.device)
        # Vectorize over all valid tokens: build the [T, 2*head_dim] row
        # to scatter, mask out -1 slots, then index_copy into state_cache.
        slot_i64 = slot_mapping.to(torch.long)
        valid = slot_i64 >= 0
        if not torch.any(valid):
            return state_cache
        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
        slots = slot_i64[valid_idx]
        pos = positions[valid_idx].to(torch.long)
        ape_rows = pos % self.compress_ratio
        kv_v = kv[valid_idx]                                    # [V, D]
        score_v = score[valid_idx]                              # [V, D]
        ape_v = ape.index_select(0, ape_rows)                   # [V, D]
        # bf16 add matches the kernel's fp32-promoted add + bf16 store —
        # numerically equivalent for the simple sum of two bf16 values
        # within the rounding behaviour of round-to-nearest-even.
        score_plus_ape = (score_v.to(torch.float32) +
                          ape_v.to(torch.float32)).to(torch.bfloat16)
        rows = torch.cat([kv_v, score_plus_ape], dim=-1)        # [V, 2D]
        # Use index_copy_ so duplicate slots (which can happen at -1-skipped
        # batches but not for valid slots in a single step) keep the last
        # write — same as the kernel.
        state_cache.index_copy_(0, slots, rows)
        return state_cache

    # ------------------------------------------------------------------
    # Grid / block heuristics
    # ------------------------------------------------------------------
    def auto_grid_dim(self, kv_dt: DTensor) -> GridDim:
        """One CTA per token; the CTA derives its ``t`` from
        ``task_metadata.token_offset``.
        """
        from ... import context as _ctx
        pk = _ctx.current_pk()
        n = kv_dt.dim(0)
        return (max(1, min(n, pk.num_workers)), 1, 1)

    def default_block_dim(self) -> BlockDim:
        """Default block dim — matches the architecture's
        ``WORKER_NUM_THREADS``.  The kernel itself runs on the first 128
        lanes (the registration function bakes ``NUM_THREADS=128`` into
        the template), so the trailing lanes simply exit at the top of
        the device function.
        """
        from ... import context as _ctx
        pk = _ctx.current_pk()
        return (128, 1, 1) if pk.target_cc < 90 else (256, 1, 1)

    # ------------------------------------------------------------------
    # MPK task registration
    # ------------------------------------------------------------------
    def compile(
        self,
        kv: DTensor,
        score: Union[torch.Tensor, DTensor],
        positions: Union[torch.Tensor, DTensor],
        slot_mapping: Union[torch.Tensor, DTensor],
        state_cache: Union[torch.Tensor, DTensor],
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``compressor_state_update_sm100`` task.

        Tensor contract:
          kv           : ``[T, head_dim]``                 bf16, IN.
          score        : ``[T, head_dim]``                 bf16, IN.
          positions    : ``[T]``                           int32, IN.
          slot_mapping : ``[T]``                           int32, IN.
          state_cache  : ``[num_slots, 2 * head_dim]``     bf16, IN/OUT.

        The ``ape`` learned parameter (registered on this module) is
        attached to the kernel automatically.  Returns the
        ``state_cache`` DTensor (post-update; same handle as the input).
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        prefix = self.prefix or "compressor_state_update_"

        # Validate kv shape.
        if kv.num_dims != 2:
            raise ValueError(
                f"kv must be a 2-D DTensor; got num_dims={kv.num_dims}"
            )
        if kv.dim(1) != self.head_dim:
            raise ValueError(
                f"kv.dim(1)={kv.dim(1)} does not match head_dim="
                f"{self.head_dim}"
            )
        T = kv.dim(0)

        def _attach_in(
            buf: Union[torch.Tensor, DTensor],
            expected_dtype_torch: torch.dtype,
            name: str,
        ) -> DTensor:
            if isinstance(buf, DTensor):
                return buf
            if isinstance(buf, torch.Tensor):
                if buf.dtype != expected_dtype_torch:
                    raise ValueError(
                        f"{name} must have dtype {expected_dtype_torch}; "
                        f"got {buf.dtype}"
                    )
                return pk.attach_input(buf, name=name)
            raise TypeError(
                f"{name} must be torch.Tensor or DTensor; got "
                f"{type(buf).__name__}"
            )

        score_dt = _attach_in(score, torch.bfloat16, f"{prefix}score")
        positions_dt = _attach_in(positions, torch.int32, f"{prefix}positions")
        slot_mapping_dt = _attach_in(
            slot_mapping, torch.int32, f"{prefix}slot_mapping"
        )
        # The learned ``ape`` is owned by this module.
        ape_dt = pk.attach_input(self.ape.data, name=f"{prefix}ape")

        # state_cache is an IN/OUT tensor; the runtime threads it through
        # ``output_ptrs[0]`` so the kernel performs an in-place write.
        if isinstance(state_cache, torch.Tensor):
            if state_cache.dtype != torch.bfloat16:
                raise ValueError(
                    f"state_cache must have dtype bfloat16; "
                    f"got {state_cache.dtype}"
                )
            state_cache_dt = pk.attach_input(
                state_cache, name=f"{prefix}state_cache"
            )
        elif isinstance(state_cache, DTensor):
            state_cache_dt = state_cache
        else:
            raise TypeError(
                f"state_cache must be torch.Tensor or DTensor; got "
                f"{type(state_cache).__name__}"
            )

        # Shape validation against the constructor configuration.
        if score_dt.num_dims != 2 or score_dt.dim(0) != T or \
                score_dt.dim(1) != self.head_dim:
            raise ValueError(
                f"score must be [T={T}, head_dim={self.head_dim}]; "
                f"got dims=({score_dt.dim(0)}, {score_dt.dim(1)})"
            )
        if positions_dt.num_dims != 1 or positions_dt.dim(0) != T:
            raise ValueError(
                f"positions must be [T={T}]; "
                f"got dim={positions_dt.dim(0)}"
            )
        if slot_mapping_dt.num_dims != 1 or slot_mapping_dt.dim(0) != T:
            raise ValueError(
                f"slot_mapping must be [T={T}]; "
                f"got dim={slot_mapping_dt.dim(0)}"
            )
        if ape_dt.num_dims != 2 or ape_dt.dim(0) != self.compress_ratio or \
                ape_dt.dim(1) != self.head_dim:
            raise ValueError(
                f"ape must be [compress_ratio={self.compress_ratio}, "
                f"head_dim={self.head_dim}]; "
                f"got ({ape_dt.dim(0)}, {ape_dt.dim(1)})"
            )
        if state_cache_dt.num_dims != 2 or \
                state_cache_dt.dim(1) != 2 * self.head_dim:
            raise ValueError(
                f"state_cache must be [num_slots, 2*head_dim="
                f"{2 * self.head_dim}]; "
                f"got ({state_cache_dt.dim(0)}, {state_cache_dt.dim(1)})"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(kv)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # All inputs are addressed via ``task_metadata.token_offset``
        # inside the kernel (the kernel does its own ``row = base + t * D``
        # arithmetic), so every map dim is (-1, -1, -1) — same pattern as
        # mhc_pre_sm100, hash_route_lookup_sm100, mla_v4_q_kv_rmsnorm_sm100,
        # and inv_rope_fp8_quant_o_sm100.
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(kv, (-1, -1, -1), -1, True)
        tb_graph.new_input(score_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(ape_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(positions_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(slot_mapping_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(state_cache_dt, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [kv, score_dt, ape_dt, positions_dt, slot_mapping_dt,
             state_cache_dt],
            tb_graph,
        )
        # params: [head_dim, compress_ratio, overlap (0/1)]
        pk.kn_graph.register_task(
            tb_graph,
            "compressor_state_update_sm100",
            [self.head_dim, self.compress_ratio, int(self.overlap)],
        )
        return state_cache_dt
