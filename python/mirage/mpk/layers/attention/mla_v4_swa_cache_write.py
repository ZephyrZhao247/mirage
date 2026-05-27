"""DeepSeek V4-Flash MLA SWA cache write-back catalog module (Wave 3.5).

Wave 3.5 follow-up to commit 83c38ecc: provides the Python catalog
scaffold for the SWA cache write task. The CUDA kernel itself
(``tasks/blackwell/mla_v4_swa_cache_write_sm100.cuh``) and the
matching ``register_mla_v4_swa_cache_write_sm100_task`` entry in
``src/kernel/task_register.cc`` are NOT landed in this commit -- the
Python module is in catalog form so the compile path can be wired
incrementally. ``compile()`` raises :class:`NotImplementedError` until
the kernel lands; tests should mark this path SKIP with the same skip
reason as :class:`DeepseekV4Block.compile`.

Math (one CTA per token):

    for t in range(T):
        pos     = positions[t]
        slot    = pos % window_size
        swa_cache[batch, slot, :] = kv_in[t, :]

Reserved TaskType: ``TASK_MLA_V4_SWA_CACHE_WRITE_SM100 = 364``
(next free slot after the Wave-2 allocation 350-363 in
``include/mirage/persistent_kernel/runtime_header.h``).
"""
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["MLAv4SWACacheWrite"]


class MLAv4SWACacheWrite(MPKModule):
    """Wave 3.5 scaffold for the per-token SWA cache write.

    Constructor args:
      head_dim     : trailing width of the cache row (= MLA latent
                      ``head_dim``, e.g. 512 in V4-Flash).
      window_size  : SWA cache length per batch (typically ``sliding_window``
                      from the config, e.g. 128).
      prefix       : MPK kernel-tensor name prefix.
    """

    def __init__(
        self,
        head_dim: int,
        window_size: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0; got {head_dim}")
        if window_size <= 0:
            raise ValueError(f"window_size must be > 0; got {window_size}")
        self.head_dim = head_dim
        self.window_size = window_size

    # ------------------------------------------------------------------
    # PyTorch reference -- a copy that the test harness can use to verify
    # the kernel against once it lands.
    # ------------------------------------------------------------------
    def forward(
        self,
        kv_in: torch.Tensor,
        positions: torch.Tensor,
        swa_cache: torch.Tensor,
        batch_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Write each token's freshly-computed K/V row into its SWA slot.

        Args:
          kv_in     : ``[T, head_dim]`` bf16 -- the K/V row(s) to write.
          positions : ``[T]`` int32 -- absolute positions for each token.
          swa_cache : ``[B, window_size, head_dim]`` bf16 -- in-place output.
                       For the layer-test (B=1) the input is a flattened
                       2-D ``[window_size, head_dim]`` view; we accept both.
          batch_ids : ``[T]`` int32 or None -- batch index per token; if
                       None, all tokens map to batch 0.

        Returns ``swa_cache`` with rows written in-place at
        ``[batch, pos % window_size, :]`` for each token.
        """
        T = kv_in.size(0)
        if swa_cache.dim() == 2:
            swa_cache_3d = swa_cache.unsqueeze(0)  # [1, W, D]
        elif swa_cache.dim() == 3:
            swa_cache_3d = swa_cache
        else:
            raise ValueError(
                "swa_cache must be 2-D [W, D] or 3-D [B, W, D]; got "
                f"shape {tuple(swa_cache.shape)}"
            )
        B, W, D = swa_cache_3d.shape
        if D != self.head_dim:
            raise ValueError(
                f"swa_cache trailing dim {D} != head_dim={self.head_dim}"
            )
        if W != self.window_size:
            raise ValueError(
                f"swa_cache window {W} != window_size={self.window_size}"
            )
        if batch_ids is None:
            batch_ids = torch.zeros(T, dtype=torch.int32, device=kv_in.device)
        for t in range(T):
            b = int(batch_ids[t].item())
            pos = int(positions[t].item())
            slot = pos % W
            swa_cache_3d[b, slot, :] = kv_in[t, :]
        return swa_cache

    # ------------------------------------------------------------------
    # Grid / block heuristics (matching the planned kernel signature).
    # ------------------------------------------------------------------
    def auto_grid_dim(self, kv_in_dt: DTensor) -> GridDim:
        """One CTA per token (matches the planned kernel)."""
        from ... import context as _ctx
        pk = _ctx.current_pk()
        n = kv_in_dt.dim(0)
        return (max(1, min(n, pk.num_workers)), 1, 1)

    def default_block_dim(self) -> BlockDim:
        from ... import context as _ctx
        pk = _ctx.current_pk()
        return (128, 1, 1) if pk.target_cc < 90 else (256, 1, 1)

    # ------------------------------------------------------------------
    # MPK task registration -- DEFERRED.
    # ------------------------------------------------------------------
    def compile(
        self,
        kv_in: DTensor,
        positions: Union[torch.Tensor, DTensor],
        swa_cache: Union[torch.Tensor, DTensor],
        *,
        batch_ids: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register ``mla_v4_swa_cache_write_sm100``.

        Wave 3.5 status: the CUDA kernel and C++ task_register entry are
        NOT landed in this commit. Calling compile() raises
        :class:`NotImplementedError` with a precise pointer to what is
        missing. The Python catalog signature is fixed so that downstream
        callers (``DeepseekV4Block.compile``) can be authored against it.
        """
        raise NotImplementedError(
            "MLAv4SWACacheWrite.compile() is a Wave-3.5 scaffold. "
            "The supporting CUDA kernel "
            "(tasks/blackwell/mla_v4_swa_cache_write_sm100.cuh) and "
            "task_register entry "
            "(register_mla_v4_swa_cache_write_sm100_task in "
            "src/kernel/task_register.cc) plus the TaskType slot "
            "TASK_MLA_V4_SWA_CACHE_WRITE_SM100 = 364 must land before "
            "this catalog module can register a real task. See "
            "docs/mpk/mpk_runtime_analysis.md for the kernel skeleton "
            "template and DeepseekV4Block.compile for the call site."
        )
