"""DeepSeek V4-Flash MTP embed + hidden fuse (decomposed v1).

At the input of the MTPBlock, V4-Flash fuses the freshly-embedded next
token with the previous-layer HC hidden state:

    fused[t, j, d] = e_proj(e)[t, d]                       (broadcast over j)
                   + h_proj(h)[t, j, d]

Per ``docs/mpk/deepseek_v4/mtp.md`` Strategy A1, v1 is implemented as a
Python composite over existing catalog primitives — **no new CUDA**.

Strategy chosen here — **fused per-hc LinearWithResidual**:

For each ``j`` in ``0..hc-1``:

1. ``h_p_j = Linear(h_per_hc[j], h_proj.weight)`` — projects the j-th
   HC slice of ``h``.
2. ``fused_j = LinearWithResidual(e, e_proj.weight, residual=h_p_j)``
   — computes ``e_proj(e) + h_p_j`` in a single fused task.

``e`` and ``e_proj.weight`` are graph inputs (KN_INPUT_OP); reading them
from ``hc`` separate LWR tasks does **not** make them a "fork-producer"
in ``annotated_graph.cc``'s sense — the case-2/case-3 check only fires
on task layers. ``h_p_j`` has exactly one consumer (the j-th LWR), so
no fork edge is created downstream of any task either. Total task
count: ``2 * hc = 8`` at ``hc=4``.

Why not the doc's literal Strategy A1:

* The doc's A1 (replicate ``e_proj.weight`` to ``[hc*D, D]`` + a single
  broadcast Linear writing into a ``[T, hc*D]`` buffer + ``hc``
  ``LinearWithResidual`` calls that read per-hc residual slices of that
  same buffer) would require each per-hc residual to be a strided ``[T,
  D]`` view of the shared ``[T, hc*D]`` buffer. ``pk.attach_input``
  enforces a strict row-major (or column-major) layout, and a strided
  ``[T, D]`` slice with row stride ``hc*D`` satisfies neither, so the
  literal A1 plan would require either a new low-level attach API or
  per-hc dedicated output buffers — which collapses A1 into the
  per-hc-LWR pattern we use here.
* Additionally, A1's broadcast Linear has ``hc`` downstream LWR
  consumers (a fork), and each of those LWRs has two task-layer
  producers (the broadcast Linear's e_p AND the j-th h-Linear) — a
  join. That triggers ``annotated_graph.cc``'s "case 3" rejection
  (layer is both fork-producer and join-producer).

Weight memory: 1x for both ``e_proj.weight`` and ``h_proj.weight`` —
neither is replicated, so strictly less than A1's 4x e_proj footprint.
The ``hc`` redundant ``e_proj`` GEMMs (one per LWR) are bf16 GEMMs of
shape ``[T, D] @ [D, D].T = [T, D]`` — at typical V4-Flash sizes
(``T=8``, ``D=4096``, ``hc=4``) the redundant compute is ~16 MFLOPs vs.
the megakernel's 100+ GFLOPs of attention/MoE work, i.e. <1% overhead.
v2 will fuse the whole MTP input fuse into a single CUDA task anyway.
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .._base import MPKModule
from ...context import current_pk
from ....core import CyTBGraph, DTensor
from ....kernel import TBGraph


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


__all__ = ["MTPEmbedHiddenFuse"]


def _linear_grid_dim(out_features: int, num_workers: int) -> GridDim:
    """Mirror of ``layers.linear.linear._grid_x_for_out_features``."""
    if out_features % 96 == 0:
        gx = out_features // 96
    elif out_features % 64 == 0:
        gx = out_features // 64
    else:
        raise ValueError(
            f"MTPEmbedHiddenFuse: out_features={out_features} is not "
            "divisible by 96 or 64."
        )
    return (max(1, min(gx, int(num_workers))), 1, 1)


def _default_block_dim(target_cc: int) -> BlockDim:
    return (128, 1, 1) if target_cc < 90 else (256, 1, 1)


def _arch_linear_task_name(target_cc: int) -> str:
    if 100 <= target_cc < 120:
        return "linear_sm100"
    if 90 <= target_cc < 100:
        return "linear_swapAB_hopper"
    if 80 <= target_cc < 90:
        return "linear"
    raise RuntimeError(
        f"MTPEmbedHiddenFuse: unsupported compute capability {target_cc}."
    )


def _register_linear_task(pk, *, x, weight_dt, out_dt, block_dim):
    """Register a Linear task ``out = x @ weight.T`` with pre-attached
    weight and output DTensors. Mirrors :meth:`Linear.compile` but
    decouples weight attachment from task registration so the catalog
    module can attach each projection weight exactly once and reuse the
    handle across ``hc`` per-copy tasks.
    """
    out_features = weight_dt.dim(0)
    grid_dim = _linear_grid_dim(out_features, pk.num_workers)
    if block_dim is None:
        block_dim = _default_block_dim(pk.target_cc)

    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(x, (-1, -1, -1), 1, True)
    tb_graph.new_input(weight_dt, (0, -1, -1), 1, True)
    tb_graph.new_input(out_dt, (1, -1, -1), -1, True)
    pk.kn_graph.customized([x, weight_dt, out_dt], tb_graph)
    pk.kn_graph.register_task(tb_graph, _arch_linear_task_name(pk.target_cc))


def _arch_linear_with_residual_task_name(target_cc: int) -> str:
    if 100 <= target_cc < 120:
        return "linear_with_residual_sm100"
    if 90 <= target_cc < 100:
        return "linear_swapAB_with_residual_hopper"
    if 80 <= target_cc < 90:
        return "linear_with_residual"
    raise RuntimeError(
        f"MTPEmbedHiddenFuse: unsupported compute capability {target_cc}."
    )


def _register_linear_with_residual_task(
    pk, *, x, weight_dt, residual_dt, out_dt, block_dim,
):
    """Register a LinearWithResidual task ``out = (x @ weight.T) + residual``
    with pre-attached weight, residual, and output DTensors.

    Mirrors :meth:`LinearWithResidual.compile`; lets the catalog module
    attach the projection weight exactly once and reuse the handle.
    """
    out_features = weight_dt.dim(0)
    atom = 128 if pk.target_cc >= 100 else 64
    if out_features % atom != 0:
        raise ValueError(
            f"_register_linear_with_residual_task: out_features="
            f"{out_features} is not divisible by the kernel output atom "
            f"size {atom} (target_cc={pk.target_cc})."
        )
    gx = max(1, min(out_features // atom, int(pk.num_workers)))
    grid_dim = (gx, 1, 1)
    if block_dim is None:
        block_dim = _default_block_dim(pk.target_cc)

    tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
    tb_graph.new_input(x, (-1, -1, -1), 1, True)
    tb_graph.new_input(weight_dt, (0, -1, -1), 1, True)
    tb_graph.new_input(residual_dt, (1, -1, -1), -1, True)
    tb_graph.new_input(out_dt, (1, -1, -1), -1, True)
    pk.kn_graph.customized([x, weight_dt, residual_dt, out_dt], tb_graph)

    enable_residual = 1
    if pk.world_size > 1 and pk.mpi_rank != 0:
        enable_residual = 0
    pk.kn_graph.register_task(
        tb_graph,
        _arch_linear_with_residual_task_name(pk.target_cc),
        [enable_residual],
    )


class MTPEmbedHiddenFuse(MPKModule):
    """V4-Flash MTPBlock embed+hidden fuse (decomposed v1).

    Owns two ``nn.Linear``-equivalent projection weights:

    * ``e_proj.weight``: ``[D, D]`` bf16 — projection of the freshly
      embedded next token. Applied once; the ``[T, D]`` result is
      broadcast-added to each of the ``hc`` HC copies.
    * ``h_proj.weight``: ``[D, D]`` bf16 — projection of the previous-layer
      HC hidden state. Applied independently per HC copy.

    Args:
        hidden_size: Per-HC-copy hidden dim ``D``. Multiple of 96 or 64
            (Linear kernel tile constraint).
        hc_mult: HC multiplicity (default 4).
        prefix: MPK kernel-tensor / state_dict name prefix.

    Forward signature (eager PyTorch reference):
        ``forward(e, h)`` with ``e: [T, D]``, ``h: [T, hc, D]`` →
        ``[T, hc, D]``.

    Compile signature:
        ``compile(e, h_per_hc, *, fused_per_hc=None)`` where ``e`` is a
        2-D DTensor ``[T, D]`` and ``h_per_hc`` is a list of ``hc`` 2-D
        DTensors ``[T, D]`` (the j-th HC slice of ``h``). The caller
        pre-attaches each slice; returns a list of ``hc`` ``[T, D]``
        DTensors representing the per-HC fused outputs.

    The decomposed-v1 split keeps weight loading vLLM-compatible:
    ``e_proj.weight`` and ``h_proj.weight`` are vanilla submodule
    parameters under the standard state_dict prefixes.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int = 4,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult

        # State_dict-loadable parameters. Vanilla nn.Linear so HF
        # checkpoints with keys "e_proj.weight" / "h_proj.weight" load
        # without any custom routing.
        self.e_proj = nn.Linear(
            hidden_size, hidden_size, bias=False, dtype=torch.bfloat16
        )
        self.h_proj = nn.Linear(
            hidden_size, hidden_size, bias=False, dtype=torch.bfloat16
        )

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(self, e: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """Eager PyTorch oracle.

        Args:
            e: ``[T, D]`` bf16.
            h: ``[T, hc, D]`` bf16.

        Returns:
            ``[T, hc, D]`` bf16 tensor equal to
            ``e_proj(e).unsqueeze(1) + h_proj(h)``.
        """
        e_p = F.linear(e, self.e_proj.weight)          # [T, D]
        h_p = F.linear(h, self.h_proj.weight)          # [T, hc, D]
        return e_p.unsqueeze(1) + h_p                  # [T, hc, D]

    # ------------------------------------------------------------------
    # Grid heuristic — delegates to the underlying Linear (per-output-dim).
    # ------------------------------------------------------------------
    def auto_grid_dim(self, *_args, **_kwargs) -> GridDim:
        """Return the Linear's grid for ``D``-wide outputs (broadcast-e
        and per-hc h shares the same out_features). ``compile()`` ignores
        this on the elementwise-add side, which picks its own row-stripe
        grid."""
        pk = current_pk()
        return _linear_grid_dim(self.hidden_size, pk.num_workers)

    # ------------------------------------------------------------------
    # MPK task registration.
    # ------------------------------------------------------------------
    def compile(
        self,
        e: DTensor,
        h_per_hc: List[DTensor],
        *,
        fused_per_hc: Optional[List[Union[torch.Tensor, DTensor]]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> List[DTensor]:
        """Register the decomposed embed+hidden-fuse pipeline.

        Tensor contract:
          e:           (T, D) bf16 DTensor — freshly-embedded next token.
          h_per_hc:    list of ``hc_mult`` (T, D) bf16 DTensors — the j-th
                       HC slice of the previous-layer hidden state, i.e.
                       ``h[:, j, :].contiguous()`` for ``j`` in
                       ``0..hc-1``. The caller is responsible for
                       attaching each slice (``pk.attach_input`` enforces
                       a contiguous-row-major layout, so a strided 3-D
                       view does not satisfy that contract — slices must
                       be ``.contiguous()`` first).
          fused_per_hc: optional list of ``hc_mult`` outputs, each a
                       ``torch.Tensor`` (for host readback in test mode)
                       or a ``DTensor`` (caller-provided buffer). If
                       ``None``, ``hc_mult`` new DTensors are allocated.

        Returns:
            ``List[DTensor]`` of length ``hc_mult``; each element is a
            ``[T, D]`` DTensor holding ``e_proj(e) + h_proj(h[:, j, :])``
            for the corresponding HC copy.
        """
        pk = current_pk()

        # ---- Validate inputs -----------------------------------------
        if e.num_dims != 2:
            raise ValueError(
                f"MTPEmbedHiddenFuse: e must be a 2-D DTensor [T, D]; got "
                f"num_dims={e.num_dims}"
            )
        T = e.dim(0)
        D = e.dim(1)
        if D != self.hidden_size:
            raise ValueError(
                f"MTPEmbedHiddenFuse: e.dim(1)={D} != hidden_size="
                f"{self.hidden_size}"
            )
        hc = self.hc_mult
        if not isinstance(h_per_hc, (list, tuple)) or len(h_per_hc) != hc:
            raise ValueError(
                f"MTPEmbedHiddenFuse: h_per_hc must be a list of {hc} "
                f"DTensors (one per HC copy); got "
                f"{type(h_per_hc).__name__} of length "
                f"{len(h_per_hc) if hasattr(h_per_hc, '__len__') else 'N/A'}"
            )
        for j, h_j in enumerate(h_per_hc):
            if h_j.num_dims != 2 or h_j.dim(0) != T or h_j.dim(1) != D:
                raise ValueError(
                    f"MTPEmbedHiddenFuse: h_per_hc[{j}] must be [T={T}, "
                    f"D={D}]; got [{h_j.dim(0)}, {h_j.dim(1)}]"
                )

        if fused_per_hc is not None:
            if not isinstance(fused_per_hc, (list, tuple)) or \
               len(fused_per_hc) != hc:
                raise ValueError(
                    f"MTPEmbedHiddenFuse: fused_per_hc must be a list of "
                    f"{hc} buffers (one per HC copy); got "
                    f"{type(fused_per_hc).__name__}"
                )

        prefix = self.prefix or "mtp_embed_hidden_fuse_"
        if block_dim is None:
            block_dim = _default_block_dim(pk.target_cc)

        # ---- Attach projection weights exactly once ------------------
        # Reading these graph inputs from `hc` separate tasks does NOT
        # create a layer-level fork — KN_INPUT_OPs are excluded from the
        # case-2/3 check in `src/kernel/annotated_graph.cc`.
        e_proj_dt = pk.attach_input(
            self.e_proj.weight, name=f"{prefix}e_proj_weight"
        )
        h_proj_dt = pk.attach_input(
            self.h_proj.weight, name=f"{prefix}h_proj_weight"
        )

        # ---- Per-hc: h-Linear → LinearWithResidual(e, e_proj, +h_p) --
        out_dts: List[DTensor] = []
        for j in range(hc):
            # h-side Linear: h_p_j = h_per_hc[j] @ h_proj.weight.T  → [T, D]
            h_p_j_dt = pk.new_tensor(
                dims=(T, D),
                dtype=e.dtype,
                name=f"{prefix}h_p_{j}",
            )
            _register_linear_task(
                pk,
                x=h_per_hc[j],
                weight_dt=h_proj_dt,
                out_dt=h_p_j_dt,
                block_dim=block_dim,
            )

            # e-side Linear-with-residual: fused_j = e @ e_proj.T + h_p_j
            if fused_per_hc is None:
                fused_j_dt = pk.new_tensor(
                    dims=(T, D),
                    dtype=e.dtype,
                    name=f"{prefix}fused_hc{j}",
                )
            elif isinstance(fused_per_hc[j], torch.Tensor):
                fused_j_dt = pk.attach_input(
                    fused_per_hc[j], name=f"{prefix}fused_hc{j}",
                )
            elif isinstance(fused_per_hc[j], DTensor):
                fused_j_dt = fused_per_hc[j]
            else:
                raise TypeError(
                    f"MTPEmbedHiddenFuse: fused_per_hc[{j}] must be None, "
                    "torch.Tensor, or DTensor; got "
                    f"{type(fused_per_hc[j]).__name__}"
                )

            _register_linear_with_residual_task(
                pk,
                x=e,
                weight_dt=e_proj_dt,
                residual_dt=h_p_j_dt,
                out_dt=fused_j_dt,
                block_dim=block_dim,
            )
            out_dts.append(fused_j_dt)

        return out_dts
