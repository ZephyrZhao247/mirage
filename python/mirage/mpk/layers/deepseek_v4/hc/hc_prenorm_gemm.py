"""V4-Flash ``hc_prenorm_gemm`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/hc_prenorm_gemm_tilelang.md``.

Decision: **NEW**.

Rationale
---------
The vLLM TileLang kernel computes two outputs in a SINGLE K-traversal of
``x``:

* ``gemm_out[s, t, j] = sum_{k in K_s} x[t, k].float() * fn[j, k]`` -- the
  split-K partial of ``x @ fn.T``.
* ``sqrsum[s, t] = sum_{k in K_s} x[t, k].float()**2`` (only the j=0
  output tile writes -- one sqrsum write per (split, token)).

There is no existing MPK task with this dual-output signature, and the
downstream consumer (``mhc_pre_big_fuse_*``) hard-requires the
``[n_splits, T, HC_MULT3]`` fp32 + ``[n_splits, T]`` fp32 contract.

So this is a NEW kernel:
* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/hc_prenorm_gemm_v4_sm100.cuh``
* Task name: ``hc_prenorm_gemm_v4_sm100``
* Enum slot: ``TASK_HC_PRENORM_GEMM_V4_SM100 = 358``

The Blackwell impl is intentionally NAIVE (per the standing "naive first,
perf later" rule):

* One CTA per token. ``grid = (num_tokens, 1, 1)``, ``block = (256,1,1)``.
* fp32 accumulation, bf16 ``x``, fp32 ``fn``, fp32 outputs.
* No TMA / no UMMA / no warp-specialization / no split-K orchestration.
  ``n_splits`` is fixed to 1 (matches the TileLang fallback's default and
  the consumer transparently reduces over the leading split dim).
* For each output index ``j in [0, HC_MULT3)``, accumulate the K-axis
  dot product in fp32 with a thread-strided loop, then warp + cross-warp
  reduce. The j=0 pass also accumulates the per-token sum-of-squares to
  share the K read.

Audit
-----
* dtype: ``x`` bf16, ``fn`` fp32. Outputs ``gemm_out``/``sqrsum`` both
  fp32 -- matches the contract that ``mhc_pre_big_fuse_*`` reads
  (audit's "gemm_out_mul bf16->fp32 type-pun" blocker: outputs are
  fp32 here, NOT bf16).
* layout: all row-major contiguous.
  ``x: [T, K]``, ``fn: [HC_MULT3, K]``, ``gemm_out: [1, T, HC_MULT3]``,
  ``sqrsum: [1, T]``. Output rank is 3 for ``gemm_out`` and 2 for
  ``sqrsum`` (audit's "2D vs 3D rank" blocker: ``gemm_out`` is 3D as
  the consumer expects).
* multi-batch: partitioned on dim 0 (token axis). Multi-batch by
  construction; test uses ``max_num_batched_requests = 4``.
* ``forward()``: faithful PyTorch fp32 reference, output cast back to
  fp32 to match the kernel boundary (no bf16 round-trip).

This kernel is CONSUMED by ``mhc_pre_big_fuse`` / ``mhc_pre_big_fuse_with_norm``
(Wave-2A); the output dtype/rank must match what those consumers read.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor

__all__ = ["V4HcPrenormGemm"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class V4HcPrenormGemm(MPKModule):
    """V4-Flash hc_prenorm_gemm naive Blackwell kernel.

    Owns one ``nn.Parameter``:

      * ``fn: [HC_MULT3, HC_MULT * HIDDEN]`` fp32 -- the HC mixing
        weights (``hc_attn_fn`` / ``hc_ffn_fn`` in the reference).

    Constructor args:

      * ``hidden_size`` -- per-stream feature dim ``HIDDEN`` (V4-Flash: 4096).
      * ``hc_mult``     -- HC multiplier (V4-Flash: 4); the flattened K
                           dim is ``HC_MULT * HIDDEN``.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if hidden_size <= 0:
            raise ValueError(
                f"V4HcPrenormGemm hidden_size must be positive; "
                f"got {hidden_size}"
            )
        if hc_mult <= 0:
            raise ValueError(
                f"V4HcPrenormGemm hc_mult must be positive; "
                f"got {hc_mult}"
            )
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.hc_mult3 = hc_mult * (2 + hc_mult)  # 24 for hc_mult=4
        self.K = hc_mult * hidden_size           # 16384 for V4-Flash
        # fn is fp32 in the reference (hc_attn_fn / hc_ffn_fn).
        self.fn = nn.Parameter(
            torch.empty(self.hc_mult3, self.K, dtype=torch.float32)
        )

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,  # bf16 [T, K]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eager reference matching the vLLM TileLang kernel.

        Returns ``(gemm_out, sqrsum)`` with shapes
        ``([1, T, HC_MULT3], [1, T])`` and dtype fp32 -- the n_splits=1
        contract expected by the downstream ``mhc_pre_big_fuse_*``.
        """
        assert x.dim() == 2, f"x must be 2-D [T, K], got shape {tuple(x.shape)}"
        assert x.shape[1] == self.K, (
            f"x last dim ({x.shape[1]}) != HC_MULT * HIDDEN ({self.K})"
        )
        T = x.shape[0]

        x_fp32 = x.to(torch.float32)
        # GEMM: x @ fn.T -> [T, HC_MULT3]
        gemm_out = x_fp32 @ self.fn.to(torch.float32).T   # [T, HC_MULT3]
        sqrsum = x_fp32.pow(2).sum(dim=-1)                 # [T]

        # Lift to the n_splits=1 contract.
        gemm_out = gemm_out.view(1, T, self.hc_mult3)
        sqrsum = sqrsum.view(1, T)
        return gemm_out, sqrsum

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, x_dt: DTensor) -> GridDim:
        """One CTA per token (capped at ``num_workers``).

        Each CTA computes one token's HC_MULT3 outputs + sqrsum in
        a sequential output-index loop; ``grid.x`` ranges over tokens.
        """
        pk = current_pk()
        num_tokens = x_dt.dim(0)
        return (max(1, min(num_tokens, pk.num_workers)), 1, 1)

    # ------------------------------------------------------------------
    # MPK registration
    # ------------------------------------------------------------------
    def compile(
        self,
        x_dt: DTensor,
        *,
        gemm_out: Optional[torch.Tensor] = None,
        sqrsum: Optional[torch.Tensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register one ``hc_prenorm_gemm_v4_sm100`` task.

        Tensor contract:
          x        : (T, HC_MULT * HIDDEN) bf16, row-major contiguous.
          fn (auto): (HC_MULT3, HC_MULT * HIDDEN) fp32 nn.Parameter.
          gemm_out : (1, T, HC_MULT3) fp32, allocated if None.
          sqrsum   : (1, T)           fp32, allocated if None.

        Notes: n_splits is fixed to 1 in the naive port. block_dim
        defaults to (256, 1, 1) on Blackwell. HC_MULT and HIDDEN are
        derived from the bgraph tensor shapes at codegen time.
        """
        from .....core import CyTBGraph, float32 as _mi_f32
        from .....kernel import TBGraph

        pk = current_pk()

        if x_dt.num_dims != 2:
            raise ValueError(
                f"V4HcPrenormGemm: x_dt must be 2-D [T, K]; got "
                f"num_dims={x_dt.num_dims}"
            )
        if x_dt.dim(1) != self.K:
            raise ValueError(
                f"V4HcPrenormGemm: x_dt.dim(1)={x_dt.dim(1)} != "
                f"HC_MULT * HIDDEN ({self.K})"
            )

        T = x_dt.dim(0)

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x_dt)
        if block_dim is None:
            block_dim = self.default_block_dim()

        fn_dt = pk.attach_input(self.fn.data, name=f"{self.prefix}fn")

        if gemm_out is None:
            gemm_out_dt = pk.new_tensor(
                dims=(1, T, self.hc_mult3),
                dtype=_mi_f32,
                name=f"{self.prefix}gemm_out",
            )
        elif isinstance(gemm_out, torch.Tensor):
            gemm_out_dt = pk.attach_input(
                gemm_out, name=f"{self.prefix}gemm_out"
            )
        elif isinstance(gemm_out, DTensor):
            gemm_out_dt = gemm_out
        else:
            raise TypeError(
                "V4HcPrenormGemm.compile gemm_out must be None, "
                f"torch.Tensor, or DTensor; got {type(gemm_out).__name__}"
            )

        if sqrsum is None:
            sqrsum_dt = pk.new_tensor(
                dims=(1, T),
                dtype=_mi_f32,
                name=f"{self.prefix}sqrsum",
            )
        elif isinstance(sqrsum, torch.Tensor):
            sqrsum_dt = pk.attach_input(sqrsum, name=f"{self.prefix}sqrsum")
        elif isinstance(sqrsum, DTensor):
            sqrsum_dt = sqrsum
        else:
            raise TypeError(
                "V4HcPrenormGemm.compile sqrsum must be None, "
                f"torch.Tensor, or DTensor; got {type(sqrsum).__name__}"
            )

        # ----- TBGraph construction --------------------------------------
        # x and outputs partition on the token axis (dim 0 for x and
        # sqrsum, dim 1 for the [1, T, HC_MULT3] gemm_out). fn is
        # broadcast. The runtime preoffsets pointers so each task sees
        # its per-token slice.
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(fn_dt, (-1, -1, -1), 0, True)
        # gemm_out has shape [1, T, HC_MULT3]; partition on dim 1 (the
        # token axis). The split-K dim is degenerate (size 1).
        tb_graph.new_input(gemm_out_dt, (1, -1, -1), 1, True)
        # sqrsum has shape [1, T]; partition on dim 1.
        tb_graph.new_input(sqrsum_dt, (1, -1, -1), 1, True)
        pk.kn_graph.customized(
            [x_dt, fn_dt, gemm_out_dt, sqrsum_dt], tb_graph
        )
        pk.kn_graph.register_task(tb_graph, "hc_prenorm_gemm_v4_sm100")

        return gemm_out_dt, sqrsum_dt
