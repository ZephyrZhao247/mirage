"""V4-Flash ``hc_prenorm_gemm_block_m`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/hc_prenorm_gemm_block_m_tilelang.md``.

Decision: **NEW**.

Rationale
---------
This is the M-blocked Class-A locked-alternative of :class:`V4HcPrenormGemm`
(spec #3). The upstream TileLang variant uses ``block_m=2`` to amortize
the ``fn`` reads across 2 tokens per CTA when ``T >= 1024``. At the
n_splits=1 contract (always 1 in the wrapper's large-T branch) the
OUTPUTS are byte-identical to the per-token variant:

* ``gemm_out: [1, T, HC_MULT3]`` fp32
* ``sqrsum:   [1, T]``           fp32

For the naive port we delegate to the SAME ``__device__`` impl as
:class:`V4HcPrenormGemm` (see
``include/mirage/persistent_kernel/tasks/blackwell/hc_prenorm_gemm_block_m_v4_sm100.cuh``,
which is a thin shim over ``hc_prenorm_gemm_v4_sm100_impl``). The
distinction is preserved at the catalog/task-name level so consumers
that branch on the kernel variant stay verbatim.

The Blackwell impl is intentionally naive: same partitioning, same
loop structure, same outputs as :class:`V4HcPrenormGemm`.

CUDA header: ``hc_prenorm_gemm_block_m_v4_sm100.cuh``.
Task name:   ``hc_prenorm_gemm_block_m_v4_sm100``.
Enum slot:   ``TASK_HC_PRENORM_GEMM_BLOCK_M_V4_SM100 = 359``.

Audit
-----
* dtype/layout/multi-batch/forward: identical to :class:`V4HcPrenormGemm`
  (this class is a thin task-name override).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from ....context import current_pk
from .....core import DTensor

from .hc_prenorm_gemm import V4HcPrenormGemm

__all__ = ["V4HcPrenormGemmBlockM"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class V4HcPrenormGemmBlockM(V4HcPrenormGemm):
    """V4-Flash hc_prenorm_gemm_block_m naive Blackwell kernel.

    Same I/O contract and the same naive backend as
    :class:`V4HcPrenormGemm`; differs only in the registered task name
    (``hc_prenorm_gemm_block_m_v4_sm100``).
    """

    def compile(
        self,
        x_dt: DTensor,
        *,
        gemm_out: Optional[torch.Tensor] = None,
        sqrsum: Optional[torch.Tensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register one ``hc_prenorm_gemm_block_m_v4_sm100`` task.

        Tensor contract: identical to :class:`V4HcPrenormGemm.compile`.
        """
        from .....core import CyTBGraph, float32 as _mi_f32
        from .....kernel import TBGraph

        pk = current_pk()

        if x_dt.num_dims != 2:
            raise ValueError(
                f"V4HcPrenormGemmBlockM: x_dt must be 2-D [T, K]; got "
                f"num_dims={x_dt.num_dims}"
            )
        if x_dt.dim(1) != self.K:
            raise ValueError(
                f"V4HcPrenormGemmBlockM: x_dt.dim(1)={x_dt.dim(1)} != "
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
                "V4HcPrenormGemmBlockM.compile gemm_out must be None, "
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
                "V4HcPrenormGemmBlockM.compile sqrsum must be None, "
                f"torch.Tensor, or DTensor; got {type(sqrsum).__name__}"
            )

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(x_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(fn_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(gemm_out_dt, (1, -1, -1), 1, True)
        tb_graph.new_input(sqrsum_dt, (1, -1, -1), 1, True)
        pk.kn_graph.customized(
            [x_dt, fn_dt, gemm_out_dt, sqrsum_dt], tb_graph
        )
        pk.kn_graph.register_task(tb_graph, "hc_prenorm_gemm_block_m_v4_sm100")

        return gemm_out_dt, sqrsum_dt
