"""V4-Flash ``hc_head_fuse`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/hc_head_fuse_tilelang.md``.

Decision: **NEW**.

Rationale
---------
This is the **terminal mHC kernel** -- collapses
``[T, HC_MULT, HIDDEN]`` -> ``[T, HIDDEN]`` for the bf16 hidden state
fed into the final RMSNorm + lm_head. The vLLM TileLang kernel fuses:

1. Flatten residual to ``[T, HC_MULT*HIDDEN]`` and compute per-token
   sum-of-squares + the ``hc_mult`` dot-products
   ``mixes[m] = sum_{c,h} residual[t,c,h] * fn[m, c*HIDDEN + h]``.
2. ``rsqrt = rsqrt(sqrsum / (HC_MULT*HIDDEN) + rms_eps)``.
3. ``pre_mix[m] = sigmoid(mixes[m] * rsqrt * hc_scale[0] + hc_base[m])
                    + hc_eps``.
4. ``out[t, h] = sum_c pre_mix[c] * residual[t, c, h]``, stored bf16.

There is no existing MPK task with this multi-output reduction +
sigmoid-gated weighted-sum pattern; the four constituent ops would be
4 separate launches with a HBM round-trip per launch.

So this is a NEW kernel:

* CUDA header: ``hc_head_fuse_v4_sm100.cuh``
* Task name:   ``hc_head_fuse_v4_sm100``
* Enum slot:   ``TASK_HC_HEAD_FUSE_V4_SM100 = 357``

The Blackwell impl is intentionally NAIVE:

* One CTA per token. ``grid = (num_tokens, 1, 1)``, ``block = (256,1,1)``.
* Pass 1: thread-strided sum across the full ``[HC_MULT, HIDDEN]`` tile
  for sqrsum + HC_MULT projections. Output reduced via warp + cross-warp
  shfl_xor.
* Pass 2: thread-strided over hidden h; for each h compute the
  hc-weighted sum with the broadcast ``pre_mix[HC_MULT]`` from pass 1.
* No TMA / no UMMA / no warp-specialization / no pipelining.

Audit
-----
* dtype: ``residual`` / ``out`` bf16; ``fn`` / ``hc_scale`` / ``hc_base``
  fp32 (matches the reference's ``hc_head_fn``, ``hc_scale``, ``hc_base``
  parameter dtypes). No silent cast.
* layout: row-major contiguous everywhere.
  ``residual: [T, HC_MULT, HIDDEN]``, ``fn: [HC_MULT, HC_MULT*HIDDEN]``,
  ``hc_scale: [1]``, ``hc_base: [HC_MULT]``, ``out: [T, HIDDEN]``.
* multi-batch: partitioned on dim 0 of residual / out (token axis).
  Multi-batch from day 1; test uses ``max_num_batched_requests = 4``.
* ``forward()``: faithful PyTorch reference in fp32 with bf16 round-out
  (matches the reference at ``model.py:729-736``).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor

__all__ = ["V4HcHeadFuse"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class V4HcHeadFuse(MPKModule):
    """V4-Flash hc_head_fuse naive Blackwell kernel.

    Owns three ``nn.Parameter`` weights:

      * ``fn:        [HC_MULT, HC_MULT * HIDDEN]`` fp32 -- hc_head_fn.
      * ``hc_scale:  [1]``                         fp32 -- scalar gate scale.
      * ``hc_base:   [HC_MULT]``                   fp32 -- per-mix bias.

    Constructor args:

      * ``hidden_size`` -- per-stream feature dim ``HIDDEN`` (V4-Flash: 4096).
      * ``hc_mult``     -- HC multiplier (V4-Flash: 4).
      * ``rms_eps``     -- RMSNorm epsilon; forwarded to ``forward()``
                           and hard-coded in codegen to 1e-6f.
      * ``hc_eps``      -- post-sigmoid floor; same plumbing as rms_eps.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int,
        rms_eps: float = 1e-6,
        hc_eps: float = 1e-6,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if hidden_size <= 0:
            raise ValueError(
                f"V4HcHeadFuse hidden_size must be positive; "
                f"got {hidden_size}"
            )
        if hc_mult <= 0:
            raise ValueError(
                f"V4HcHeadFuse hc_mult must be positive; got {hc_mult}"
            )
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.K = hc_mult * hidden_size
        self.rms_eps = rms_eps
        self.hc_eps = hc_eps
        # All weights fp32 -- matches the reference's hc_head_fn /
        # hc_scale / hc_base dtypes.
        self.fn = nn.Parameter(
            torch.empty(hc_mult, self.K, dtype=torch.float32)
        )
        self.hc_scale = nn.Parameter(torch.empty(1, dtype=torch.float32))
        self.hc_base = nn.Parameter(torch.empty(hc_mult, dtype=torch.float32))

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        residual: torch.Tensor,  # bf16 [T, HC_MULT, HIDDEN]
    ) -> torch.Tensor:
        """Eager reference matching ParallelHead.hc_head.

        Returns ``out`` of shape ``[T, HIDDEN]`` bf16.
        """
        assert residual.dim() == 3, (
            f"residual must be 3-D [T, HC_MULT, HIDDEN]; got shape "
            f"{tuple(residual.shape)}"
        )
        T = residual.shape[0]
        assert residual.shape == (T, self.hc_mult, self.hidden_size), (
            f"residual shape {tuple(residual.shape)} != "
            f"({T}, {self.hc_mult}, {self.hidden_size})"
        )
        in_dtype = residual.dtype

        # Flatten across (hc, h). fp32 reduction throughout.
        x = residual.to(torch.float32).reshape(T, self.K)            # [T, K]
        sqrsum = x.pow(2).sum(dim=-1)                                  # [T]
        rsqrt = torch.rsqrt(sqrsum / float(self.K) + self.rms_eps)     # [T]
        mixes = x @ self.fn.to(torch.float32).T                         # [T, HC_MULT]
        # Apply rsqrt + scale + base, sigmoid + eps.
        pre_mix = torch.sigmoid(
            mixes * rsqrt.unsqueeze(-1) * self.hc_scale.to(torch.float32)[0]
            + self.hc_base.to(torch.float32)
        ) + self.hc_eps                                                 # [T, HC_MULT]

        # Weighted sum across hc streams.
        x_3d = residual.to(torch.float32)                               # [T, HC_MULT, H]
        out = (pre_mix.unsqueeze(-1) * x_3d).sum(dim=1)                 # [T, H]
        return out.to(in_dtype)

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, residual_dt: DTensor) -> GridDim:
        """One CTA per token (capped at ``num_workers``).

        Each CTA runs both passes for one token; ``grid.x`` ranges
        over tokens.
        """
        pk = current_pk()
        num_tokens = residual_dt.dim(0)
        return (max(1, min(num_tokens, pk.num_workers)), 1, 1)

    # ------------------------------------------------------------------
    # MPK registration
    # ------------------------------------------------------------------
    def compile(
        self,
        residual_dt: DTensor,
        *,
        output: Optional[torch.Tensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``hc_head_fuse_v4_sm100`` task.

        Tensor contract:
          residual         : (T, HC_MULT, HIDDEN) bf16, row-major contig.
          fn       (auto)  : (HC_MULT, HC_MULT*HIDDEN) fp32 nn.Parameter.
          hc_scale (auto)  : (1,)                       fp32 nn.Parameter.
          hc_base  (auto)  : (HC_MULT,)                 fp32 nn.Parameter.
          output           : (T, HIDDEN)         bf16, allocated if None.

        Notes: ``rms_eps`` and ``hc_eps`` are hard-coded to ``1e-6f`` in
        codegen (matches V4-Flash defaults). block_dim defaults to
        (256, 1, 1) on Blackwell.
        """
        from .....core import CyTBGraph
        from .....kernel import TBGraph

        pk = current_pk()

        if residual_dt.num_dims != 3:
            raise ValueError(
                f"V4HcHeadFuse: residual_dt must be 3-D "
                f"(T, HC_MULT, HIDDEN); got num_dims={residual_dt.num_dims}"
            )
        if residual_dt.dim(1) != self.hc_mult:
            raise ValueError(
                f"V4HcHeadFuse: residual_dt.dim(1) ({residual_dt.dim(1)}) "
                f"!= hc_mult ({self.hc_mult})"
            )
        if residual_dt.dim(2) != self.hidden_size:
            raise ValueError(
                f"V4HcHeadFuse: residual_dt.dim(2) ({residual_dt.dim(2)}) "
                f"!= hidden_size ({self.hidden_size})"
            )

        T = residual_dt.dim(0)

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(residual_dt)
        if block_dim is None:
            block_dim = self.default_block_dim()

        fn_dt = pk.attach_input(self.fn.data, name=f"{self.prefix}fn")
        hc_scale_dt = pk.attach_input(
            self.hc_scale.data, name=f"{self.prefix}hc_scale"
        )
        hc_base_dt = pk.attach_input(
            self.hc_base.data, name=f"{self.prefix}hc_base"
        )

        if output is None:
            out_dt = pk.new_tensor(
                dims=(T, self.hidden_size),
                dtype=residual_dt.dtype,
                name=f"{self.prefix}out",
            )
        elif isinstance(output, torch.Tensor):
            out_dt = pk.attach_input(output, name=f"{self.prefix}out")
        elif isinstance(output, DTensor):
            out_dt = output
        else:
            raise TypeError(
                "V4HcHeadFuse.compile output must be None, torch.Tensor, "
                f"or DTensor; got {type(output).__name__}"
            )

        # ----- TBGraph construction --------------------------------------
        # residual + out partition on dim 0 (token axis); fn, hc_scale,
        # hc_base broadcast (no partition).
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(residual_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(fn_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(hc_scale_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(hc_base_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(out_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [residual_dt, fn_dt, hc_scale_dt, hc_base_dt, out_dt], tb_graph
        )
        pk.kn_graph.register_task(tb_graph, "hc_head_fuse_v4_sm100")

        return out_dt
