"""V4-Flash ``mhc_post`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/mhc_post_tilelang.md``.

Decision: **NEW**.

Rationale
---------
The vLLM TileLang kernel folds a per-token element-wise term
(``post * x``) with a small (HC x HC) matmul (``comb.T @ residual``) into
one pass over the hidden axis to produce the next hc-stream residual.
There is no existing MPK task with this exact signature; the closest
generic primitive (linear) does not encode the HC-vectorized form, and
calling 4 separate matmul/elementwise tasks would (a) bloat
launches and (b) miss the per-thread fused FMA pattern that even the
naive port relies on for register reuse.

So this is a NEW kernel:
* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/mhc_post_v4_sm100.cuh``.
* Task name: ``mhc_post_v4_sm100``.
* Enum slot: ``TASK_MHC_POST_V4_SM100 = 356``.

The Blackwell impl is intentionally naive (per the standing
"naive first, perf later" rule):

* One CTA per token. ``grid = (num_tokens, 1, 1)``, ``block = (256,1,1)``.
* fp32 accumulation, bf16 input/output, no TMA/UMMA/warp-spec.
* ``comb_mix`` (HC*HC fp32) and ``post_mix`` (HC fp32) are staged in
  shared memory at the top of the CTA so every thread can read them.
* Per-hidden index: load HC-wide residual_in column + x_in[h], compute
  HC-wide ``new_r``, write HC bf16 lanes to ``residual_out``.

Audit
-----
* dtype: ``comb_mix``/``post_mix`` fp32, ``residual_in``/``x_in``/
  ``residual_out`` bf16. Matches the spec. No silent cast.
* layout: row-major contiguous everywhere.
  ``comb_mix [T, HC, HC]``, ``residual_in [T, HC, HIDDEN]``,
  ``post_mix [T, HC]``, ``x_in [T, HIDDEN]``,
  ``residual_out [T, HC, HIDDEN]``.
* multi-batch: partitioned on dim 0 (token axis); multi-batch from day 1.
* ``forward()``: faithful PyTorch reference in fp32 with bf16 round-out.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor


__all__ = ["V4MhcPost"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class V4MhcPost(MPKModule):
    """V4-Flash mhc_post: produce the new hc-stream residual.

    Tensor contract:
      comb_mix      : (T, HC, HC)      fp32
      residual_in   : (T, HC, HIDDEN)  bf16
      post_mix      : (T, HC)          fp32
      x_in          : (T, HIDDEN)      bf16
      residual_out  : (T, HC, HIDDEN)  bf16
    """

    def __init__(
        self,
        hc: int,
        hidden_size: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if hc <= 0:
            raise ValueError(f"V4MhcPost hc must be positive; got {hc}")
        if hidden_size <= 0:
            raise ValueError(
                f"V4MhcPost hidden_size must be positive; got {hidden_size}"
            )
        self.hc = hc
        self.hidden_size = hidden_size

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        comb_mix: torch.Tensor,
        residual_in: torch.Tensor,
        post_mix: torch.Tensor,
        x_in: torch.Tensor,
    ) -> torch.Tensor:
        """Reference: ``x_out[h] = c[h] * d + sum_i a[i, h] * b[i, :]``.

        Cast to fp32 for the reduction, then back to bf16.
        """
        out_dtype = residual_in.dtype
        T = residual_in.shape[0]
        HC = self.hc
        H = self.hidden_size
        assert residual_in.shape == (T, HC, H)
        assert comb_mix.shape == (T, HC, HC)
        assert post_mix.shape == (T, HC)
        assert x_in.shape == (T, H)

        c = post_mix.to(torch.float32)              # (T, HC)
        d = x_in.to(torch.float32)                  # (T, H)
        a = comb_mix.to(torch.float32)              # (T, HC, HC)
        b = residual_in.to(torch.float32)           # (T, HC, H)

        # element-wise: c[:, h] * d[:, :]
        elem = c.unsqueeze(-1) * d.unsqueeze(1)     # (T, HC, H)
        # matmul: comb_mix.T @ residual_in -> per-token (HC, H).
        # a[t, i, h] * b[t, i, :] summed over i.
        # Use einsum for clarity.
        matm = torch.einsum("tio,tih->toh", a, b)   # (T, HC, H)

        out = (elem + matm).to(out_dtype)
        return out

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, residual_in_dt: DTensor) -> GridDim:
        pk = current_pk()
        num_tokens = residual_in_dt.dim(0)
        return (max(1, min(num_tokens, pk.num_workers)), 1, 1)

    # ------------------------------------------------------------------
    # MPK registration
    # ------------------------------------------------------------------
    def compile(
        self,
        comb_mix: DTensor,
        residual_in: DTensor,
        post_mix: DTensor,
        x_in: DTensor,
        *,
        residual_out: Optional[torch.Tensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register a ``mhc_post_v4_sm100`` task. eps-free.

        Notes: HC and HIDDEN are derived from ``residual_out`` shape on
        the C++ side (codegen reads input_ops[1]/output_ops[0]).
        block_dim defaults to (256, 1, 1) on Blackwell.
        """
        pk = current_pk()

        if residual_in.num_dims != 3:
            raise ValueError(
                "V4MhcPost: residual_in must be 3-D (T, HC, HIDDEN); "
                f"got num_dims={residual_in.num_dims}"
            )
        if residual_in.dim(1) != self.hc:
            raise ValueError(
                f"V4MhcPost: residual_in.dim(1)={residual_in.dim(1)} "
                f"!= hc={self.hc}"
            )
        if residual_in.dim(2) != self.hidden_size:
            raise ValueError(
                f"V4MhcPost: residual_in.dim(2)={residual_in.dim(2)} "
                f"!= hidden_size={self.hidden_size}"
            )
        if comb_mix.num_dims != 3 or comb_mix.dim(1) != self.hc \
                or comb_mix.dim(2) != self.hc:
            raise ValueError(
                f"V4MhcPost: comb_mix must be (T, {self.hc}, {self.hc}); "
                f"got shape ({comb_mix.dim(0)}, {comb_mix.dim(1)}, "
                f"{comb_mix.dim(2)})"
            )
        if post_mix.num_dims != 2 or post_mix.dim(1) != self.hc:
            raise ValueError(
                f"V4MhcPost: post_mix must be (T, {self.hc}); "
                f"got num_dims={post_mix.num_dims}, dim(1)={post_mix.dim(1)}"
            )
        if x_in.num_dims != 2 or x_in.dim(1) != self.hidden_size:
            raise ValueError(
                f"V4MhcPost: x_in must be (T, {self.hidden_size}); "
                f"got num_dims={x_in.num_dims}, dim(1)={x_in.dim(1)}"
            )

        num_tokens = residual_in.dim(0)

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(residual_in)
        if block_dim is None:
            block_dim = self.default_block_dim()

        if residual_out is None:
            residual_out_dt = pk.new_tensor(
                dims=(num_tokens, self.hc, self.hidden_size),
                dtype=residual_in.dtype,
                name=f"{self.prefix}mhc_post_residual_out",
            )
        elif isinstance(residual_out, torch.Tensor):
            residual_out_dt = pk.attach_input(
                residual_out, name=f"{self.prefix}mhc_post_residual_out"
            )
        elif isinstance(residual_out, DTensor):
            residual_out_dt = residual_out
        else:
            raise TypeError(
                "V4MhcPost.compile residual_out must be None, "
                f"a torch.Tensor, or a DTensor; got {type(residual_out).__name__}"
            )

        from .....core import CyTBGraph
        from .....kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(comb_mix, (0, -1, -1), 1, True)
        tb_graph.new_input(residual_in, (0, -1, -1), 1, True)
        tb_graph.new_input(post_mix, (0, -1, -1), 1, True)
        tb_graph.new_input(x_in, (0, -1, -1), 1, True)
        tb_graph.new_input(residual_out_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [comb_mix, residual_in, post_mix, x_in, residual_out_dt],
            tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "mhc_post_v4_sm100")

        return residual_out_dt
