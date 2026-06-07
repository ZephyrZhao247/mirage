"""V4-Flash ``mhc_fused`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/mhc_fused_tilelang.md``.

Decision: **NEW**.

Rationale
---------
The vLLM TileLang kernel fuses two ops for the small-token (decode)
regime: ``hc_post`` (compute new residual) + the first half of the next
``hc_pre`` (``mixes = new_residual @ fn.T`` and ``sqrsum = sum(new^2)``)
in one pass, avoiding HBM round-trip on ``residual_cur``.

The naive port keeps the same I/O contract but does not require the
optimized layered split-K fan-out -- a single CTA per token computes
``new_r`` once per hidden index, stores it as bf16, and accumulates the
per-(token, n_out) GEMM partial in registers (then warp-block reduces).
``SPLIT_K`` is hard-coded to 1 in the naive impl; the output shape
``[SPLIT_K=1, T, N_OUT]`` preserves the downstream consumer's expected
3-D layout so ``mhc_pre_big_fuse(_with_norm)`` can chain unchanged.

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/mhc_fused_v4_sm100.cuh``.
* Task name: ``mhc_fused_v4_sm100``.
* Enum slot: ``TASK_MHC_FUSED_V4_SM100 = 360``.

Audit
-----
* dtype: ``comb_mix``/``post_mix``/``weight_t`` fp32; ``residual_in``/
  ``x_in``/``residual_out`` bf16; ``gemm_out_mul``/``gemm_out_sqrsum``
  fp32 -- producer-consumer dtype matches the no-norm and with-norm
  pre kernels (which read fp32 gemm_out_*).
* layout: row-major contiguous everywhere.
* multi-batch: partitioned on token dim; multi-batch from day 1.
* ``forward()``: faithful PyTorch reference matching the spec's Math
  section -- ``new_r``, ``residual_out``, ``sqrsum``, and ``mul``.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor


__all__ = ["V4MhcFused"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class V4MhcFused(MPKModule):
    """V4-Flash mhc_fused (decode-regime fused mhc_post + hc_prenorm_gemm).

    Tensor contract:
      comb_mix        : (T, HC, HC)        fp32
      residual_in     : (T, HC, HIDDEN)    bf16
      post_mix        : (T, HC)            fp32
      x_in            : (T, HIDDEN)        bf16
      weight_t        : (N_OUT, HC, HIDDEN) fp32
      ---
      gemm_out_mul    : (1, T, N_OUT)      fp32  (SPLIT_K=1)
      gemm_out_sqrsum : (1, T)             fp32  (SPLIT_K=1)
      residual_out    : (T, HC, HIDDEN)    bf16
    """

    def __init__(
        self,
        hc: int,
        hidden_size: int,
        n_out: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if hc <= 0 or hidden_size <= 0 or n_out <= 0:
            raise ValueError("V4MhcFused: positive sizes required")
        self.hc = hc
        self.hidden_size = hidden_size
        self.n_out = n_out

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        comb_mix: torch.Tensor,
        residual_in: torch.Tensor,
        post_mix: torch.Tensor,
        x_in: torch.Tensor,
        weight_t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        T, HC, H = residual_in.shape
        N = self.n_out
        residual_dtype = residual_in.dtype

        c = post_mix.to(torch.float32)              # (T, HC)
        d = x_in.to(torch.float32)                  # (T, H)
        a = comb_mix.to(torch.float32)              # (T, HC, HC)
        b = residual_in.to(torch.float32)           # (T, HC, H)
        w = weight_t.to(torch.float32)              # (N, HC, H)

        # new_r[t, hco, h] = c[t, hco] * d[t, h]
        #                  + sum_i a[t, i, hco] * b[t, i, h]
        elem = c.unsqueeze(-1) * d.unsqueeze(1)     # (T, HC, H)
        matm = torch.einsum("tio,tih->toh", a, b)   # (T, HC, H)
        new_r = elem + matm                          # (T, HC, H), fp32

        residual_out = new_r.to(residual_dtype)

        # sqrsum: sum over (hc, h) of new_r^2 per token. Single split.
        sqrsum = new_r.pow(2).sum(dim=(1, 2)).unsqueeze(0)  # (1, T)

        # gemm_out_mul[0, t, n] = sum_hc sum_h w[n, hc, h] * new_r[t, hc, h].
        # equivalent to einsum "nih,toh->toh"... simpler with einsum:
        mul = torch.einsum("nih,tih->tn", w, new_r)         # (T, N)
        mul = mul.unsqueeze(0)                              # (1, T, N)

        return mul, sqrsum, residual_out

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
        weight_t: DTensor,
        *,
        gemm_out_mul: Optional[torch.Tensor] = None,
        gemm_out_sqrsum: Optional[torch.Tensor] = None,
        residual_out: Optional[torch.Tensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor, DTensor]:
        pk = current_pk()

        if residual_in.num_dims != 3 \
                or residual_in.dim(1) != self.hc \
                or residual_in.dim(2) != self.hidden_size:
            raise ValueError(
                f"V4MhcFused: residual_in must be (T, {self.hc}, "
                f"{self.hidden_size})"
            )
        T = residual_in.dim(0)
        if comb_mix.num_dims != 3 or comb_mix.dim(1) != self.hc \
                or comb_mix.dim(2) != self.hc:
            raise ValueError(
                f"V4MhcFused: comb_mix must be (T, {self.hc}, {self.hc})"
            )
        if post_mix.num_dims != 2 or post_mix.dim(1) != self.hc:
            raise ValueError(
                f"V4MhcFused: post_mix must be (T, {self.hc})"
            )
        if x_in.num_dims != 2 or x_in.dim(1) != self.hidden_size:
            raise ValueError(
                f"V4MhcFused: x_in must be (T, {self.hidden_size})"
            )
        if weight_t.num_dims != 3 \
                or weight_t.dim(0) != self.n_out \
                or weight_t.dim(1) != self.hc \
                or weight_t.dim(2) != self.hidden_size:
            raise ValueError(
                f"V4MhcFused: weight_t must be ({self.n_out}, {self.hc}, "
                f"{self.hidden_size})"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(residual_in)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # ----- allocate outputs ------------------------------------------
        def _resolve(out, dims, dtype, label):
            if out is None:
                return pk.new_tensor(
                    dims=dims, dtype=dtype,
                    name=f"{self.prefix}mhc_fused_{label}",
                )
            if isinstance(out, torch.Tensor):
                return pk.attach_input(
                    out, name=f"{self.prefix}mhc_fused_{label}"
                )
            if isinstance(out, DTensor):
                return out
            raise TypeError(
                f"V4MhcFused.compile {label} must be None/Tensor/DTensor"
            )

        from .....core import float32, bfloat16
        gemm_out_mul_dt = _resolve(
            gemm_out_mul, (1, T, self.n_out), float32, "gemm_out_mul",
        )
        gemm_out_sqrsum_dt = _resolve(
            gemm_out_sqrsum, (1, T), float32, "gemm_out_sqrsum",
        )
        residual_out_dt = _resolve(
            residual_out,
            (T, self.hc, self.hidden_size),
            bfloat16,
            "residual_out",
        )

        # ----- TBGraph -------------------------------------------------
        from .....core import CyTBGraph
        from .....kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(comb_mix, (0, -1, -1), 1, True)
        tb_graph.new_input(residual_in, (0, -1, -1), 1, True)
        tb_graph.new_input(post_mix, (0, -1, -1), 1, True)
        tb_graph.new_input(x_in, (0, -1, -1), 1, True)
        # weight_t: pure broadcast (no token partition).
        tb_graph.new_input(weight_t, (-1, -1, -1), 0, True)
        # gemm_out_*: token-keyed on dim 1.
        tb_graph.new_input(gemm_out_mul_dt, (1, -1, -1), 1, True)
        tb_graph.new_input(gemm_out_sqrsum_dt, (1, -1, -1), 1, True)
        tb_graph.new_input(residual_out_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [
                comb_mix,
                residual_in,
                post_mix,
                x_in,
                weight_t,
                gemm_out_mul_dt,
                gemm_out_sqrsum_dt,
                residual_out_dt,
            ],
            tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "mhc_fused_v4_sm100")

        return gemm_out_mul_dt, gemm_out_sqrsum_dt, residual_out_dt
