"""V4-Flash ``silu_and_mul_with_clamp`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/silu_and_mul_with_clamp.md``.

Decision: **NEW**.

Rationale
---------
The vLLM kernel is a fused SwiGLU + asymmetric-clamp activation used by
DeepSeek V4-Flash's shared experts (and dense MLP path). It computes:

    out = silu(clamp(gate, max=L)) * clamp(up, -L, L)

with bf16 in / bf16 out and ``L = swiglu_limit`` (V4-Flash: 10.0). The
clamp asymmetry (one-sided on gate, two-sided on up) matches the reference
``Expert.forward`` (``model.py:596-606``).

The existing :class:`mirage.mpk.layers.SiluMul` (backed by
``tasks/{ampere,hopper}/silu_mul*.cuh``) does **plain** ``silu(gate) * up``
with NO clamp support. Adding a clamp parameter would require either
templating the existing kernel on ``HAS_CLAMP`` (touches every callsite in
qwen3 / Llama / etc.) or wiring an extra optional input, neither of which
is in scope for the naive port. We therefore add a NEW kernel that takes
the limit as a per-task scalar parameter; existing :class:`SiluMul`
callers are untouched.

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/silu_and_mul_with_clamp_v4_sm100.cuh``
* Task name: ``silu_and_mul_with_clamp_v4_sm100``
* Enum slot: ``TASK_SILU_AND_MUL_WITH_CLAMP_V4_SM100 = 382``

Blackwell impl is intentionally NAIVE:

* One CTA per token slab (``grid.x`` slices the intermediate dim,
  ``grid.y`` = tokens). For per-token simplicity we register a single
  per-token CTA layout (``grid = (T, 1, 1)``) -- the kernel walks the
  full ``INTERMEDIATE_SIZE`` with thread-strided loops.
* ``block_dim = (256, 1, 1)`` on Blackwell.
* fp32 internal math; bf16 in/out.
* No vectorization / no 256b loads / no warp-spec.

Audit
-----
* dtype: bf16 in / bf16 out -- matches the spec's input.dtype==out.dtype
  contract.
* layout: row-major contiguous. Per-row split: ``input[t, :d]`` = gate,
  ``input[t, d:]`` = up; ``out[t, :d]``.
* multi-batch: kernel partitions on dim 0 (tokens). Test uses
  ``max_num_batched_requests = 4``.
* The ``swiglu_limit`` parameter is forwarded into the task's ``params[]``
  as a float bitcast (int) and reconstituted in codegen via
  ``__uint_as_float``.
"""
from __future__ import annotations

import struct
from typing import Optional, Tuple

import torch

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor

__all__ = ["V4SiluAndMulWithClamp"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


def _float_to_int_bits(x: float) -> int:
    """Return the IEEE-754 fp32 bit pattern of ``x`` as a non-negative int.

    We forward swiglu_limit through ``params[]`` as int; the codegen calls
    ``__uint_as_float(bits)`` on the device side to reconstitute the float
    bit-exactly.
    """
    packed = struct.pack("<f", float(x))
    return int.from_bytes(packed, byteorder="little", signed=False)


class V4SiluAndMulWithClamp(MPKModule):
    """V4-Flash ``silu_and_mul_with_clamp`` naive Blackwell kernel.

    Constructor args:

      * ``intermediate_size`` -- the per-token output width ``d``
        (V4-Flash shared expert: 2048; the input is ``2*d``-wide).
      * ``swiglu_limit``      -- ``L``. V4-Flash: 10.0.
    """

    def __init__(
        self,
        intermediate_size: int,
        swiglu_limit: float = 10.0,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if intermediate_size <= 0:
            raise ValueError(
                f"V4SiluAndMulWithClamp intermediate_size must be positive; "
                f"got {intermediate_size}"
            )
        if not (swiglu_limit > 0):
            # The spec lets limit <= 0 pass through the kernel's fminf/fmaxf
            # well-definedly, but every V4-Flash caller uses 10.0; refuse
            # surprises.
            raise ValueError(
                f"V4SiluAndMulWithClamp swiglu_limit must be positive; "
                f"got {swiglu_limit}"
            )
        self.intermediate_size = intermediate_size
        self.swiglu_limit = float(swiglu_limit)

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(self, gateup: torch.Tensor) -> torch.Tensor:
        """Eager reference matching the vLLM kernel.

        gateup is the fused ``[T, 2*d]`` tensor; the first ``d`` columns are
        the gate, the next ``d`` columns are the up. fp32 internal math;
        output cast back to ``gateup.dtype`` (bf16 on V4-Flash).
        """
        if gateup.dim() != 2:
            raise ValueError(
                f"gateup must be 2-D (T, 2*d); got shape {tuple(gateup.shape)}"
            )
        if gateup.size(1) != 2 * self.intermediate_size:
            raise ValueError(
                f"gateup.size(1) ({gateup.size(1)}) must equal "
                f"2*intermediate_size ({2 * self.intermediate_size})"
            )
        gate = gateup[:, : self.intermediate_size].to(torch.float32)
        up = gateup[:, self.intermediate_size :].to(torch.float32)
        L = self.swiglu_limit
        gate_c = torch.clamp(gate, max=L)
        up_c = torch.clamp(up, min=-L, max=L)
        silu = gate_c / (1.0 + torch.exp(-gate_c))
        return (silu * up_c).to(gateup.dtype)

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, gateup_dt: DTensor) -> GridDim:
        """One CTA per token (capped at ``num_workers``)."""
        pk = current_pk()
        num_tokens = gateup_dt.dim(0)
        return (max(1, min(num_tokens, pk.num_workers)), 1, 1)

    # ------------------------------------------------------------------
    # MPK registration
    # ------------------------------------------------------------------
    def compile(
        self,
        gateup: DTensor,
        *,
        output: Optional[torch.Tensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``silu_and_mul_with_clamp_v4_sm100`` task.

        Tensor contract:
          gateup : (T, 2 * INTERMEDIATE_SIZE) bf16, row-major contiguous.
          out    : (T, INTERMEDIATE_SIZE)     bf16, allocated if None.
        """
        from .....core import CyTBGraph
        from .....kernel import TBGraph

        pk = current_pk()

        if gateup.num_dims != 2:
            raise ValueError(
                "V4SiluAndMulWithClamp: gateup must be 2-D (T, 2*d); got "
                f"num_dims={gateup.num_dims}"
            )
        if gateup.dim(1) != 2 * self.intermediate_size:
            raise ValueError(
                f"V4SiluAndMulWithClamp: gateup.dim(1)={gateup.dim(1)} != "
                f"2*intermediate_size ({2 * self.intermediate_size})"
            )
        num_tokens = gateup.dim(0)

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(gateup)
        if block_dim is None:
            block_dim = self.default_block_dim()

        if output is None:
            out_dt = pk.new_tensor(
                dims=(num_tokens, self.intermediate_size),
                dtype=gateup.dtype,
                name=f"{self.prefix}silu_clamp_out",
            )
        elif isinstance(output, torch.Tensor):
            out_dt = pk.attach_input(
                output, name=f"{self.prefix}silu_clamp_out"
            )
        elif isinstance(output, DTensor):
            out_dt = output
        else:
            raise TypeError(
                "V4SiluAndMulWithClamp.compile output must be None, "
                f"torch.Tensor, or DTensor; got {type(output).__name__}"
            )

        # ----- TBGraph construction --------------------------------------
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(gateup, (0, -1, -1), 1, True)
        tb_graph.new_input(out_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized([gateup, out_dt], tb_graph)
        # Pass swiglu_limit via params[0] (float-bitcast int). Codegen
        # reconstitutes it via __uint_as_float on the device side.
        limit_bits = _float_to_int_bits(self.swiglu_limit)
        pk.kn_graph.register_task(
            tb_graph, "silu_and_mul_with_clamp_v4_sm100", [limit_bits]
        )
        return out_dt
