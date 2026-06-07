"""V4-Flash ``fused_mtp_input_rmsnorm`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fused_mtp_input_rmsnorm.md``.

Decision: **NEW**.

Rationale
---------
The vLLM Triton kernel jointly runs ``HC_MULT + 1`` RMS norms per token:

* slot 0          -- ``enorm(inputs_embeds[t, :])`` with **pos==0 zero-mask**
* slot 1..HC_MULT -- ``hnorm(prev_hidden[t, slot-1, :])``, unconditional

REUSE/EXTEND is **not** viable:

* The existing generic :class:`mirage.mpk.layers.RMSNorm` does not
  accept a per-token boolean mask, takes a single (input, weight, output)
  trio, and reads dim-0-partitioned 2-D tensors. The MTP path needs 5
  inputs / 2 outputs, has an ``int64`` ``positions`` mask, and indexes
  the 3-D ``prev_hidden`` buffer per-slot.
* Calling ``RMSNorm`` ``HC_MULT + 1`` times would (a) miss the pos==0
  mask semantics that vLLM grafts on at the kernel boundary, (b) produce
  a dim-mismatch between the 2-D inputs_embeds and the 3-D prev_hidden,
  and (c) launch ``HC_MULT + 1`` separate task instances per token --
  defeating the joint-grid intent of the original Triton kernel.

Hence: a new ``__device__`` impl at
``include/mirage/persistent_kernel/tasks/blackwell/fused_mtp_input_rmsnorm_v4_sm100.cuh``
plus a new ``TaskType`` slot
(``TASK_FUSED_MTP_INPUT_RMSNORM_V4_SM100 = 389``).

The Blackwell impl is intentionally **naive** (per the integration
brief: "Naive implementations only. Plain CUDA loops, fixed block_dim
= 256 on Blackwell, single CTA, no TMA/UMMA/warp-specialization."):

* One CTA per token (``grid = (num_tokens, 1, 1)``,
  ``block = (256, 1, 1)``).
* The CTA runs the (HC_MULT + 1) norms **sequentially** in a single
  thread block; performance is not a goal.
* fp32 sum-of-squares via thread-strided loop + intra-warp ``shfl_xor``
  + cross-warp shared-mem combine; bf16 store with on-the-fly fp32 ->
  bf16 cast at the write.
* The pos==0 mask is implemented as "skip the reduction and write 0";
  variance of an all-zero row is 0, so the Triton math ``x * rsqrt(eps)
  * w`` is also 0 -- byte-for-byte equivalent.

Audit
-----
* dtype: bf16 in / bf16 out. ``positions`` is ``int64`` (matches
  ``torch.long``). No silent dtype mismatch.
* layout: row-major contiguous everywhere
  (``inputs_embeds: [T, H]``, ``positions: [T]``,
  ``prev_hidden: [T, HC_MULT, H]``, weights: ``[H]``,
  ``enorm_out: [T, H]``, ``hnorm_out: [T, HC_MULT, H]``).
* multi-batch: kernel partitions on the token dim; multi-batch is the
  default. Covered by ``tests/runtime_python/layers/test_fused_mtp_input_rmsnorm_v4.py``
  (``max_num_batched_requests = 4``).
* ``forward()`` provides the faithful PyTorch reference (fp32 reduction,
  pos==0 mask, bf16 output).
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor


__all__ = ["V4FusedMTPInputRMSNorm"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class V4FusedMTPInputRMSNorm(MPKModule):
    """V4-Flash joint MTP input RMSNorm (enorm + per-slot hnorm).

    Owns two ``nn.Parameter`` weights:

      * ``enorm_weight: [hidden_size]`` bf16
      * ``hnorm_weight: [hidden_size]`` bf16

    Constructor args:

      * ``hidden_size`` -- per-token feature dim ``H`` (e.g., 4096).
      * ``hc_mult``     -- the HC multiplier (e.g., 8); number of
                           slots in ``prev_hidden``.
      * ``eps``         -- forwarded to ``forward()``; the kernel
                           hard-codes ``1e-6f`` in codegen to match
                           ``config.rms_norm_eps``.
    """

    def __init__(
        self,
        hidden_size: int,
        hc_mult: int,
        eps: float = 1e-6,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if hidden_size <= 0:
            raise ValueError(
                f"V4FusedMTPInputRMSNorm hidden_size must be positive; "
                f"got {hidden_size}"
            )
        if hc_mult <= 0:
            raise ValueError(
                f"V4FusedMTPInputRMSNorm hc_mult must be positive; "
                f"got {hc_mult}"
            )
        self.hidden_size = hidden_size
        self.hc_mult = hc_mult
        self.eps = eps
        # Match nn.RMSNorm convention: weight defaults to ones.
        self.enorm_weight = nn.Parameter(torch.ones(hidden_size))
        self.hnorm_weight = nn.Parameter(torch.ones(hidden_size))

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        positions: torch.Tensor,
        previous_hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Eager reference matching the vLLM Triton kernel.

        ``enorm_out[t]`` is 0 when ``positions[t] == 0`` (the pos-mask
        semantics vLLM added on top of the model's plain RMSNorm path).
        """
        in_dtype = inputs_embeds.dtype
        assert inputs_embeds.shape == (
            inputs_embeds.shape[0],
            self.hidden_size,
        ), f"inputs_embeds shape: {tuple(inputs_embeds.shape)}"
        assert previous_hidden_states.shape == (
            inputs_embeds.shape[0],
            self.hc_mult,
            self.hidden_size,
        ), f"previous_hidden_states shape: {tuple(previous_hidden_states.shape)}"
        assert positions.shape == (inputs_embeds.shape[0],), (
            f"positions shape: {tuple(positions.shape)}"
        )
        assert positions.dtype in (torch.int64, torch.long), (
            f"positions dtype: {positions.dtype}"
        )

        # enorm with pos==0 mask. Cast to fp32 for the reduction.
        keep = (positions != 0).view(-1, 1).to(torch.float32)
        e = inputs_embeds.to(torch.float32) * keep
        e_var = e.pow(2).mean(dim=-1, keepdim=True)
        e_norm = e * torch.rsqrt(e_var + self.eps)
        enorm_out = (e_norm * self.enorm_weight.to(torch.float32)).to(in_dtype)

        # hnorm (unconditional per slot). prev_hidden -> [T, HC_MULT, H].
        h = previous_hidden_states.to(torch.float32)
        h_var = h.pow(2).mean(dim=-1, keepdim=True)
        h_norm = h * torch.rsqrt(h_var + self.eps)
        hnorm_out = (h_norm * self.hnorm_weight.to(torch.float32)).to(in_dtype)

        return enorm_out, hnorm_out

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, inputs_embeds_dt: DTensor) -> GridDim:
        """One CTA per token (capped at ``num_workers``).

        Kernel processes (HC_MULT + 1) norms sequentially inside one
        CTA; ``grid.x`` ranges over tokens, ``y`` and ``z`` are 1.
        """
        pk = current_pk()
        num_tokens = inputs_embeds_dt.dim(0)
        return (max(1, min(num_tokens, pk.num_workers)), 1, 1)

    # ------------------------------------------------------------------
    # MPK registration
    # ------------------------------------------------------------------
    def compile(
        self,
        inputs_embeds: DTensor,
        positions: DTensor,
        previous_hidden_states: DTensor,
        *,
        enorm_out: Optional[torch.Tensor] = None,
        hnorm_out: Optional[torch.Tensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register one ``fused_mtp_input_rmsnorm_v4_sm100`` task.

        Tensor contract:
          inputs_embeds          : (T, H) bf16, row-major contiguous.
          positions              : (T,)   int64, dense.
          previous_hidden_states : (T, HC_MULT, H) bf16, row-major contiguous.
          enorm_weight (auto)    : (H,)   bf16 nn.Parameter.
          hnorm_weight (auto)    : (H,)   bf16 nn.Parameter.
          enorm_out              : (T, H) bf16, allocated if None.
          hnorm_out              : (T, HC_MULT, H) bf16, allocated if None.

        Notes: eps is hard-coded to ``1e-6f`` in codegen (matches
        V4-Flash's ``rms_norm_eps``); the ``eps`` ctor arg only affects
        ``forward()``. block_dim defaults to (256, 1, 1) on Blackwell.
        """
        pk = current_pk()

        if inputs_embeds.num_dims != 2:
            raise ValueError(
                "V4FusedMTPInputRMSNorm: inputs_embeds must be 2-D "
                f"(T, H); got num_dims={inputs_embeds.num_dims}"
            )
        if positions.num_dims != 1:
            raise ValueError(
                "V4FusedMTPInputRMSNorm: positions must be 1-D "
                f"(T,); got num_dims={positions.num_dims}"
            )
        if previous_hidden_states.num_dims != 3:
            raise ValueError(
                "V4FusedMTPInputRMSNorm: previous_hidden_states must be "
                f"3-D (T, HC_MULT, H); got num_dims="
                f"{previous_hidden_states.num_dims}"
            )
        if inputs_embeds.dim(1) != self.hidden_size:
            raise ValueError(
                f"V4FusedMTPInputRMSNorm: inputs_embeds last-dim "
                f"({inputs_embeds.dim(1)}) != hidden_size "
                f"({self.hidden_size})"
            )
        if previous_hidden_states.dim(1) != self.hc_mult:
            raise ValueError(
                f"V4FusedMTPInputRMSNorm: previous_hidden_states[1] "
                f"({previous_hidden_states.dim(1)}) != hc_mult "
                f"({self.hc_mult})"
            )
        if previous_hidden_states.dim(2) != self.hidden_size:
            raise ValueError(
                f"V4FusedMTPInputRMSNorm: previous_hidden_states[2] "
                f"({previous_hidden_states.dim(2)}) != hidden_size "
                f"({self.hidden_size})"
            )

        num_tokens = inputs_embeds.dim(0)

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(inputs_embeds)
        if block_dim is None:
            block_dim = self.default_block_dim()  # (256,1,1) on Blackwell

        # ----- attach / allocate weights and outputs ----------------------
        enorm_w_dt = pk.attach_input(
            self.enorm_weight.data, name=f"{self.prefix}enorm_weight"
        )
        hnorm_w_dt = pk.attach_input(
            self.hnorm_weight.data, name=f"{self.prefix}hnorm_weight"
        )

        if enorm_out is None:
            enorm_out_dt = pk.new_tensor(
                dims=(num_tokens, self.hidden_size),
                dtype=inputs_embeds.dtype,
                name=f"{self.prefix}enorm_out",
            )
        elif isinstance(enorm_out, torch.Tensor):
            enorm_out_dt = pk.attach_input(
                enorm_out, name=f"{self.prefix}enorm_out"
            )
        elif isinstance(enorm_out, DTensor):
            enorm_out_dt = enorm_out
        else:
            raise TypeError(
                "V4FusedMTPInputRMSNorm.compile enorm_out must be None, "
                f"a torch.Tensor, or a DTensor; got {type(enorm_out).__name__}"
            )

        if hnorm_out is None:
            hnorm_out_dt = pk.new_tensor(
                dims=(num_tokens, self.hc_mult, self.hidden_size),
                dtype=inputs_embeds.dtype,
                name=f"{self.prefix}hnorm_out",
            )
        elif isinstance(hnorm_out, torch.Tensor):
            hnorm_out_dt = pk.attach_input(
                hnorm_out, name=f"{self.prefix}hnorm_out"
            )
        elif isinstance(hnorm_out, DTensor):
            hnorm_out_dt = hnorm_out
        else:
            raise TypeError(
                "V4FusedMTPInputRMSNorm.compile hnorm_out must be None, "
                f"a torch.Tensor, or a DTensor; got {type(hnorm_out).__name__}"
            )

        # ----- TBGraph construction ---------------------------------------
        # Partition all token-keyed tensors on dim 0 (the token axis);
        # the weights are broadcast (no partition). Same convention as
        # rmsnorm_hopper -- the runtime pre-offsets pointers so each
        # task gets a per-token slice.
        from .....core import CyTBGraph
        from .....kernel import TBGraph

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(inputs_embeds, (0, -1, -1), 1, True)
        tb_graph.new_input(positions, (0, -1, -1), 1, True)
        tb_graph.new_input(previous_hidden_states, (0, -1, -1), 1, True)
        tb_graph.new_input(enorm_w_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(hnorm_w_dt, (-1, -1, -1), 0, True)
        tb_graph.new_input(enorm_out_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(hnorm_out_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [
                inputs_embeds,
                positions,
                previous_hidden_states,
                enorm_w_dt,
                hnorm_w_dt,
                enorm_out_dt,
                hnorm_out_dt,
            ],
            tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "fused_mtp_input_rmsnorm_v4_sm100")

        return enorm_out_dt, hnorm_out_dt
