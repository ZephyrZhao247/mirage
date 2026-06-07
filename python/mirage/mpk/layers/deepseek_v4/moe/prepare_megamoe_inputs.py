"""V4-Flash ``prepare_megamoe_inputs`` -- NEW Blackwell naive kernel.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/prepare_megamoe_inputs.md``.

Decision: **NEW**.

Rationale
---------
The vLLM reference is a Triton kernel (
``vllm/models/deepseek_v4/nvidia/ops/prepare_megamoe.py``) that fuses three
things into a single per-token pass:

1. ``hidden_states (bf16) -> x_fp8 (fp8_e4m3)`` quantization with
   per-group (GROUP_K=32) UE8M0 scales, walking the hidden vector in
   BLOCK_K=128 chunks. The 4 UE8M0 exponent bytes for each block are
   packed into one int32 word (the layout DeepGEMM's UTCCP path
   consumes).
2. ``topk_ids (int) -> topk_idx_out (int64)`` cast.
3. ``topk_weights (fp32) -> topk_weights_out (fp32)`` byte-copy.

No existing MPK task matches this composite contract. The existing
``per_token_group_quantize_fp8`` covers (1) but uses a different scale
layout (one byte per group, not packed-int32-per-block) and does NOT
fuse the topk repack. Reusing it would require both a layout transform
and a second task; the naive port emits a single fused kernel that
matches the spec exactly.

* CUDA header:
  ``include/mirage/persistent_kernel/tasks/blackwell/prepare_megamoe_inputs_v4_sm100.cuh``
* Task name: ``prepare_megamoe_inputs_v4_sm100``
* Enum slot: ``TASK_PREPARE_MEGAMOE_INPUTS_V4_SM100 = 383``

Blackwell impl is intentionally NAIVE:

* One CTA per token. ``grid = (T, 1, 1)``, ``block = (256, 1, 1)``.
* Inside the CTA, iterate over the ``HIDDEN_SIZE // BLOCK_K`` chunks
  sequentially. Thread 0 computes the 4 group absmax + UE8M0 exponents +
  packed int32; threads 0..BLOCK_K-1 emit the FP8 elements using the
  per-group inv-scale shared via shared memory.
* Topk repack happens once per token (every CTA emits it).
* BLOCK_K = 128, GROUP_K = 32, NUM_THREADS = 256 baked at codegen.

Audit
-----
* dtype: hidden_states bf16, topk_ids int32, topk_weights fp32.
  Outputs: x_fp8 (fp8_e4m3), x_sf (int32 packed UE8M0), topk_idx_out
  (int64), topk_weights_out (fp32). Matches the spec.
* layout: all row-major contiguous. ``x_sf: [T, H/128]``.
* multi-batch: partitioned on dim 0 (tokens). Test uses
  ``max_num_batched_requests = 4``.
* The UE8M0 rounding rule (``mantissa-nonzero -> bump exponent``)
  matches the Triton reference bit-for-bit; see the device-side
  ``ue8m0_exponent`` helper for the bit-level construction.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch

from ..._base import MPKModule
from ....context import current_pk
from .....core import DTensor

__all__ = ["V4PrepareMegaMoEInputs"]

GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


def _ue8m0_ref(
    hidden: torch.Tensor,
    block_k: int = 128,
    group_k: int = 32,
    eps: float = 1.0e-4,
    fp8_max: float = 448.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """PyTorch reference for the UE8M0-packed FP8 quant.

    Returns (x_fp8: fp8_e4m3 [T,H], x_sf: int32 [T, H//block_k] packed).
    Matches the Triton kernel's bit-level convention exactly.
    """
    if hidden.dim() != 2:
        raise ValueError(f"hidden must be 2-D [T, H]; got {tuple(hidden.shape)}")
    T, H = hidden.shape
    if H % block_k != 0:
        raise ValueError(
            f"H={H} must be a multiple of block_k={block_k}"
        )
    if block_k % group_k != 0:
        raise ValueError(
            f"block_k={block_k} must be a multiple of group_k={group_k}"
        )
    num_blocks = H // block_k
    num_groups_per_block = block_k // group_k

    h32 = hidden.to(torch.float32)
    # Per-group absmax: shape [T, num_blocks, num_groups_per_block]
    grouped = h32.view(T, num_blocks, num_groups_per_block, group_k)
    amax = grouped.abs().amax(dim=-1).clamp_min(eps)  # [T, NB, NGB]

    scale = amax / fp8_max  # fp32
    scale_bits = scale.view(torch.int32)
    exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
        torch.int32
    )
    exp = exp.clamp(1, 254)

    # Rounded scale = 2^k where k = exp - 127 (so exp_bits << 23 viewed as fp32).
    rounded_scale_bits = (exp << 23)
    rounded_scale = rounded_scale_bits.view(torch.float32)

    inv_scale = 1.0 / rounded_scale  # [T, NB, NGB]
    # Broadcast inv_scale to each element in the group:
    inv_scale_full = inv_scale.unsqueeze(-1).expand(
        T, num_blocks, num_groups_per_block, group_k
    )
    quantized = grouped * inv_scale_full  # [T, NB, NGB, GK]
    quantized = quantized.reshape(T, H)
    x_fp8 = quantized.to(torch.float8_e4m3fn)

    # Pack 4 UE8M0 exponents per block into one int32 word: exp[g] << (g*8).
    shifts = torch.arange(num_groups_per_block, device=hidden.device, dtype=torch.int32) * 8
    packed = ((exp & 0xFF) << shifts).sum(dim=-1).to(torch.int32)  # [T, NB]
    return x_fp8, packed


class V4PrepareMegaMoEInputs(MPKModule):
    """V4-Flash ``prepare_megamoe_inputs`` naive Blackwell kernel.

    Constructor args:

      * ``hidden_size`` -- ``H``. Spec requires multiple of 128
                           (BLOCK_K). V4-Flash: 7168.
      * ``topk``        -- ``K``. V4-Flash: 8 (routed experts).
    """

    BLOCK_K: int = 128
    GROUP_K: int = 32

    def __init__(
        self,
        hidden_size: int,
        topk: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if hidden_size <= 0:
            raise ValueError(
                f"V4PrepareMegaMoEInputs hidden_size must be positive; "
                f"got {hidden_size}"
            )
        if hidden_size % self.BLOCK_K != 0:
            raise ValueError(
                f"V4PrepareMegaMoEInputs hidden_size must be a multiple of "
                f"{self.BLOCK_K}; got {hidden_size}"
            )
        if topk <= 0:
            raise ValueError(
                f"V4PrepareMegaMoEInputs topk must be positive; got {topk}"
            )
        self.hidden_size = hidden_size
        self.topk = topk

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ):
        """Eager reference. Returns (x_fp8, x_sf, topk_idx_out, topk_weights_out)."""
        if hidden_states.dim() != 2:
            raise ValueError(
                "hidden_states must be 2-D [T, H]; got shape "
                f"{tuple(hidden_states.shape)}"
            )
        if topk_ids.dim() != 2 or topk_ids.shape[1] != self.topk:
            raise ValueError(
                f"topk_ids must be 2-D [T, K={self.topk}]; got "
                f"{tuple(topk_ids.shape)}"
            )
        if topk_weights.shape != topk_ids.shape:
            raise ValueError(
                f"topk_weights shape {tuple(topk_weights.shape)} must match "
                f"topk_ids shape {tuple(topk_ids.shape)}"
            )
        x_fp8, x_sf = _ue8m0_ref(
            hidden_states, block_k=self.BLOCK_K, group_k=self.GROUP_K
        )
        topk_idx_out = topk_ids.to(torch.int64).contiguous()
        topk_weights_out = topk_weights.to(torch.float32).contiguous()
        return x_fp8, x_sf, topk_idx_out, topk_weights_out

    # ------------------------------------------------------------------
    # Grid heuristic
    # ------------------------------------------------------------------
    def auto_grid_dim(self, hidden_states_dt: DTensor) -> GridDim:
        pk = current_pk()
        num_tokens = hidden_states_dt.dim(0)
        return (max(1, min(num_tokens, pk.num_workers)), 1, 1)

    # ------------------------------------------------------------------
    # MPK registration
    # ------------------------------------------------------------------
    def compile(
        self,
        hidden_states: DTensor,
        topk_ids: DTensor,
        topk_weights: DTensor,
        *,
        x_fp8: Optional[torch.Tensor] = None,
        x_sf: Optional[torch.Tensor] = None,
        topk_idx_out: Optional[torch.Tensor] = None,
        topk_weights_out: Optional[torch.Tensor] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ):
        """Register one ``prepare_megamoe_inputs_v4_sm100`` task."""
        from .....core import (
            CyTBGraph,
            float32 as _mi_f32,
            float8_e4m3 as _mi_fp8,
            int32 as _mi_i32,
            int64 as _mi_i64,
        )
        from .....kernel import TBGraph

        pk = current_pk()

        if hidden_states.num_dims != 2:
            raise ValueError(
                "V4PrepareMegaMoEInputs: hidden_states must be 2-D; got "
                f"num_dims={hidden_states.num_dims}"
            )
        if hidden_states.dim(1) != self.hidden_size:
            raise ValueError(
                f"V4PrepareMegaMoEInputs: hidden_states.dim(1)="
                f"{hidden_states.dim(1)} != hidden_size ({self.hidden_size})"
            )
        num_tokens = hidden_states.dim(0)
        num_blocks = self.hidden_size // self.BLOCK_K
        K = self.topk

        if topk_ids.num_dims != 2 or topk_ids.dim(1) != K:
            raise ValueError(
                f"V4PrepareMegaMoEInputs: topk_ids must be 2-D [T, K={K}]; "
                f"got num_dims={topk_ids.num_dims}, "
                f"dim(1)={topk_ids.dim(1) if topk_ids.num_dims >= 2 else 'NA'}"
            )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(hidden_states)
        if block_dim is None:
            block_dim = self.default_block_dim()

        if x_fp8 is None:
            x_fp8_dt = pk.new_tensor(
                dims=(num_tokens, self.hidden_size),
                dtype=_mi_fp8,
                name=f"{self.prefix}x_fp8",
            )
        elif isinstance(x_fp8, torch.Tensor):
            x_fp8_dt = pk.attach_input(x_fp8, name=f"{self.prefix}x_fp8")
        else:
            x_fp8_dt = x_fp8

        if x_sf is None:
            x_sf_dt = pk.new_tensor(
                dims=(num_tokens, num_blocks),
                dtype=_mi_i32,
                name=f"{self.prefix}x_sf",
            )
        elif isinstance(x_sf, torch.Tensor):
            x_sf_dt = pk.attach_input(x_sf, name=f"{self.prefix}x_sf")
        else:
            x_sf_dt = x_sf

        if topk_idx_out is None:
            topk_idx_out_dt = pk.new_tensor(
                dims=(num_tokens, K),
                dtype=_mi_i64,
                name=f"{self.prefix}topk_idx_out",
            )
        elif isinstance(topk_idx_out, torch.Tensor):
            topk_idx_out_dt = pk.attach_input(
                topk_idx_out, name=f"{self.prefix}topk_idx_out"
            )
        else:
            topk_idx_out_dt = topk_idx_out

        if topk_weights_out is None:
            topk_weights_out_dt = pk.new_tensor(
                dims=(num_tokens, K),
                dtype=_mi_f32,
                name=f"{self.prefix}topk_weights_out",
            )
        elif isinstance(topk_weights_out, torch.Tensor):
            topk_weights_out_dt = pk.attach_input(
                topk_weights_out, name=f"{self.prefix}topk_weights_out"
            )
        else:
            topk_weights_out_dt = topk_weights_out

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(hidden_states, (0, -1, -1), 1, True)
        tb_graph.new_input(topk_ids, (0, -1, -1), 1, True)
        tb_graph.new_input(topk_weights, (0, -1, -1), 1, True)
        tb_graph.new_input(x_fp8_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(x_sf_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(topk_idx_out_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(topk_weights_out_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized(
            [
                hidden_states,
                topk_ids,
                topk_weights,
                x_fp8_dt,
                x_sf_dt,
                topk_idx_out_dt,
                topk_weights_out_dt,
            ],
            tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "prepare_megamoe_inputs_v4_sm100")
        return x_fp8_dt, x_sf_dt, topk_idx_out_dt, topk_weights_out_dt
