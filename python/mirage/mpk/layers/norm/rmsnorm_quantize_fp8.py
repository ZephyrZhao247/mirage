"""Fused RMSNorm + per-128-element-block FP8 (E4M3) quantization (Blackwell).

Backed by ``pk.fused_rmsnorm_quantize_fp8_layer`` →
``tasks/blackwell/fused_rmsnorm_quantize_fp8.cuh`` (task
``fused_rmsnorm_quantize_fp8_sm100``). Collapses the two-task chain
``RMSNorm`` + ``QuantizeFP8`` into one fused kernel, saving a dispatch
wave + a BF16 HBM round-trip per layer.

DeepSeek V3 uses this in two places (see
``models/deepseek_v3/builder.py``):

* **input-layernorm → qkv_a**: full-width norm, ``scale_ue8m0=False``
  (f32 scales consumed by the new dense FP8 GEMM family). ``emit_bf16``
  is usually ``False`` (qkv_a GEMM reads FP8 directly) but ``True`` when
  a downstream consumer needs the normalized BF16.
* **inner q_a-layernorm → q_b**: column-slice norm via ``process_dim``
  (``= q_lora_rank``), ``scale_ue8m0=False``, ``emit_bf16=False``.

The companion :class:`~mirage.mpk.layers.QuantizeFP8UE8M0` /
:class:`~mirage.mpk.layers.norm.RMSNorm` remain the unfused fallback.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from .._base import BlockDim, GridDim, MPKModule
from ..quantize_fp8 import _quantize_fp8_reference
from ...context import current_pk
from ....core import DTensor, bfloat16, float8_e4m3, float32, uint32


__all__ = ["FusedRMSNormQuantizeFP8"]


def _rmsnorm_grid(rows: int) -> GridDim:
    """One CTA per row (kernel partitions batch on grid.x).

    Mirrors ``_rmsnorm_grid`` in the DeepSeek V3 builder at the default
    ``ROWS_PER_TASK=1``: grid.x must divide ``rows`` (the kernel's
    ``BATCH_SIZE``). With one row per CTA that is trivially satisfied.
    """
    return (max(1, rows), 1, 1)


class FusedRMSNormQuantizeFP8(MPKModule):
    """RMSNorm (fp32 reduction, learnable scale) fused with FP8 quantization.

    Owns the RMSNorm ``weight`` (per-channel scale). ``compile`` registers
    a single ``fused_rmsnorm_quantize_fp8_sm100`` task that writes (up to)
    three outputs: the normalized BF16 (optional, gated by ``emit_bf16``),
    the FP8 E4M3 bytes, and the per-128-block scale.

    Constraints (from the .cuh / pk method):
      * SM100 (Blackwell) only; kernel hard-codes ``eps = 1e-6``.
      * ``process_dim`` (the quantized width) must be a multiple of 128.
      * ``scale_ue8m0=True`` → packed UE8M0 ``uint32`` scales (dense-GEMM
        column-major layout); ``False`` → ``float32`` scales ``(rows,
        process_dim // 128)`` row-major (DeepSeek V3 dense / MoE path).
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        scale_ue8m0: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.hidden_size = hidden_size
        self.eps = eps
        self.scale_ue8m0 = scale_ue8m0
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(
        self,
        x: torch.Tensor,
        *,
        process_dim: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Faithful reference: RMSNorm(fp32)·weight → BF16, then block-quantize.

        Returns ``(bf16, fp8_bytes, scale)``. When ``process_dim`` is set,
        only the first ``process_dim`` columns are normalized + quantized
        (the column-slice contract of the inner q_a fusion); the returned
        BF16 keeps the full input width with the tail copied through.
        """
        width = x.shape[-1] if process_dim is None else process_dim
        sliced = x[..., :width]
        variance = sliced.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
        normed = sliced.to(torch.float32) * torch.rsqrt(variance + self.eps)
        normed = (normed.to(x.dtype) * self.weight[:width]).to(x.dtype)

        bf16 = x.clone()
        bf16[..., :width] = normed
        fp8, scale = _quantize_fp8_reference(normed, self.scale_ue8m0)
        return bf16, fp8, scale

    def auto_grid_dim(self, x: DTensor) -> GridDim:
        return _rmsnorm_grid(x.dim(0))

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)

    def compile(
        self,
        x: DTensor,
        *,
        process_dim: Optional[int] = None,
        output_bf16: Optional[Union[torch.Tensor, DTensor]] = None,
        output_fp8: Optional[Union[torch.Tensor, DTensor]] = None,
        output_scale: Optional[Union[torch.Tensor, DTensor]] = None,
        emit_bf16: bool = True,
        scale_ue8m0: Optional[bool] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor, DTensor]:
        """Register one ``fused_rmsnorm_quantize_fp8_sm100`` task.

        Tensor contract (mirrors ``pk.fused_rmsnorm_quantize_fp8_layer``):
          x:            (rows, in_width) bf16.
          output_bf16:  (rows, in_width) bf16  (full width; written only if
                        ``emit_bf16``).
          output_fp8:   (rows, process_dim) float8_e4m3.
          output_scale: f32 path → (rows, process_dim // 128) float32.
                        ue8m0 path → (rows, process_dim // 128 // 4) uint32.

        ``process_dim`` defaults to ``x.dim(1)`` (full width).
        """
        pk = current_pk()
        if scale_ue8m0 is None:
            scale_ue8m0 = self.scale_ue8m0
        in_width = x.dim(1)
        width = process_dim if process_dim is not None else in_width
        if width % 128 != 0:
            raise ValueError(
                f"FusedRMSNormQuantizeFP8: process_dim={width} must be a "
                "multiple of 128 (FP8 block size)."
            )
        rows = x.dim(0)
        num_groups = width // 128

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x)
        if block_dim is None:
            block_dim = self.default_block_dim()

        w_dt = pk.attach_input(self.weight.data, name=f"{self.prefix}weight")

        def _resolve(out, dims, dtype, suffix):
            if out is None:
                return pk.new_tensor(dims=dims, dtype=dtype, name=f"{self.prefix}{suffix}")
            if isinstance(out, torch.Tensor):
                return pk.attach_input(out, name=f"{self.prefix}{suffix}")
            return out

        out_bf16_dt = _resolve(output_bf16, (rows, in_width), bfloat16, "fused_bf16")
        out_fp8_dt = _resolve(output_fp8, (rows, width), float8_e4m3, "fused_fp8")
        if scale_ue8m0:
            scale_dims = (rows, max(1, num_groups // 4))
            scale_dtype = uint32
        else:
            scale_dims = (rows, num_groups)
            scale_dtype = float32
        out_scale_dt = _resolve(output_scale, scale_dims, scale_dtype, "fused_scale")

        pk.fused_rmsnorm_quantize_fp8_layer(
            input=x,
            weight=w_dt,
            output_bf16=out_bf16_dt,
            output_fp8=out_fp8_dt,
            output_scale=out_scale_dt,
            grid_dim=grid_dim,
            block_dim=block_dim,
            process_dim=None if width == in_width else width,
            scale_ue8m0=scale_ue8m0,
            emit_bf16=emit_bf16,
        )
        return out_bf16_dt, out_fp8_dt, out_scale_dt
