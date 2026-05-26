"""DeepSeek V4-Flash MLA post-attention fused inverse-RoPE + per-block FP8 quant.

Backed by ``tasks/blackwell/inv_rope_fp8_quant_o_sm100.cuh`` (task name
``"inv_rope_fp8_quant_o_sm100"``).

For each token ``t`` and head ``h`` of the attention output
``o[T, H, head_dim]`` (bf16):

  1. **Inverse RoPE** on the last ``rope_dim`` channels using the
     GPT-J-interleaved pair convention. The forward GPT-J rotation is
     applied per pair ``(o[2k], o[2k+1])``. The inverse uses the conjugate
     ``freqs_cis`` (``+sin → -sin`` on the cross-term)::

        new o[2k]   = o[2k]   * cos[k] + o[2k+1] * sin[k]
        new o[2k+1] = o[2k+1] * cos[k] - o[2k]   * sin[k]

     The first ``head_dim - rope_dim`` (= nope) channels pass through
     unchanged. See
     ``deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:232-244``
     for the official ``apply_rotary_emb(..., inverse=True)`` reference.

  2. **Per-block FP8e4m3fn quantization** over the full ``head_dim`` row,
     grouped into ``head_dim / block_size`` blocks of ``block_size`` (128
     by default) elements. The scale is plain fp32 in v1::

        absmax = max(|x_block|)
        scale  = max(absmax, eps) / 448.0          # 448 = fp8_e4m3 max
        x_fp8  = clamp(x / scale, [-448, 448]).to(float8_e4m3fn)

     The UE8M0 packed-uint32 scale (vLLM TMA-aligned layout) is a
     follow-up — see the OPEN flag in the task spec.

Outputs:

  ``o_fp8``   ``[T, H, head_dim]``                fp8_e4m3fn (stored as uint8)
  ``o_scale`` ``[T, H, head_dim / block_size]``   fp32

``cos_sin_cache`` is laid out as ``[max_pos, rope_dim]`` with the first
``rope_dim / 2`` entries the cos table and the next ``rope_dim / 2`` the
sin table — matching the vLLM Triton convention used by the upstream V4
kernel.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["InvRopeFP8QuantO"]


# FP8 E4M3 max representable absolute value.
_FP8_MAX = 448.0
_EPS = 1e-12


def _inverse_rope_pair(
    o: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """GPT-J-interleaved inverse RoPE on the trailing channels.

    ``o``    : ``[..., rope_dim]`` float-promoted, contiguous.
    ``cos``  : ``[..., rope_dim / 2]``, cos table indexed at the same
                positions as ``o``'s leading axes.
    ``sin``  : ``[..., rope_dim / 2]``, sin table.

    Returns a new tensor of the same shape and dtype as ``o``.
    """
    # Pair (o0, o1) -> (o0*cos + o1*sin, o1*cos - o0*sin).
    o0 = o[..., 0::2]
    o1 = o[..., 1::2]
    new0 = o0 * cos + o1 * sin
    new1 = o1 * cos - o0 * sin
    out = torch.empty_like(o)
    out[..., 0::2] = new0
    out[..., 1::2] = new1
    return out


class InvRopeFP8QuantO(MPKModule):
    """Inverse-RoPE + per-block FP8 quant for the MLA attention output.

    Constructor args:
      num_heads  : ``H`` — number of attention heads (e.g. 64 in V4-Flash).
      head_dim   : ``D`` — full per-head dimension (e.g. 512 in V4-Flash).
      rope_dim   : ``D_rope`` — rope-channel count on the tail (e.g. 64).
      block_size : ``B`` — FP8 quant block size on the trailing axis;
                    must divide ``head_dim`` (default 128).
      prefix     : MPK kernel-tensor name prefix.

    No ``nn.Parameter`` is registered: ``cos_sin_cache`` is provided at
    compile/forward time, mirroring the upstream V4 kernel's signature.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        rope_dim: int,
        block_size: int = 128,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0; got {num_heads}")
        if head_dim <= 0 or head_dim % block_size != 0:
            raise ValueError(
                f"head_dim={head_dim} must be a positive multiple of "
                f"block_size={block_size}"
            )
        if rope_dim <= 0 or rope_dim % 2 != 0:
            raise ValueError(
                f"rope_dim={rope_dim} must be a positive even integer"
            )
        if rope_dim > head_dim:
            raise ValueError(
                f"rope_dim={rope_dim} cannot exceed head_dim={head_dim}"
            )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.block_size = block_size

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        o: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reference (PyTorch) implementation.

        Returns ``(o_fp8 [T, H, D] fp8_e4m3fn, o_scale [T, H, D/B] fp32)``.
        """
        if o.dim() != 3:
            raise ValueError(
                f"o must have shape [T, H, head_dim]; got {tuple(o.shape)}"
            )
        T, H, D = o.shape
        if H != self.num_heads or D != self.head_dim:
            raise ValueError(
                f"o shape {tuple(o.shape)} does not match num_heads="
                f"{self.num_heads}, head_dim={self.head_dim}"
            )
        if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[-1] != self.rope_dim:
            raise ValueError(
                f"cos_sin_cache must have shape [max_pos, rope_dim={self.rope_dim}]; "
                f"got {tuple(cos_sin_cache.shape)}"
            )
        if positions.dim() != 1 or positions.shape[0] != T:
            raise ValueError(
                f"positions must have shape [T={T}]; got {tuple(positions.shape)}"
            )
        device = o.device
        half_rope = self.rope_dim // 2

        # Split cos_sin_cache into halves per the vLLM convention:
        # first HALF_ROPE entries are cos, next HALF_ROPE are sin.
        cs = cos_sin_cache.to(device=device, dtype=torch.float32)
        cos_full = cs[:, :half_rope]    # [max_pos, half_rope]
        sin_full = cs[:, half_rope:]    # [max_pos, half_rope]

        # Per-token cos/sin lookup, then broadcast across heads.
        pos = positions.to(device=device).long()
        cos = cos_full.index_select(0, pos)[:, None, :].expand(T, H, half_rope)
        sin = sin_full.index_select(0, pos)[:, None, :].expand(T, H, half_rope)

        o_f32 = o.float()
        nope = self.head_dim - self.rope_dim
        rope_tail = o_f32[..., nope:]
        rotated = _inverse_rope_pair(rope_tail, cos, sin)
        o_rot = torch.cat([o_f32[..., :nope], rotated], dim=-1)

        # Per-block FP8 quant.
        num_blocks = self.head_dim // self.block_size
        blocks = o_rot.view(T, H, num_blocks, self.block_size)
        absmax = blocks.abs().amax(dim=-1)                 # [T, H, num_blocks]
        scale = torch.clamp(absmax, min=_EPS) / _FP8_MAX   # fp32
        x_scaled = blocks / scale.unsqueeze(-1)
        x_clamped = torch.clamp(x_scaled, -_FP8_MAX, _FP8_MAX)
        o_fp8 = x_clamped.to(torch.float8_e4m3fn).view(T, H, self.head_dim)
        return o_fp8, scale.to(torch.float32)

    # ------------------------------------------------------------------
    # MPK plumbing
    # ------------------------------------------------------------------
    def auto_grid_dim(self, o_dt: DTensor) -> GridDim:
        """One CTA per token; the CTA derives its ``t`` from
        ``task_metadata.token_offset`` and loops over all H heads
        internally.
        """
        from ... import context as _ctx
        pk = _ctx.current_pk()
        n = o_dt.dim(0)
        return (max(1, min(n, pk.num_workers)), 1, 1)

    def default_block_dim(self) -> BlockDim:
        """The kernel uses exactly one warp (NUM_THREADS=32). Higher
        lanes are gated off inside the kernel; we still launch the worker
        block dim so the runtime's surrounding ``__syncthreads()`` is
        well-defined.
        """
        from ... import context as _ctx
        pk = _ctx.current_pk()
        return (128, 1, 1) if pk.target_cc < 90 else (256, 1, 1)

    def compile(
        self,
        o: DTensor,
        cos_sin_cache: Union[torch.Tensor, DTensor],
        positions: Union[torch.Tensor, DTensor],
        *,
        o_fp8: Optional[Union[torch.Tensor, DTensor]] = None,
        o_scale: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register one ``inv_rope_fp8_quant_o_sm100`` task.

        Tensor contract:
          o             : ``[T, H, head_dim]``               bf16, input.
          cos_sin_cache : ``[max_pos, rope_dim]``            bf16,
                          (cos || sin) packed along the last dim.
          positions     : ``[T]``                            int32.
          o_fp8         : ``[T, H, head_dim]``               uint8 (fp8_e4m3fn).
          o_scale       : ``[T, H, head_dim / block_size]``  fp32.

        Notes: grid is ``(T, 1, 1)`` with one CTA per token; each CTA
        addresses its row via ``task_metadata.token_offset``. One param
        is passed: ``block_size`` (default 128). ``num_heads``,
        ``head_dim`` and ``rope_dim`` are inferred from the tensor shapes
        by the registration function.
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        prefix = self.prefix or "inv_rope_fp8_quant_o_"

        # Resolve DTensor handles for the auxiliary inputs.
        def _attach_in(
            buf: Union[torch.Tensor, DTensor],
            expected_dtype_torch: torch.dtype,
            name: str,
        ) -> DTensor:
            if isinstance(buf, DTensor):
                return buf
            if isinstance(buf, torch.Tensor):
                if buf.dtype != expected_dtype_torch:
                    raise ValueError(
                        f"{name} must have dtype {expected_dtype_torch}; "
                        f"got {buf.dtype}"
                    )
                return pk.attach_input(buf, name=name)
            raise TypeError(
                f"{name} must be torch.Tensor or DTensor; got "
                f"{type(buf).__name__}"
            )

        cos_sin_cache_dt = _attach_in(
            cos_sin_cache, torch.bfloat16, f"{prefix}cos_sin_cache"
        )
        positions_dt = _attach_in(
            positions, torch.int32, f"{prefix}positions"
        )

        # Validate input shapes.
        assert o.num_dims == 3
        t = o.dim(0)
        h = o.dim(1)
        d = o.dim(2)
        if h != self.num_heads:
            raise ValueError(
                f"o.dim(1)={h} does not match num_heads={self.num_heads}"
            )
        if d != self.head_dim:
            raise ValueError(
                f"o.dim(2)={d} does not match head_dim={self.head_dim}"
            )
        assert cos_sin_cache_dt.num_dims == 2
        if cos_sin_cache_dt.dim(1) != self.rope_dim:
            raise ValueError(
                f"cos_sin_cache.dim(1)={cos_sin_cache_dt.dim(1)} "
                f"does not match rope_dim={self.rope_dim}"
            )
        assert positions_dt.num_dims == 1
        if positions_dt.dim(0) != t:
            raise ValueError(
                f"positions.dim(0)={positions_dt.dim(0)} does not match "
                f"o.dim(0)={t}"
            )

        num_blocks = self.head_dim // self.block_size

        def _attach_out(buf, default_dims, default_dtype_torch,
                        default_dtype_mi, name):
            if buf is None:
                return pk.new_tensor(
                    dims=default_dims, dtype=default_dtype_mi, name=name
                )
            if isinstance(buf, torch.Tensor):
                if buf.dtype != default_dtype_torch:
                    raise ValueError(
                        f"{name} must have dtype {default_dtype_torch}; "
                        f"got {buf.dtype}"
                    )
                return pk.attach_input(buf, name=name)
            if isinstance(buf, DTensor):
                return buf
            raise TypeError(
                f"{name} must be None, torch.Tensor, or DTensor; got "
                f"{type(buf).__name__}"
            )

        o_fp8_dt = _attach_out(
            o_fp8,
            (t, h, d),
            torch.uint8,
            mi.uint8,
            f"{prefix}o_fp8",
        )
        o_scale_dt = _attach_out(
            o_scale,
            (t, h, num_blocks),
            torch.float32,
            mi.float32,
            f"{prefix}o_scale",
        )
        assert o_fp8_dt.num_dims == 3
        assert o_fp8_dt.dim(0) == t and o_fp8_dt.dim(1) == h
        assert o_fp8_dt.dim(2) == d
        assert o_scale_dt.num_dims == 3
        assert o_scale_dt.dim(0) == t and o_scale_dt.dim(1) == h
        assert o_scale_dt.dim(2) == num_blocks

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(o)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # All inputs/outputs are addressed via ``task_metadata.token_offset``
        # (not via TBGraph partitioning), so all map dims are (-1, -1, -1).
        # This matches the mhc_pre_sm100 / hash_route_lookup_sm100 convention.
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(o, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_sin_cache_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(positions_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(o_fp8_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(o_scale_dt, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [o, cos_sin_cache_dt, positions_dt, o_fp8_dt, o_scale_dt],
            tb_graph,
        )
        pk.kn_graph.register_task(
            tb_graph,
            "inv_rope_fp8_quant_o_sm100",
            [self.block_size],
        )
        return o_fp8_dt, o_scale_dt
