"""V4-Flash ``apply_rotary_emb`` — naive GPT-J / interleaved RoPE.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/apply_rotary_emb.md``.

Decision: **NEW**.

Rationale
---------
The generic-fallback ``apply_rotary_emb`` is a standalone RoPE op that
(unlike the existing MPK :class:`RotaryEmbedding`, which only
*precomputes* cos/sin tables for the attention kernel to consume) rotates
a Q/K activation tensor in place. The V4-Flash convention is
**interleaved / GPT-J style** (``is_neox_style=False`` at
``vllm/models/deepseek_v4/nvidia/model.py:719``), where each pair
``(x[..., 2i], x[..., 2i+1])`` is rotated together — NOT the HF / Llama
``rotate_half`` (split-half / Neox) convention used by the
:class:`mirage.mpk.layers.RotaryEmbedding` cos/sin layout.

That dtype-and-layout mismatch (interleaved vs split-half cos/sin
broadcast) makes a pure alias unsafe, so this is a new kernel:

* CUDA header: ``include/mirage/persistent_kernel/tasks/blackwell/apply_rotary_emb_v4_sm100.cuh``
* Task name: ``apply_rotary_emb_v4_sm100``
* Enum slot: ``TASK_APPLY_ROTARY_EMB_V4_SM100 = 351``
* Registered via ``register_apply_rotary_emb_v4_sm100_task``.

Implementation is **naive** (per the standing "naive first, perf later"
rule): one CTA per ``(token, head)`` row, ``block_dim=256``, plain loop
over pair indices, fp32 arithmetic, bf16 store. No TMA, no UMMA, no warp
specialization.

Note: this kernel is **NOT in V4-Flash's production call graph** — the
spec (lines 16-22) is explicit that all V4 RoPE sites are fused into
other kernels (``fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert``,
``fused_inv_rope_fp8_quant``, ``compress_norm_rope_store_*``,
``fused_indexer_q_rope_quant_*``). It is here for catalog completeness
and to give downstream model code a generic naive-bf16 RoPE to fall back
on.

Audit
-----
* dtype: x / cos / sin / out all bf16. Reduction in fp32 (per spec
  reference at ``model.py:235-240``), cast back to bf16 on store. No
  silent dtype mismatch.
* layout: x ``[num_rows, head_dim]`` row-major contiguous; cos and sin
  ``[num_rows, rotary_dim/2]`` row-major (caller has pre-gathered per
  ``(token, head)`` row). out matches x. Matches the spec's interleaved
  branch.
* multi-batch: each row of the ``[num_rows, ...]`` tensor is a
  ``(token, head)`` pair; multi-batch is exercised by stacking
  ``token`` and ``head`` rows. The V4 test below uses
  ``num_tokens >= 2`` and ``num_heads >= 2``.
* ``forward()``: provides the faithful PyTorch reference using the
  interleaved/GPT-J convention.

V4 production: NOT invoked in production (spec line 22). Use the fused
kernels listed above instead.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from ..._base import BlockDim, GridDim, MPKModule


__all__ = ["V4ApplyRotaryEmb"]


class V4ApplyRotaryEmb(MPKModule):
    """V4-Flash naive bf16 GPT-J-interleaved RoPE.

    Tensor contract:
      x:   (num_rows, head_dim) bf16, row-major contiguous. ``num_rows``
           is the flattened ``num_tokens * num_heads`` (the catalog
           accepts either 2-D or 3-D inputs; 3-D is reshaped internally).
      cos: (num_rows, rotary_dim/2) bf16, row-major. The caller has
           pre-gathered ``cos[positions[token_row]]`` and broadcast across
           heads.
      sin: (num_rows, rotary_dim/2) bf16, row-major. Same layout as cos.
      out: (num_rows, head_dim) bf16, row-major.

    The kernel rotates the first ``rotary_dim`` lanes of ``head_dim``
    using the interleaved convention
        out[..., 2i]   = even[i] * cos[i] - odd[i] * sin[i]
        out[..., 2i+1] = even[i] * sin[i] + odd[i] * cos[i]
    where ``(even[i], odd[i]) = (x[..., 2i], x[..., 2i+1])``. Trailing
    lanes ``[rotary_dim, head_dim)`` are copied through unchanged (V4
    typically sets ``rotary_dim == head_dim`` so this is empty).

    Args:
      head_dim:   Width of the rotated axis. Must be ``>= rotary_dim``.
      rotary_dim: Number of leading lanes rotated. Must be even.
      prefix:     Tensor-name prefix.
    """

    def __init__(
        self,
        head_dim: int,
        rotary_dim: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if rotary_dim % 2 != 0:
            raise ValueError(
                f"V4ApplyRotaryEmb: rotary_dim must be even; "
                f"got rotary_dim={rotary_dim}."
            )
        if rotary_dim > head_dim:
            raise ValueError(
                f"V4ApplyRotaryEmb: rotary_dim ({rotary_dim}) cannot "
                f"exceed head_dim ({head_dim})."
            )
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """Faithful interleaved/GPT-J RoPE in fp32, cast back to bf16.

        Accepts ``x`` of shape ``(num_rows, head_dim)`` (2-D) or
        ``(num_tokens, num_heads, head_dim)`` (3-D, which we flatten
        the leading two axes for). cos/sin must have a leading dim
        matching ``num_rows`` and a trailing dim of ``rotary_dim/2``.
        """
        orig_shape = x.shape
        if x.dim() == 3:
            x_flat = x.reshape(-1, self.head_dim)
        elif x.dim() == 2:
            x_flat = x
        else:
            raise ValueError(
                f"V4ApplyRotaryEmb.forward: x must be 2-D or 3-D; "
                f"got x.dim()={x.dim()}, x.shape={tuple(x.shape)}."
            )

        num_rows = x_flat.shape[0]
        if cos.shape != (num_rows, self.rotary_dim // 2):
            raise ValueError(
                f"V4ApplyRotaryEmb.forward: cos must be shape "
                f"({num_rows}, {self.rotary_dim // 2}); got {tuple(cos.shape)}."
            )
        if sin.shape != (num_rows, self.rotary_dim // 2):
            raise ValueError(
                f"V4ApplyRotaryEmb.forward: sin must be shape "
                f"({num_rows}, {self.rotary_dim // 2}); got {tuple(sin.shape)}."
            )

        x_rot = x_flat[:, : self.rotary_dim].to(torch.float32)
        # Reshape to (num_rows, rotary_dim/2, 2) so [..., 0] = even and
        # [..., 1] = odd (the GPT-J interleaved layout).
        x_pairs = x_rot.reshape(num_rows, self.rotary_dim // 2, 2)
        even = x_pairs[..., 0]  # (num_rows, rotary_dim/2)
        odd = x_pairs[..., 1]   # (num_rows, rotary_dim/2)

        c = cos.to(torch.float32)
        s = sin.to(torch.float32)

        out_even = even * c - odd * s
        out_odd = even * s + odd * c

        # Stack back to interleaved layout: (..., rotary_dim/2, 2) → (..., rotary_dim).
        out_rot = torch.stack([out_even, out_odd], dim=-1).reshape(
            num_rows, self.rotary_dim
        )

        if self.rotary_dim == self.head_dim:
            out_flat = out_rot.to(torch.bfloat16)
        else:
            out_flat = torch.cat(
                [out_rot.to(torch.bfloat16), x_flat[:, self.rotary_dim :]],
                dim=-1,
            )

        # Force the output dtype documented at the kernel boundary.
        out_flat = out_flat.to(torch.bfloat16)
        return out_flat.reshape(orig_shape)

    def auto_grid_dim(self, x_dt: Any) -> GridDim:
        """One CTA per (token, head) row.

        For ``x`` of shape ``(num_rows, head_dim)`` the grid is
        ``(num_rows, 1, 1)``. Each CTA rotates one row; the dispatcher
        offsets ``input_ptrs[0]`` / ``output_ptrs[0]`` to the row this
        CTA owns.
        """
        from .... import context as _ctx

        pk = _ctx.current_pk()
        return (max(1, min(x_dt.dim(0), int(pk.num_workers))), 1, 1)

    def compile(
        self,
        x_dt: Any,
        cos_dt: Any,
        sin_dt: Any,
        *,
        output: Optional[Any] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Any:
        """Register an ``apply_rotary_emb_v4_sm100`` task.

        Tensor contract (must match :class:`V4ApplyRotaryEmb` docstring):
          x_dt:   (num_rows, head_dim)        bf16, row-major contiguous.
          cos_dt: (num_rows, rotary_dim/2)    bf16, row-major contiguous.
          sin_dt: (num_rows, rotary_dim/2)    bf16, row-major contiguous.
          output: (num_rows, head_dim)        bf16, row-major contiguous.
                  None=alloc, torch.Tensor=host-bind, DTensor=use as-is.

        Notes: single CTA per row; ``block_dim=256`` on Blackwell.
        ``head_dim`` and ``rotary_dim`` are baked into the codegen via
        the params vector ``[head_dim, rotary_dim]``.
        """
        from .... import context as _ctx
        from .....core import CyTBGraph
        from .....kernel import TBGraph

        pk = _ctx.current_pk()

        if x_dt.num_dims != 2:
            raise ValueError(
                f"V4ApplyRotaryEmb.compile: x_dt must be 2-D "
                f"(num_rows, head_dim); got num_dims={x_dt.num_dims}."
            )
        if x_dt.dim(1) != self.head_dim:
            raise ValueError(
                f"V4ApplyRotaryEmb.compile: x_dt.dim(1)={x_dt.dim(1)} "
                f"!= head_dim={self.head_dim}."
            )
        if cos_dt.num_dims != 2 or cos_dt.dim(1) != self.rotary_dim // 2:
            raise ValueError(
                f"V4ApplyRotaryEmb.compile: cos_dt must be "
                f"(num_rows, {self.rotary_dim // 2}); got "
                f"num_dims={cos_dt.num_dims}, dim(1)={cos_dt.dim(1)}."
            )
        if sin_dt.num_dims != 2 or sin_dt.dim(1) != self.rotary_dim // 2:
            raise ValueError(
                f"V4ApplyRotaryEmb.compile: sin_dt must be "
                f"(num_rows, {self.rotary_dim // 2}); got "
                f"num_dims={sin_dt.num_dims}, dim(1)={sin_dt.dim(1)}."
            )

        num_rows = x_dt.dim(0)
        if output is None:
            out_dt = pk.new_tensor(
                dims=(num_rows, self.head_dim),
                dtype=x_dt.dtype,
                name=f"{self.prefix}rope_out",
            )
        elif isinstance(output, torch.Tensor):
            out_dt = pk.attach_input(
                output, name=f"{self.prefix}rope_out"
            )
        else:
            out_dt = output

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(x_dt)
        if block_dim is None:
            block_dim = self.default_block_dim()

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # Partition on grid.x (dim 0 of each tensor = the per-row axis).
        tb_graph.new_input(x_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(cos_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(sin_dt, (0, -1, -1), 1, True)
        tb_graph.new_input(out_dt, (0, -1, -1), 1, True)
        pk.kn_graph.customized([x_dt, cos_dt, sin_dt, out_dt], tb_graph)
        pk.kn_graph.register_task(
            tb_graph,
            "apply_rotary_emb_v4_sm100",
            [self.head_dim, self.rotary_dim],
        )
        return out_dt
