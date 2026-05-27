"""DeepSeek V4-Flash Compressor *compress* step catalog module (Wave-2 C2).

Backed by ``tasks/blackwell/compressor_compress_sm100.cuh`` (task name
``"compressor_compress_sm100"``). Sibling of ``CompressorStateUpdate``
(``compressor_state_update_sm100``).

The Compressor compress kernel reads from a state ring buffer (filled by
the state-update sibling task every step) every ``compress_ratio``
tokens, applies gated softmax pooling, RMSNorm, RoPE, and FP8 quant, then
writes a single compressed KV entry to the compressed paged KV cache.

Math (per boundary token at absolute position ``p``, position-in-cache
``p // compress_ratio``):

.. code-block:: text

    state    = state_cache[batch, :W, :2 * head_dim]      # bf16
    kv_part  = state[:, :head_dim]
    score    = state[:, head_dim:2*head_dim] + ape        # ape: bf16 [R, D]
    w        = softmax(score, dim=0)                       # along window
    pooled   = sum_w(kv_part[w, :] * w[w, :])              # fp32
    rrms     = rsqrt(mean(pooled**2) + eps)
    normed   = pooled * rrms * norm_weight                 # RMSNorm
    rotated  = apply_rope(normed, p // compress_ratio, theta=rope_theta)
    out_fp8  = quantize_fp8(rotated[:head_dim - rope_dim])
    out_rope = rotated[head_dim - rope_dim:]               # bf16
    out_scale= per-block fp32 scales
    kv_cache[p // compress_ratio] = [out_fp8 | out_rope | out_scale]

Reference call sites:

  * vLLM Triton kernel:
    ``deps/vllm/vllm/model_executor/layers/deepseek_compressor.py`` —
    fused RMSNorm + RoPE + FP8 insert.
  * Official:
    ``deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`` —
    ``Compressor.forward`` compress branch.
  * Spec: ``docs/mpk/deepseek_v4/sparse.md`` § ``compressor_layer``.

V1 simplifications (relative to the production spec):

  * Single batch — ``state_cache.dim(0)`` indexes the batch but the v1
    kernel assumes ``batch == token_offset`` (no separate slot_mapping or
    block_table indirection). Multi-batch / paged-state plumbing is a v2
    follow-up; the API reserves ``state_cache`` as a 3-D tensor with a
    leading ``B`` axis so the same Python contract works once the kernel
    adds paged indexing.
  * Per-block scales are stored as **fp32** (4 bytes / block) rather than
    UE8M0 single-byte exponents. The math reads ``absmax / 448`` directly
    rather than encoding the ceil-log2 exponent. UE8M0 is a v2 follow-up.
  * The compressed-KV cache slot is a contiguous
    ``[NOPE_DIM | 2*ROPE_DIM | 4*NUM_SCALE_BLOCKS]`` byte sequence; the
    production layout splits the scale region into a separate trailing
    page section (``bs*576 + bs*8``). The v1 contiguous layout matches
    the prompt's spec.
"""
from __future__ import annotations

import struct
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["CompressorCompress"]


# FP8 E4M3 max representable absolute value.
_FP8_MAX = 448.0
_EPS = 1e-12


def _quant_block_size(nope_dim: int) -> int:
    """Pick a default per-block FP8 quant width.

    The production V4 config uses ``BLOCK_SIZE = 64`` over a 448-wide
    nope, giving 7 blocks of 64. For the small test sizes (nope=48)
    we fall back to the full nope width (1 block), which is the
    smallest valid quant granularity.
    """
    if nope_dim % 64 == 0:
        return 64
    return nope_dim


class CompressorCompress(MPKModule):
    """V4-Flash Compressor compress catalog module (v1).

    Constructor args:
      head_dim        : full per-head width (e.g. 512 in V4-Flash).
      rope_dim        : trailing rope width (e.g. 64). Must be even.
      compress_ratio  : softmax window stride (e.g. 4 or 128).
      overlap         : True ⇒ window = 2 * compress_ratio (coff=2); else
                         window = compress_ratio (coff=1).
      rope_theta      : RoPE base used by this layer. V4-Flash uses
                         ``compress_rope_theta = 160000`` (distinct from
                         the base ``rope_theta = 10000`` used by SWA).
                         The cos/sin cache passed to ``forward`` / ``compile``
                         must be pre-computed at this theta.
      eps             : RMSNorm epsilon.
      prefix          : MPK kernel-tensor name prefix.

    State:
      norm_weight (nn.Parameter) : bf16 ``[head_dim]`` — RMSNorm weight,
        initialised to ones so a brand-new module is a numeric identity
        on the RMSNorm step.
    """

    def __init__(
        self,
        head_dim: int,
        rope_dim: int,
        compress_ratio: int,
        overlap: bool = True,
        rope_theta: float = 160000.0,
        eps: float = 1e-6,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0; got {head_dim}")
        if rope_dim <= 0 or rope_dim % 2 != 0:
            raise ValueError(
                f"rope_dim must be a positive even int; got {rope_dim}"
            )
        if rope_dim > head_dim:
            raise ValueError(
                f"rope_dim={rope_dim} cannot exceed head_dim={head_dim}"
            )
        if compress_ratio <= 0:
            raise ValueError(
                f"compress_ratio must be > 0; got {compress_ratio}"
            )
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.nope_dim = head_dim - rope_dim
        self.compress_ratio = compress_ratio
        self.overlap = bool(overlap)
        self.window = (2 if self.overlap else 1) * compress_ratio
        self.rope_theta = float(rope_theta)
        self.eps = float(eps)

        # FP8 quant block size over the NoPE region.
        self.block_size = _quant_block_size(self.nope_dim)
        if self.nope_dim % self.block_size != 0:
            raise ValueError(
                f"nope_dim={self.nope_dim} not divisible by "
                f"block_size={self.block_size}"
            )
        self.num_scale_blocks = self.nope_dim // self.block_size
        # Per-slot byte stride. fp32 scales = 4 bytes/block.
        self.slot_bytes = (
            self.nope_dim + 2 * self.rope_dim + 4 * self.num_scale_blocks
        )

        # Trainable per-channel RMSNorm weight, default identity.
        self.norm_weight = nn.Parameter(
            torch.ones(head_dim, dtype=torch.bfloat16)
        )

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        state_cache: torch.Tensor,
        ape: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Reference (PyTorch) implementation.

        Args:
          state_cache   : ``[B, window, 2 * head_dim]`` bf16. The leading
                            ``head_dim`` of the trailing axis is the
                            ``kv_part``; the next ``head_dim`` is the
                            ``score_part`` (without ape).
          ape           : ``[compress_ratio, head_dim]`` bf16.
          cos_sin_cache : ``[max_pos, rope_dim]`` bf16 — cos in
                            ``[:, :rope_dim/2]``, sin in ``[:, rope_dim/2:]``.
          positions     : ``[T]`` int32 — absolute position for each
                            compress-triggering token. In v1 we treat the
                            batch index as ``i``, the token index in
                            ``positions``; the v2 path passes a separate
                            batch index.

        Returns ``compressed_kv_cache`` of shape ``[T, slot_bytes]``
        (``uint8``) where ``T = positions.shape[0]``. Caller is
        responsible for stitching this into the paged compressed-KV
        cache via the ``p // compress_ratio`` row offsets.
        """
        if state_cache.dim() != 3:
            raise ValueError(
                "state_cache must be [B, window, 2*head_dim]; got "
                f"shape {tuple(state_cache.shape)}"
            )
        B, W, K = state_cache.shape
        if W != self.window:
            raise ValueError(
                f"state_cache window dim {W} != configured "
                f"window={self.window} (compress_ratio={self.compress_ratio}, "
                f"overlap={self.overlap})"
            )
        if K != 2 * self.head_dim:
            raise ValueError(
                f"state_cache trailing dim {K} != 2*head_dim="
                f"{2 * self.head_dim}"
            )
        if ape.dim() != 2 or ape.shape != (
            self.compress_ratio,
            self.head_dim,
        ):
            raise ValueError(
                f"ape must be [{self.compress_ratio}, {self.head_dim}]; "
                f"got {tuple(ape.shape)}"
            )
        if cos_sin_cache.dim() != 2 or cos_sin_cache.shape[-1] != self.rope_dim:
            raise ValueError(
                "cos_sin_cache must be [max_pos, rope_dim="
                f"{self.rope_dim}]; got {tuple(cos_sin_cache.shape)}"
            )
        if positions.dim() != 1:
            raise ValueError(
                f"positions must be 1-D; got shape {tuple(positions.shape)}"
            )

        device = state_cache.device
        T = positions.shape[0]
        if T > B:
            raise ValueError(
                f"positions has {T} entries but state_cache has only "
                f"B={B} batches; v1 maps token t→batch t."
            )

        # FP32 everywhere for numerical stability.
        sc = state_cache.float()                         # [B, W, 2D]
        kv = sc[..., : self.head_dim]                    # [B, W, D]
        score = sc[..., self.head_dim:]                  # [B, W, D]
        ape_f = ape.float()                              # [R, D]
        # Broadcast ape across the window axis modulo compress_ratio.
        w_idx = torch.arange(self.window, device=device) % self.compress_ratio
        ape_full = ape_f[w_idx]                          # [W, D]

        positions_long = positions.to(device=device).long()
        cos_full = cos_sin_cache.float()[:, : self.rope_dim // 2]
        sin_full = cos_sin_cache.float()[:, self.rope_dim // 2:]

        out_uint8 = torch.zeros(T, self.slot_bytes, dtype=torch.uint8,
                                 device=device)

        for t in range(T):
            b = t  # v1: batch = token offset
            pos = int(positions_long[t].item())
            pos_compress = pos // self.compress_ratio

            kv_t = kv[b]                                  # [W, D]
            score_t = score[b] + ape_full                 # [W, D]
            weights = torch.softmax(score_t, dim=0)       # [W, D]
            pooled = (kv_t * weights).sum(dim=0)          # [D]

            # RMSNorm.
            rrms = torch.rsqrt((pooled * pooled).mean() + self.eps)
            normed = pooled * rrms * self.norm_weight.float()

            # GPT-J interleaved RoPE on the trailing rope_dim.
            nope = normed[: self.nope_dim]
            rope_tail = normed[self.nope_dim:]
            x = rope_tail[0::2]
            y = rope_tail[1::2]
            cos = cos_full[pos_compress]
            sin = sin_full[pos_compress]
            new_x = x * cos - y * sin
            new_y = y * cos + x * sin
            rotated_tail = torch.empty_like(rope_tail)
            rotated_tail[0::2] = new_x
            rotated_tail[1::2] = new_y

            # Per-block FP8 quant on the nope region.
            nb = self.num_scale_blocks
            bs = self.block_size
            blocks = nope.view(nb, bs)
            absmax = blocks.abs().amax(dim=-1)            # [nb]
            scale = torch.clamp(absmax, min=_EPS) / _FP8_MAX
            inv_scale = 1.0 / scale
            quant = blocks * inv_scale.unsqueeze(-1)
            quant = torch.clamp(quant, -_FP8_MAX, _FP8_MAX)
            fp8_block = quant.view(self.nope_dim).to(torch.float8_e4m3fn)

            # Pack the slot bytes.
            fp8_bytes = fp8_block.view(torch.uint8)
            out_uint8[t, : self.nope_dim] = fp8_bytes
            rope_bf16 = rotated_tail.to(torch.bfloat16)
            rope_as_bytes = rope_bf16.view(torch.uint8)
            out_uint8[t, self.nope_dim : self.nope_dim + 2 * self.rope_dim] = (
                rope_as_bytes
            )
            scale_fp32 = scale.to(torch.float32).contiguous()
            scale_as_bytes = scale_fp32.view(torch.uint8)
            out_uint8[
                t,
                self.nope_dim + 2 * self.rope_dim:
                self.nope_dim + 2 * self.rope_dim
                + 4 * self.num_scale_blocks,
            ] = scale_as_bytes

        return out_uint8

    # ------------------------------------------------------------------
    # Grid / block heuristics
    # ------------------------------------------------------------------
    def auto_grid_dim(self, positions_dt: DTensor) -> GridDim:
        """One CTA per compress-triggering token; capped at num_workers."""
        from ... import context as _ctx
        pk = _ctx.current_pk()
        n = positions_dt.dim(0)
        return (max(1, min(n, pk.num_workers)), 1, 1)

    def default_block_dim(self) -> BlockDim:
        from ... import context as _ctx
        pk = _ctx.current_pk()
        return (128, 1, 1) if pk.target_cc < 90 else (256, 1, 1)

    # ------------------------------------------------------------------
    # MPK task registration
    # ------------------------------------------------------------------
    def compile(
        self,
        state_cache: Union[torch.Tensor, DTensor],
        ape: Union[torch.Tensor, DTensor],
        cos_sin_cache: Union[torch.Tensor, DTensor],
        positions: Union[torch.Tensor, DTensor],
        compressor_kv_cache: Union[torch.Tensor, DTensor],
        *,
        norm_weight_override: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
        kv_cache_token_stride_bytes: Optional[int] = None,
    ) -> DTensor:
        """Register one ``compressor_compress_sm100`` task.

        Tensor contract:
          state_cache         : ``[B, window, 2*head_dim]``  bf16, input.
          ape                 : ``[compress_ratio, head_dim]`` bf16, input.
          cos_sin_cache       : ``[max_pos, rope_dim]``      bf16, input.
          positions           : ``[T]``                       int32, input.
          compressor_kv_cache : ``[num_slots, slot_bytes]`` or
                                 ``[total_bytes]``            uint8, output.
          norm_weight         : ``[head_dim]``                bf16 (auto-
                                 attached from ``self.norm_weight``;
                                 override via ``norm_weight_override``).

        ``kv_cache_token_stride_bytes`` defaults to ``self.slot_bytes``;
        passing 0 makes the kernel use the same default.
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        prefix = self.prefix or "compressor_compress_"

        def _attach_in(
            buf: Union[torch.Tensor, DTensor],
            expected_dtype: torch.dtype,
            name: str,
        ) -> DTensor:
            if isinstance(buf, DTensor):
                return buf
            if isinstance(buf, torch.Tensor):
                if buf.dtype != expected_dtype:
                    raise ValueError(
                        f"{name} must have dtype {expected_dtype}; "
                        f"got {buf.dtype}"
                    )
                return pk.attach_input(buf, name=name)
            raise TypeError(
                f"{name} must be torch.Tensor or DTensor; got "
                f"{type(buf).__name__}"
            )

        state_cache_dt = _attach_in(
            state_cache, torch.bfloat16, f"{prefix}state_cache"
        )
        ape_dt = _attach_in(ape, torch.bfloat16, f"{prefix}ape")
        cos_sin_cache_dt = _attach_in(
            cos_sin_cache, torch.bfloat16, f"{prefix}cos_sin_cache"
        )
        positions_dt = _attach_in(
            positions, torch.int32, f"{prefix}positions"
        )

        # Validate shapes.
        if state_cache_dt.num_dims != 3:
            raise ValueError(
                "state_cache must be 3-D [B, window, 2*head_dim]; got "
                f"num_dims={state_cache_dt.num_dims}"
            )
        if state_cache_dt.dim(1) != self.window:
            raise ValueError(
                f"state_cache.dim(1)={state_cache_dt.dim(1)} != "
                f"window={self.window}"
            )
        if state_cache_dt.dim(2) != 2 * self.head_dim:
            raise ValueError(
                f"state_cache.dim(2)={state_cache_dt.dim(2)} != "
                f"2*head_dim={2 * self.head_dim}"
            )
        if ape_dt.num_dims != 2:
            raise ValueError(
                f"ape must be 2-D; got num_dims={ape_dt.num_dims}"
            )
        if ape_dt.dim(0) != self.compress_ratio:
            raise ValueError(
                f"ape.dim(0)={ape_dt.dim(0)} != compress_ratio="
                f"{self.compress_ratio}"
            )
        if ape_dt.dim(1) != self.head_dim:
            raise ValueError(
                f"ape.dim(1)={ape_dt.dim(1)} != head_dim={self.head_dim}"
            )
        if cos_sin_cache_dt.num_dims != 2:
            raise ValueError(
                "cos_sin_cache must be 2-D [max_pos, rope_dim]; got "
                f"num_dims={cos_sin_cache_dt.num_dims}"
            )
        if cos_sin_cache_dt.dim(1) != self.rope_dim:
            raise ValueError(
                f"cos_sin_cache.dim(1)={cos_sin_cache_dt.dim(1)} != "
                f"rope_dim={self.rope_dim}"
            )
        if positions_dt.num_dims != 1:
            raise ValueError(
                "positions must be 1-D; got num_dims="
                f"{positions_dt.num_dims}"
            )

        # Attach norm_weight (bf16 [head_dim]).
        nw_buf = (
            norm_weight_override
            if norm_weight_override is not None
            else self.norm_weight.data
        )
        if isinstance(nw_buf, torch.Tensor):
            if nw_buf.dtype != torch.bfloat16:
                raise ValueError(
                    f"norm_weight must be bf16; got {nw_buf.dtype}"
                )
            nw_dt = pk.attach_input(nw_buf, name=f"{prefix}norm_weight")
        elif isinstance(nw_buf, DTensor):
            nw_dt = nw_buf
        else:
            raise TypeError(
                "norm_weight_override must be torch.Tensor or DTensor"
            )
        if nw_dt.num_dims != 1 or nw_dt.dim(0) != self.head_dim:
            raise ValueError(
                f"norm_weight must be [head_dim={self.head_dim}]; got "
                f"shape with dim(0)={nw_dt.dim(0)}"
            )

        # Output cache: uint8 paged-ish slab. Accept either a 2-D
        # [num_slots, slot_bytes] tensor or a 1-D byte buffer. The kernel
        # treats it as a byte-addressed slab indexed by pos_compress.
        if isinstance(compressor_kv_cache, torch.Tensor):
            if compressor_kv_cache.dtype != torch.uint8:
                raise ValueError(
                    "compressor_kv_cache must be uint8; got "
                    f"{compressor_kv_cache.dtype}"
                )
            kv_cache_dt = pk.attach_input(
                compressor_kv_cache, name=f"{prefix}kv_cache"
            )
        elif isinstance(compressor_kv_cache, DTensor):
            kv_cache_dt = compressor_kv_cache
        else:
            raise TypeError(
                "compressor_kv_cache must be torch.Tensor or DTensor"
            )

        if kv_cache_token_stride_bytes is None:
            kv_cache_token_stride_bytes = self.slot_bytes

        # Pack the eps as int bits — the std::vector<int> param channel
        # carries the fp32 payload (memcpy-decoded in task_register.cc).
        eps_bits = struct.unpack(
            "i", struct.pack("f", float(self.eps))
        )[0]
        params = [
            int(self.head_dim),
            int(self.rope_dim),
            int(self.compress_ratio),
            1 if self.overlap else 0,
            int(self.block_size),
            int(eps_bits),
            int(kv_cache_token_stride_bytes),
        ]

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(positions_dt)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # All inputs/outputs addressed via task_metadata.token_offset.
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(state_cache_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(ape_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_sin_cache_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(nw_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(positions_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(kv_cache_dt, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [
                state_cache_dt,
                ape_dt,
                cos_sin_cache_dt,
                nw_dt,
                positions_dt,
                kv_cache_dt,
            ],
            tb_graph,
        )
        pk.kn_graph.register_task(
            tb_graph, "compressor_compress_sm100", params
        )
        return kv_cache_dt
