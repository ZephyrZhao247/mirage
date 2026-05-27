"""DeepSeek V4-Flash Indexer per-step Q transform.

Backed by ``tasks/blackwell/indexer_q_transform_sm100.cuh`` (task name
``"indexer_q_transform_sm100"``).

Per token ``t``, per index head ``h`` (H = INDEX_N_HEADS = 64,
D = INDEX_HEAD_DIM = 128 in V4-Flash):

  1. **Q expand via wq_b** (fp32 GEMV)::

        q_idx[t, h, d] = sum_k q_lora[t, k] * wq_b[h*D + d, k]

     where ``q_lora`` is the q-LoRA-A normalized output (output of
     ``mla_v4_q_kv_rmsnorm`` upstream, shape ``[T, q_lora_rank=1024]``)
     and ``wq_b`` is ASSUMED PRE-HADAMARD-ABSORBED at convert time (the
     stored weight is ``H · wq_b`` for the 128x128 Sylvester Hadamard
     ``H`` — i.e., the runtime needs no Hadamard pass). See §F.2 of
     ``docs/mpk/deepseek_v4/sparse.md``.

  2. **GPT-J interleaved RoPE** on the last ``rope_dim`` channels with
     ``compress_rope_theta = 160000`` (the cos/sin table is precomputed
     once per layer at this theta and passed as ``cos_sin_cache``).

  3. **Per-block MXFP4 quant with UE8M0 scales** on the full
     ``index_head_dim`` row, in blocks of ``block_size = 32``::

        amax  = max(|q_block|)
        exp   = ceil(log2(max(amax, eps) / 6.0))   # 6 == E2M1 max
        scale = 2^exp                               # power-of-two
        ue8m0 = clamp(exp + 127, [0, 255])          # uint8 byte
        v_e2m1 = quantize_e2m1(q / scale)           # 4-bit nibble code

     Two E2M1 nibbles pack into one uint8 byte::

        byte[d/2] bits[3:0] = code for d_even (lower d of the pair)
        byte[d/2] bits[7:4] = code for d_odd

     The E2M1 magnitude levels are
     ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}`` with a sign bit in the MSB.

V1 simplification: Hadamard is **absorbed offline into wq_b** at
convert time — no runtime Hadamard matmul. The next-layer
``indexer_score_topk`` task consumes ``(q_fp4, q_scale)`` directly.

Outputs:

  ``q_fp4``   ``[T, H, D/2]``    uint8   — packed MXFP4/E2M1 nibbles
  ``q_scale`` ``[T, H, D/32]``   uint8   — UE8M0 exponent bytes
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["IndexerQTransform"]


# E2M1 element magnitudes (positive part). Indices map 1:1 to the 4-bit
# magnitude code (0..7); the sign bit (bit 3) is separately set when the
# value is negative.
_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MAX = 6.0
_MXFP4_BLOCK_SIZE = 32
_EPS = 1e-30


def _quantize_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Round each element of ``x`` (fp32) to the nearest E2M1 magnitude
    code (0..7), preserving sign. Returns a ``uint8`` tensor of the same
    shape containing the 4-bit code (with the sign bit in bit 3).

    Midpoints between consecutive levels are
    ``[0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0]``; we use these as
    breakpoints (round-half-to-even isn't required for parity with the
    kernel because the CUDA code uses the same comparison breakpoints).
    """
    sign_bit = (x < 0).to(torch.uint8) << 3
    a = x.abs()
    # 7 breakpoints (between the 8 levels). torch.bucketize uses the
    # right-open convention: index i if a < boundaries[i]; ==
    # boundaries[i] goes to bucket i+1 (acceptable here; ties are rare).
    boundaries = torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        dtype=a.dtype,
        device=a.device,
    )
    mag = torch.bucketize(a, boundaries).to(torch.uint8)
    mag.clamp_(0, 7)
    return sign_bit | mag


def _e2m1_dequantize(code: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`_quantize_e2m1`. Returns the fp32 value
    represented by each 4-bit code (lower 4 bits used)."""
    levels = torch.tensor(_E2M1_LEVELS, dtype=torch.float32, device=code.device)
    mag = (code & 0x7).long()
    sign = ((code >> 3) & 0x1).to(torch.float32)
    val = levels[mag]
    return val * (1.0 - 2.0 * sign)


def _apply_gptj_rope(
    q: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Forward GPT-J interleaved RoPE on the trailing rope_dim channels.

    ``q``  : ``[..., rope_dim]`` fp32 contiguous.
    ``cos`` / ``sin`` : ``[..., rope_dim / 2]``.

    Per pair ``(q[2k], q[2k+1])`` -> ``(q[2k]*cos - q[2k+1]*sin,
    q[2k+1]*cos + q[2k]*sin)``.
    """
    q0 = q[..., 0::2]
    q1 = q[..., 1::2]
    new0 = q0 * cos - q1 * sin
    new1 = q1 * cos + q0 * sin
    out = torch.empty_like(q)
    out[..., 0::2] = new0
    out[..., 1::2] = new1
    return out


class IndexerQTransform(MPKModule):
    """Indexer per-step Q transform: wq_b GEMV + RoPE + MXFP4/UE8M0 quant.

    Constructor args:
      q_lora_rank    : ``K`` — q-LoRA-A rank (1024 in V4-Flash).
      index_n_heads  : ``H`` — number of indexer heads (64 in V4-Flash).
      index_head_dim : ``D`` — per-indexer-head dim (128 in V4-Flash).
                       Must be a multiple of 32 (MXFP4 block size).
      rope_dim       : RoPE-channel count on the trailing axis (default
                       64, matching V4-Flash). Must be even and ``<= D``.
      rope_theta     : Documented for the user (the cos/sin table is
                       precomputed externally with this theta); stored
                       as ``float`` on the module for reference. Default
                       is ``compress_rope_theta = 160000``.
      prefix         : MPK kernel-tensor name prefix / state_dict prefix.

    State:
      wq_b (nn.Parameter): ``[H * D, q_lora_rank]`` bf16. Assumed
        Hadamard-pre-absorbed at convert time so the runtime kernel
        produces an already-rotated Q.
    """

    def __init__(
        self,
        q_lora_rank: int,
        index_n_heads: int,
        index_head_dim: int,
        rope_dim: int = 64,
        rope_theta: float = 160000.0,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if q_lora_rank <= 0:
            raise ValueError(f"q_lora_rank must be > 0; got {q_lora_rank}")
        if index_n_heads <= 0:
            raise ValueError(
                f"index_n_heads must be > 0; got {index_n_heads}"
            )
        if index_head_dim <= 0 or index_head_dim % _MXFP4_BLOCK_SIZE != 0:
            raise ValueError(
                f"index_head_dim={index_head_dim} must be a positive "
                f"multiple of {_MXFP4_BLOCK_SIZE}"
            )
        if rope_dim <= 0 or rope_dim % 2 != 0:
            raise ValueError(
                f"rope_dim={rope_dim} must be a positive even integer"
            )
        if rope_dim > index_head_dim:
            raise ValueError(
                f"rope_dim={rope_dim} cannot exceed "
                f"index_head_dim={index_head_dim}"
            )
        self.q_lora_rank = q_lora_rank
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.rope_dim = rope_dim
        self.rope_theta = float(rope_theta)
        # wq_b is Hadamard-absorbed at convert time. Initialized empty so
        # ``state_dict`` load fills it; tests overwrite directly.
        self.wq_b = nn.Parameter(
            torch.empty(
                index_n_heads * index_head_dim,
                q_lora_rank,
                dtype=torch.bfloat16,
            )
        )

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        q_lora: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reference (PyTorch) implementation.

        Mirrors the CUDA kernel: matmul fp32 -> GPT-J RoPE -> per-block
        MXFP4 quant with UE8M0 power-of-two scales.

        Returns ``(q_fp4 [T, H, D/2] uint8, q_scale [T, H, D/32] uint8)``.
        """
        if q_lora.dim() != 2 or q_lora.shape[1] != self.q_lora_rank:
            raise ValueError(
                f"q_lora must have shape [T, q_lora_rank={self.q_lora_rank}]"
                f"; got {tuple(q_lora.shape)}"
            )
        T = q_lora.shape[0]
        if (cos_sin_cache.dim() != 2
                or cos_sin_cache.shape[-1] != self.rope_dim):
            raise ValueError(
                f"cos_sin_cache must have shape "
                f"[max_pos, rope_dim={self.rope_dim}]; got "
                f"{tuple(cos_sin_cache.shape)}"
            )
        if positions.dim() != 1 or positions.shape[0] != T:
            raise ValueError(
                f"positions must have shape [T={T}]; got "
                f"{tuple(positions.shape)}"
            )

        H = self.index_n_heads
        D = self.index_head_dim
        device = q_lora.device

        # Step 1: matmul in fp32 — q_idx = q_lora @ wq_b.T.
        # wq_b: [H*D, K] -> q_lora @ wq_b.T -> [T, H*D] -> reshape to
        # [T, H, D].
        q_idx = torch.matmul(
            q_lora.float(), self.wq_b.float().t()
        ).view(T, H, D)

        # Step 2: GPT-J interleaved RoPE on the last rope_dim channels.
        nope = D - self.rope_dim
        half_rope = self.rope_dim // 2
        cs = cos_sin_cache.to(device=device, dtype=torch.float32)
        cos_full = cs[:, :half_rope]   # [max_pos, half_rope]
        sin_full = cs[:, half_rope:]   # [max_pos, half_rope]
        pos = positions.to(device=device).long()
        cos = cos_full.index_select(0, pos)[:, None, :].expand(T, H, half_rope)
        sin = sin_full.index_select(0, pos)[:, None, :].expand(T, H, half_rope)

        if self.rope_dim > 0:
            rope_tail = q_idx[..., nope:]
            rotated = _apply_gptj_rope(rope_tail, cos, sin)
            q_idx = torch.cat([q_idx[..., :nope], rotated], dim=-1)

        # Step 3: Per-block MXFP4 quant with UE8M0 scale.
        num_blocks = D // _MXFP4_BLOCK_SIZE
        blocks = q_idx.view(T, H, num_blocks, _MXFP4_BLOCK_SIZE)
        amax = blocks.abs().amax(dim=-1)                  # [T, H, num_blocks]
        amax_safe = torch.clamp(amax, min=_EPS)
        # Exponent: ceil(log2(amax / 6.0)), clamp into the UE8M0 byte range.
        exp_f = torch.ceil(torch.log2(amax_safe / _E2M1_MAX))
        exp_i = exp_f.to(torch.int32).clamp_(-127, 128)
        ue8m0 = (exp_i + 127).clamp_(0, 255).to(torch.uint8)
        scale = torch.pow(2.0, exp_i.to(torch.float32))
        inv_scale = 1.0 / scale
        scaled_blocks = blocks * inv_scale.unsqueeze(-1)
        scaled_blocks = torch.clamp(scaled_blocks, -_E2M1_MAX, _E2M1_MAX)
        codes = _quantize_e2m1(scaled_blocks)             # [T, H, num_blocks, BLOCK]

        # Step 4: Pack pairs of nibbles into bytes along the trailing axis.
        # codes layout is [..., BLOCK] with BLOCK=32. Reshape to view pairs
        # (even, odd), then pack: byte = even | (odd << 4).
        codes_flat = codes.view(T, H, D)                  # [T, H, D]
        even = codes_flat[..., 0::2] & 0xF
        odd = codes_flat[..., 1::2] & 0xF
        q_fp4 = (even | (odd << 4)).to(torch.uint8)       # [T, H, D/2]
        return q_fp4, ue8m0.contiguous()

    # ------------------------------------------------------------------
    # MPK plumbing
    # ------------------------------------------------------------------
    def auto_grid_dim(self, q_lora_dt: DTensor) -> GridDim:
        """One CTA per token; the CTA derives its ``t`` from
        ``task_metadata.token_offset`` and loops over all H heads
        internally.
        """
        from ... import context as _ctx
        pk = _ctx.current_pk()
        n = q_lora_dt.dim(0)
        return (max(1, min(n, pk.num_workers)), 1, 1)

    def default_block_dim(self) -> BlockDim:
        """The kernel needs ``index_head_dim`` lanes (one thread per
        output channel); higher lanes are gated inside the kernel.
        Returns the standard worker-block width so the runtime's
        surrounding ``__syncthreads()`` is well-defined.
        """
        from ... import context as _ctx
        pk = _ctx.current_pk()
        return (128, 1, 1) if pk.target_cc < 90 else (256, 1, 1)

    def compile(
        self,
        q_lora: DTensor,
        cos_sin_cache: Union[torch.Tensor, DTensor],
        positions: Union[torch.Tensor, DTensor],
        *,
        q_fp4: Optional[Union[torch.Tensor, DTensor]] = None,
        q_scale: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register one ``indexer_q_transform_sm100`` task.

        Tensor contract:
          q_lora        : ``[T, q_lora_rank]``                 bf16, input.
          wq_b          : ``[H*D, q_lora_rank]``               bf16,
                          owned by ``self`` (Hadamard-pre-absorbed).
          cos_sin_cache : ``[max_pos, rope_dim]``              bf16.
                          ``cos`` in ``[:, :rope_dim/2]``, ``sin`` in
                          ``[:, rope_dim/2:]``.
          positions     : ``[T]``                              int32.
          q_fp4         : ``[T, H, D/2]``                      uint8 OUT.
          q_scale       : ``[T, H, D/32]``                     uint8 OUT.

        Returns ``(q_fp4_dt, q_scale_dt)``.
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        prefix = self.prefix or "indexer_q_transform_"

        # Validate q_lora.
        if q_lora.num_dims != 2:
            raise ValueError(
                "IndexerQTransform.compile expects a 2-D q_lora DTensor; "
                f"got num_dims={q_lora.num_dims}"
            )
        if q_lora.dim(1) != self.q_lora_rank:
            raise ValueError(
                f"q_lora.dim(1)={q_lora.dim(1)} does not match "
                f"q_lora_rank={self.q_lora_rank}"
            )
        T = q_lora.dim(0)

        # Helper: resolve an auxiliary input (Tensor or DTensor) to a
        # DTensor handle with the expected dtype.
        def _attach_in(buf, expected_dtype_torch: torch.dtype,
                       name: str) -> DTensor:
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

        # Attach wq_b (persistent weight owned by the catalog module).
        wq_b_dt = pk.attach_input(self.wq_b.data, name=f"{prefix}wq_b")

        cos_sin_cache_dt = _attach_in(
            cos_sin_cache, torch.bfloat16, f"{prefix}cos_sin_cache"
        )
        positions_dt = _attach_in(
            positions, torch.int32, f"{prefix}positions"
        )

        # Validate aux shapes.
        if (cos_sin_cache_dt.num_dims != 2
                or cos_sin_cache_dt.dim(1) != self.rope_dim):
            raise ValueError(
                f"cos_sin_cache.dim(1)={cos_sin_cache_dt.dim(1)} "
                f"does not match rope_dim={self.rope_dim}"
            )
        if positions_dt.num_dims != 1 or positions_dt.dim(0) != T:
            raise ValueError(
                f"positions must have shape [T={T}]; got "
                f"({positions_dt.dim(0)},)"
            )
        assert wq_b_dt.num_dims == 2
        assert wq_b_dt.dim(0) == self.index_n_heads * self.index_head_dim
        assert wq_b_dt.dim(1) == self.q_lora_rank

        # Output buffers.
        H = self.index_n_heads
        D = self.index_head_dim
        half_D = D // 2
        num_scale_blocks = D // _MXFP4_BLOCK_SIZE

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

        q_fp4_dt = _attach_out(
            q_fp4,
            (T, H, half_D),
            torch.uint8,
            mi.uint8,
            f"{prefix}q_fp4",
        )
        q_scale_dt = _attach_out(
            q_scale,
            (T, H, num_scale_blocks),
            torch.uint8,
            mi.uint8,
            f"{prefix}q_scale",
        )
        assert q_fp4_dt.num_dims == 3
        assert (q_fp4_dt.dim(0) == T and q_fp4_dt.dim(1) == H
                and q_fp4_dt.dim(2) == half_D)
        assert q_scale_dt.num_dims == 3
        assert (q_scale_dt.dim(0) == T and q_scale_dt.dim(1) == H
                and q_scale_dt.dim(2) == num_scale_blocks)

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(q_lora)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # All inputs/outputs are addressed via task_metadata.token_offset
        # (not via TBGraph partitioning), so all map dims are (-1, -1, -1)
        # — same convention as inv_rope_fp8_quant_o_sm100 /
        # mla_v4_q_kv_rmsnorm_sm100.
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_lora, (-1, -1, -1), -1, True)
        tb_graph.new_input(wq_b_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(cos_sin_cache_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(positions_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_fp4_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_scale_dt, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [q_lora, wq_b_dt, cos_sin_cache_dt, positions_dt,
             q_fp4_dt, q_scale_dt],
            tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "indexer_q_transform_sm100")
        return q_fp4_dt, q_scale_dt
