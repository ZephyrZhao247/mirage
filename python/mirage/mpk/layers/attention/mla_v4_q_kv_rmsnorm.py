"""DeepSeek V4-Flash joint Q-lora + KV-lora RMSNorm (pre-attention).

Backed by ``tasks/blackwell/mla_v4_q_kv_rmsnorm_sm100.cuh`` (task name
``"mla_v4_q_kv_rmsnorm_sm100"``).

V4-Flash's MLA attention RMSNorms two streams together pre-attention
with their own learnable per-channel weights:

    q_norm  = rmsnorm(q_lora,  q_norm_weight)    # [T, q_lora_rank=1024]
    kv_norm = rmsnorm(kv_lora, kv_norm_weight)   # [T, kv_lora_rank=512]

vLLM fuses these into one kernel pass (see
``deps/vllm/vllm/v1/attention/ops/deepseek_v4_ops/fused_qk_rmsnorm.py``)
so that the per-token reduction overhead is shared across the two
streams. Our MPK kernel runs them serially inside the same CTA (one CTA
per token), reusing the shared-memory reduction workspace.

Reference call site:
``deps/vllm/vllm/model_executor/layers/deepseek_v4_attention.py:409-415``

    qr, kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
    qr, kv = fused_q_kv_rmsnorm(
        qr, kv,
        self.q_norm.weight.data,
        self.kv_norm.weight.data,
        self.eps,
    )
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["MLAv4QKVRMSNorm"]


class MLAv4QKVRMSNorm(MPKModule):
    """Joint Q-lora + KV-lora RMSNorm for DeepSeek V4-Flash MLA.

    Constructor args:
      q_lora_rank   : ``Q_SIZE`` — width of the q_lora stream (1024 in V4-Flash).
      kv_lora_rank  : ``KV_SIZE`` — width of the kv_lora stream
                       (= ``head_dim = 512`` in V4-Flash; V4-Flash uses a
                       single 512-wide KV latent that doubles as the K/V
                       cache row).
      eps           : RMSNorm epsilon. Affects the PyTorch reference; the
                       compiled kernel currently bakes in ``1e-6``.
      prefix        : ``state_dict`` key prefix.

    Constraints (from the .cuh):
      * dtype: bf16 only.
      * Each rank must be a multiple of the kernel's NUM_THREADS (128 in v1;
        the registration function downgrades to 64 or 32 if either rank is
        smaller). V4-Flash sizes (1024, 512) both satisfy 128|rank.

    State:
      q_norm_weight  (nn.Parameter): ``[q_lora_rank]`` bf16.
      kv_norm_weight (nn.Parameter): ``[kv_lora_rank]`` bf16.
    """

    def __init__(
        self,
        q_lora_rank: int,
        kv_lora_rank: int,
        eps: float = 1e-6,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if q_lora_rank <= 0:
            raise ValueError(
                f"q_lora_rank must be > 0; got {q_lora_rank}"
            )
        if kv_lora_rank <= 0:
            raise ValueError(
                f"kv_lora_rank must be > 0; got {kv_lora_rank}"
            )
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.eps = eps
        # bf16 weights per the task spec — matches both the kernel and
        # the bf16-only model file.
        self.q_norm_weight = nn.Parameter(
            torch.ones(q_lora_rank, dtype=torch.bfloat16)
        )
        self.kv_norm_weight = nn.Parameter(
            torch.ones(kv_lora_rank, dtype=torch.bfloat16)
        )

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    @staticmethod
    def _rmsnorm_ref(
        x: torch.Tensor, weight: torch.Tensor, eps: float
    ) -> torch.Tensor:
        input_dtype = x.dtype
        x_f = x.to(torch.float32)
        var = x_f.pow(2).mean(dim=-1, keepdim=True)
        y = x_f * torch.rsqrt(var + eps)
        return (y.to(input_dtype) * weight).to(input_dtype)

    def forward(
        self, q_lora: torch.Tensor, kv_lora: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Faithful per-stream RMSNorm reference.

        Equivalent to applying ``RMSNorm.forward`` separately to ``q_lora``
        and ``kv_lora`` — which is exactly what the vLLM Triton kernel
        does (just fused for throughput).
        """
        if q_lora.dim() != 2:
            raise ValueError(
                f"q_lora must be 2-D; got shape {tuple(q_lora.shape)}"
            )
        if kv_lora.dim() != 2:
            raise ValueError(
                f"kv_lora must be 2-D; got shape {tuple(kv_lora.shape)}"
            )
        if q_lora.size(0) != kv_lora.size(0):
            raise ValueError(
                "q_lora and kv_lora must share the leading (token) "
                f"dimension; got {q_lora.size(0)} vs {kv_lora.size(0)}"
            )
        if q_lora.size(1) != self.q_lora_rank:
            raise ValueError(
                f"q_lora last dim {q_lora.size(1)} != q_lora_rank "
                f"{self.q_lora_rank}"
            )
        if kv_lora.size(1) != self.kv_lora_rank:
            raise ValueError(
                f"kv_lora last dim {kv_lora.size(1)} != kv_lora_rank "
                f"{self.kv_lora_rank}"
            )
        q_norm = self._rmsnorm_ref(q_lora, self.q_norm_weight, self.eps)
        kv_norm = self._rmsnorm_ref(kv_lora, self.kv_norm_weight, self.eps)
        return q_norm, kv_norm

    # ------------------------------------------------------------------
    # Grid / block heuristics
    # ------------------------------------------------------------------
    def auto_grid_dim(self, q_lora_dt: DTensor) -> GridDim:
        """One CTA per token; CTAs derive ``t`` from ``token_offset``."""
        from ... import context as _ctx
        pk = _ctx.current_pk()
        n = q_lora_dt.dim(0)
        return (max(1, min(n, pk.num_workers)), 1, 1)

    # ------------------------------------------------------------------
    # MPK task registration
    # ------------------------------------------------------------------
    def compile(
        self,
        q_lora: DTensor,
        kv_lora: DTensor,
        *,
        q_norm: Optional[Union[torch.Tensor, DTensor]] = None,
        kv_norm: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register one ``mla_v4_q_kv_rmsnorm_sm100`` task.

        Tensor contract:
          q_lora           : (T, q_lora_rank)  bf16  IN
          kv_lora          : (T, kv_lora_rank) bf16  IN
          q_norm_weight    : (q_lora_rank,)    bf16  IN  (attached from self)
          kv_norm_weight   : (kv_lora_rank,)   bf16  IN  (attached from self)
          q_norm           : (T, q_lora_rank)  bf16  OUT (auto-allocated if None)
          kv_norm          : (T, kv_lora_rank) bf16  OUT (auto-allocated if None)

        Returns ``(q_norm_dt, kv_norm_dt)``.
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        if q_lora.num_dims != 2:
            raise ValueError(
                "MLAv4QKVRMSNorm.compile expects a 2-D q_lora DTensor; "
                f"got num_dims={q_lora.num_dims}"
            )
        if kv_lora.num_dims != 2:
            raise ValueError(
                "MLAv4QKVRMSNorm.compile expects a 2-D kv_lora DTensor; "
                f"got num_dims={kv_lora.num_dims}"
            )
        if q_lora.dim(0) != kv_lora.dim(0):
            raise ValueError(
                "q_lora and kv_lora must share the leading (token) "
                f"dimension; got {q_lora.dim(0)} vs {kv_lora.dim(0)}"
            )
        if q_lora.dim(1) != self.q_lora_rank:
            raise ValueError(
                f"q_lora last dim {q_lora.dim(1)} != q_lora_rank "
                f"{self.q_lora_rank}"
            )
        if kv_lora.dim(1) != self.kv_lora_rank:
            raise ValueError(
                f"kv_lora last dim {kv_lora.dim(1)} != kv_lora_rank "
                f"{self.kv_lora_rank}"
            )

        n = q_lora.dim(0)
        prefix = self.prefix or "mla_v4_q_kv_rmsnorm_"

        # Attach the two persistent per-channel weights.
        q_weight_dt = pk.attach_input(
            self.q_norm_weight.data, name=f"{prefix}q_weight"
        )
        kv_weight_dt = pk.attach_input(
            self.kv_norm_weight.data, name=f"{prefix}kv_weight"
        )

        def _attach_out(buf, default_dims, name):
            if buf is None:
                return pk.new_tensor(
                    dims=default_dims, dtype=mi.bfloat16, name=name
                )
            if isinstance(buf, torch.Tensor):
                if buf.dtype != torch.bfloat16:
                    raise ValueError(
                        f"{name} must have dtype bfloat16; got {buf.dtype}"
                    )
                return pk.attach_input(buf, name=name)
            if isinstance(buf, DTensor):
                return buf
            raise TypeError(
                f"{name} must be None, torch.Tensor, or DTensor; got "
                f"{type(buf).__name__}"
            )

        q_norm_dt = _attach_out(
            q_norm, (n, self.q_lora_rank), f"{prefix}q_out"
        )
        kv_norm_dt = _attach_out(
            kv_norm, (n, self.kv_lora_rank), f"{prefix}kv_out"
        )

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(q_lora)
        if block_dim is None:
            block_dim = self.default_block_dim()

        # All inputs/outputs are addressed via ``task_metadata.token_offset``
        # (not via TBGraph partitioning), so all map dims are (-1, -1, -1).
        # This matches the mhc_pre_sm100 / hash_route_lookup_sm100
        # convention: the kernel receives base pointers and does its own
        # ``row = base + t * SIZE`` arithmetic.
        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_lora, (-1, -1, -1), -1, True)
        tb_graph.new_input(kv_lora, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_weight_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(kv_weight_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(q_norm_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(kv_norm_dt, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [q_lora, kv_lora, q_weight_dt, kv_weight_dt, q_norm_dt, kv_norm_dt],
            tb_graph,
        )
        pk.kn_graph.register_task(tb_graph, "mla_v4_q_kv_rmsnorm_sm100")
        return q_norm_dt, kv_norm_dt
