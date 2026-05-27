"""DeepSeek V4-Flash Indexer score + Top-K catalog module.

Backed by ``tasks/blackwell/indexer_score_topk_sm100.cuh`` (task name
``"indexer_score_topk_sm100"``).

This is the final step of the Indexer pipeline for ``compress_ratio == 4``
layers. Given the dequantised per-head Q produced by
``indexer_q_transform_layer`` and the per-head ``weights_proj`` output,
the kernel:

1. Computes ``score[t, h, s] = q[t, h, :] @ kv_cache[s, :]``.
2. Applies ReLU (sign matters; the official's
   ``Indexer.forward`` `:418-432` applies ``relu_()`` before the per-head
   weight folding).
3. Folds in the per-head weights and reduces over the head axis to obtain
   ``score_summed[t, s] = sum_h relu(score[t, h, s]) * weights_proj[t, h]``.
4. Applies the causal mask ``s > positions[t] // compress_ratio -> -inf``
   (sentinel ``-1`` in the output).
5. Selects the top-``topk`` candidates per token.

Output ``topk_indices [T, topk]`` int32 feeds the sparse decode of
``mla_v4_decode`` (v2 follow-up).

V1 simplifications (per the sub-batch C2 task description):
  * ``q`` and ``kv_cache`` are bf16. FP4-native q is reserved for v2; the
    upstream ``indexer_q_transform`` is expected to dequantise to bf16
    before invoking this kernel.
  * ``kv_cache`` is contiguous ``[S_max, INDEX_HEAD_DIM]`` — i.e., the
    Python caller is responsible for staging a 2-D slab for the test
    harness. Paged-KV plumbing (``kv_cache_indexer`` +
    ``block_table_indexer``) is reserved for v2.
  * ``weights_proj`` is fp32 and already contains the ``softmax_scale *
    n_heads ** -0.5`` factor (matching the official's pre-multiplied
    ``self.weights_proj(x) * (softmax_scale * n_heads ** -0.5)`` on
    `model.py:418`).
  * Single-batch only (``B=1``). The compile path validates ``q.dim(0) ==
    weights_proj.dim(0) == positions.dim(0)``.

Reference call sites:
  * vLLM:    ``deps/vllm/vllm/model_executor/layers/sparse_attn_indexer.py``
             (``paged_mqa_logits`` + ``persistent_topk``).
  * Official: ``deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py``
             ``Indexer.forward`` lines 418-432.
  * Spec:    ``docs/mpk/deepseek_v4/sparse.md`` § F.3
             ``indexer_score_topk_layer``.
"""
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn

import mirage as mi

from .._base import BlockDim, GridDim, MPKModule

from ....core import DTensor


__all__ = ["IndexerScoreTopK"]


class IndexerScoreTopK(MPKModule):
    """V4-Flash Indexer score + Top-K (v1: bf16, contiguous KV cache).

    Constructor args:
      index_n_heads   : number of indexer heads (64 in V4-Flash).
      index_head_dim  : per-head dim of the indexer cache (128).
      topk            : top-K count (512 in V4-Flash; ``index_topk``).
      compress_ratio  : causal-mask divisor — for token at position
                         ``pos`` only candidate compressed positions
                         ``s <= pos / compress_ratio`` are valid. Default
                         ``4`` (the V4-Flash indexer cache stride).
      prefix          : ``state_dict`` key prefix.

    State: this module owns no ``nn.Parameter`` — all inputs are
    activations produced by upstream layers.
    """

    def __init__(
        self,
        index_n_heads: int,
        index_head_dim: int,
        topk: int = 512,
        compress_ratio: int = 4,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if index_n_heads <= 0:
            raise ValueError(
                f"index_n_heads must be > 0; got {index_n_heads}"
            )
        if index_head_dim <= 0:
            raise ValueError(
                f"index_head_dim must be > 0; got {index_head_dim}"
            )
        if topk <= 0:
            raise ValueError(f"topk must be > 0; got {topk}")
        if compress_ratio <= 0:
            raise ValueError(
                f"compress_ratio must be > 0; got {compress_ratio}"
            )
        self.index_n_heads = index_n_heads
        self.index_head_dim = index_head_dim
        self.topk = topk
        self.compress_ratio = compress_ratio

    # ------------------------------------------------------------------
    # PyTorch reference
    # ------------------------------------------------------------------
    def forward(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        weights_proj: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """V1 reference: bf16 q + kv_cache, fp32 weights_proj.

        Args:
          q            : ``[T, index_n_heads, index_head_dim]`` bf16.
                          Already dequantised (the FP8/FP4-native path is
                          a v2 follow-up).
          kv_cache     : ``[S_max, index_head_dim]`` bf16 (contiguous).
          weights_proj : ``[T, index_n_heads]`` fp32 — already multiplied
                          by ``softmax_scale * n_heads ** -0.5``.
          positions    : ``[T]`` int32 absolute positions.

        Returns ``topk_indices`` ``[T, topk]`` int32. Slots beyond the
        causal mask or beyond the number of valid candidates are filled
        with sentinel ``-1``.
        """
        if q.dim() != 3 or q.size(1) != self.index_n_heads or q.size(
            2
        ) != self.index_head_dim:
            raise ValueError(
                "q must be (T, index_n_heads, index_head_dim); got "
                f"{tuple(q.shape)}, expected (*, "
                f"{self.index_n_heads}, {self.index_head_dim})"
            )
        if kv_cache.dim() != 2 or kv_cache.size(-1) != self.index_head_dim:
            raise ValueError(
                "kv_cache must be (S_max, index_head_dim); got "
                f"{tuple(kv_cache.shape)}"
            )
        if (
            weights_proj.dim() != 2
            or weights_proj.size(0) != q.size(0)
            or weights_proj.size(1) != self.index_n_heads
        ):
            raise ValueError(
                "weights_proj must be (T, index_n_heads); got "
                f"{tuple(weights_proj.shape)} vs q {tuple(q.shape)}"
            )
        if positions.dim() != 1 or positions.size(0) != q.size(0):
            raise ValueError(
                "positions must be 1-D with the same leading dim as q; "
                f"got {tuple(positions.shape)} vs q {tuple(q.shape)}"
            )

        T = q.size(0)
        S_max = kv_cache.size(0)
        K = self.topk
        device = q.device

        q_f = q.to(torch.float32)
        kv_f = kv_cache.to(torch.float32)
        w_f = weights_proj.to(torch.float32)

        # score[t, h, s] = q[t, h, :] @ kv_cache[s, :]
        # -> [T, H, S]
        score = torch.einsum("thd,sd->ths", q_f, kv_f)
        # relu then per-head weight, sum over heads.
        score = torch.relu(score) * w_f.unsqueeze(-1)
        score_summed = score.sum(dim=1)  # [T, S]

        # Causal mask: s > positions[t] // compress_ratio -> -inf
        s_idx = torch.arange(S_max, device=device).unsqueeze(0)        # [1, S]
        cutoff = (
            positions.to(torch.int64) // self.compress_ratio
        ).unsqueeze(1)                                                  # [T, 1]
        mask = s_idx > cutoff                                           # [T, S]
        score_summed = torch.where(
            mask,
            torch.full_like(score_summed, float("-inf")),
            score_summed,
        )

        # Top-K. If there are fewer than K valid candidates, the trailing
        # entries are -inf — we replace those indices with -1 sentinel.
        K_eff = min(K, S_max)
        topk_vals, topk_idx = torch.topk(score_summed, K_eff, dim=-1)
        # Replace -inf slots (invalid candidates) with -1 sentinel.
        invalid = torch.isinf(topk_vals) & (topk_vals < 0)
        topk_idx = torch.where(
            invalid, torch.full_like(topk_idx, -1), topk_idx
        )
        topk_idx = topk_idx.to(torch.int32)
        if K_eff < K:
            pad = torch.full(
                (T, K - K_eff), -1, dtype=torch.int32, device=device
            )
            topk_idx = torch.cat([topk_idx, pad], dim=-1)
        return topk_idx

    # ------------------------------------------------------------------
    # Grid / block heuristics
    # ------------------------------------------------------------------
    def auto_grid_dim(self, q_dt: DTensor) -> GridDim:
        """One CTA per token; CTAs derive ``t`` from ``token_offset``."""
        from ... import context as _ctx
        pk = _ctx.current_pk()
        n = q_dt.dim(0)
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
        q: DTensor,
        kv_cache: Union[torch.Tensor, DTensor],
        weights_proj: Union[torch.Tensor, DTensor],
        positions: Union[torch.Tensor, DTensor],
        *,
        topk_indices: Optional[Union[torch.Tensor, DTensor]] = None,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register one ``indexer_score_topk_sm100`` task.

        Tensor contract:
          q            : ``[T, index_n_heads, index_head_dim]`` bf16, input.
          kv_cache     : ``[S_max, index_head_dim]``           bf16, input.
          weights_proj : ``[T, index_n_heads]``                fp32, input.
          positions    : ``[T]``                                int32, input.
          topk_indices : ``[T, topk]``                          int32, output
                          (auto-allocated if ``None``).
        """
        from ... import context as _ctx
        from ....core import CyTBGraph
        from ....kernel import TBGraph

        pk = _ctx.current_pk()
        prefix = self.prefix or "indexer_score_topk_"

        # Validate q.
        if q.num_dims != 3:
            raise ValueError(
                "q must be a 3-D DTensor (T, index_n_heads, "
                f"index_head_dim); got num_dims={q.num_dims}"
            )
        if q.dim(1) != self.index_n_heads:
            raise ValueError(
                f"q.dim(1)={q.dim(1)} != "
                f"index_n_heads={self.index_n_heads}"
            )
        if q.dim(2) != self.index_head_dim:
            raise ValueError(
                f"q.dim(2)={q.dim(2)} != "
                f"index_head_dim={self.index_head_dim}"
            )
        T = q.dim(0)

        def _attach_in(buf, expected_dtype_torch, name):
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

        kv_cache_dt = _attach_in(
            kv_cache, torch.bfloat16, f"{prefix}kv_cache"
        )
        if kv_cache_dt.num_dims != 2:
            raise ValueError(
                "kv_cache must be a 2-D DTensor (S_max, "
                f"index_head_dim); got num_dims={kv_cache_dt.num_dims}"
            )
        if kv_cache_dt.dim(1) != self.index_head_dim:
            raise ValueError(
                f"kv_cache.dim(1)={kv_cache_dt.dim(1)} != "
                f"index_head_dim={self.index_head_dim}"
            )

        weights_dt = _attach_in(
            weights_proj, torch.float32, f"{prefix}weights_proj"
        )
        if (
            weights_dt.num_dims != 2
            or weights_dt.dim(0) != T
            or weights_dt.dim(1) != self.index_n_heads
        ):
            raise ValueError(
                "weights_proj must be (T, index_n_heads); got "
                f"num_dims={weights_dt.num_dims}, "
                f"dim(0)={weights_dt.dim(0)}, "
                f"dim(1)={weights_dt.dim(1)}"
            )

        positions_dt = _attach_in(
            positions, torch.int32, f"{prefix}positions"
        )
        if positions_dt.num_dims != 1 or positions_dt.dim(0) != T:
            raise ValueError(
                "positions must be 1-D with length T; got "
                f"num_dims={positions_dt.num_dims}, len="
                f"{positions_dt.dim(0)} vs T={T}"
            )

        # Output.
        def _attach_out(buf, default_dims, name):
            if buf is None:
                return pk.new_tensor(
                    dims=default_dims, dtype=mi.int32, name=name
                )
            if isinstance(buf, torch.Tensor):
                if buf.dtype != torch.int32:
                    raise ValueError(
                        f"{name} must have dtype int32; got {buf.dtype}"
                    )
                return pk.attach_input(buf, name=name)
            if isinstance(buf, DTensor):
                return buf
            raise TypeError(
                f"{name} must be None, torch.Tensor, or DTensor; got "
                f"{type(buf).__name__}"
            )

        topk_dt = _attach_out(
            topk_indices, (T, self.topk), f"{prefix}topk_indices"
        )

        params = [int(self.topk), int(self.compress_ratio)]

        if grid_dim is None:
            grid_dim = self.auto_grid_dim(q)
        if block_dim is None:
            block_dim = self.default_block_dim()

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q, (-1, -1, -1), -1, True)
        tb_graph.new_input(kv_cache_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(weights_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(positions_dt, (-1, -1, -1), -1, True)
        tb_graph.new_input(topk_dt, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [q, kv_cache_dt, weights_dt, positions_dt, topk_dt],
            tb_graph,
        )
        pk.kn_graph.register_task(
            tb_graph, "indexer_score_topk_sm100", params
        )
        return topk_dt
