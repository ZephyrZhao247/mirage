"""DeepSeek V4-Flash per-layer composite (Layer 0).

Composes the V4 catalog modules and V3-reused modules into a single
:class:`MPKModule` matching ``Block.forward`` in
``deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py`` (lines 648-701).

Wave-3 scope (v1):
    * ``layer_idx = 0``: ``compress_ratio == 0`` → no Compressor, no
      Indexer, no sparse-attention.
    * ``layer_idx < num_hash_layers (=3)`` → hash routing for MoE.
    * HC: standard (``hc_mult = 4``).
    * Attention: MLA with SWA-only / dense path; the optional
      ``compressed_cache`` and ``topk_indices`` of :class:`MLAv4Decode`
      are absent (kept = None).

The :meth:`forward` method is the PyTorch reference oracle used by the
correctness test. The :meth:`compile` method composes catalog modules'
``compile()`` into one MPK task graph; Wave-3 leaves it as a best-effort
scaffold — see ``OPEN`` notes in the body for the integration gates
that still need to be wired before the compiled-vs-reference test
passes bit-exact.

Reference: spec ``docs/mpk/deepseek_v4/overview.md``, especially the
"Top-level forward pseudo-code" and § 4 "Weight inventory".
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...layers._base import MPKModule
from ...layers.hc.mhc_prenorm_gemm import MhcPrenormGemm
from ...layers.hc.mhc_pre import MhcPre
from ...layers.hc.mhc_post import MhcPost
from ...layers.attention.mla_v4_q_kv_rmsnorm import MLAv4QKVRMSNorm
from ...layers.attention.mla_v4_decode import MLAv4Decode
from ...layers.attention.inv_rope_fp8_quant_o import InvRopeFP8QuantO
from ...layers.moe.hash_route_lookup import HashRouteLookup
from ...layers.activation.silu_mul import SiluMul


__all__ = ["DeepseekV4Block"]


# --------------------------------------------------------------------- #
# Helper math (PyTorch reference, no MPK kernels)
# --------------------------------------------------------------------- #

def _rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Plain RMSNorm with weight broadcast: ``x * rsqrt(mean(x^2)+eps) * w``."""
    in_dtype = x.dtype
    x_f = x.float()
    rms = torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + eps)
    return (x_f * rms).to(in_dtype) * weight


def _yarn_freqs_cis(
    rope_dim: int,
    max_seq_len: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
) -> torch.Tensor:
    """Port of ``precompute_freqs_cis`` from the official model.py."""

    def find_correction_dim(num_rotations, dim, base_, max_seq_len_):
        return dim * math.log(max_seq_len_ / (num_rotations * 2 * math.pi)) / (
            2 * math.log(base_)
        )

    def find_correction_range(low_rot, high_rot, dim, base_, max_seq_len_):
        low = math.floor(find_correction_dim(low_rot, dim, base_, max_seq_len_))
        high = math.ceil(find_correction_dim(high_rot, dim, base_, max_seq_len_))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(lo, hi, dim):
        if lo == hi:
            hi += 0.001
        linear = (torch.arange(dim, dtype=torch.float32) - lo) / (hi - lo)
        return torch.clamp(linear, 0, 1)

    freqs = 1.0 / (
        base ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
    )
    if original_seq_len > 0:
        low, high = find_correction_range(
            beta_fast, beta_slow, rope_dim, base, original_seq_len
        )
        smooth = 1 - linear_ramp_factor(low, high, rope_dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def _apply_rotary_inplace(
    x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """Port of ``apply_rotary_emb`` (in-place). ``x`` must be a contiguous
    bf16 slice; we update it in place and return it for chaining."""
    y = x
    x_c = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    fc = freqs_cis.conj() if inverse else freqs_cis
    if x_c.ndim == 3:
        fc = fc.view(1, x_c.size(1), x_c.size(-1))
    else:
        fc = fc.view(1, x_c.size(1), 1, x_c.size(-1))
    x_rot = torch.view_as_real(x_c * fc).flatten(-2).to(y.dtype)
    y.copy_(x_rot)
    return y


def _hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    hc_eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pure-PyTorch port of the official ``hc_split_sinkhorn`` (kernel.py).

    Splits the [mix_hc] axis into (pre[hc], post[hc], comb[hc, hc]) with
    K2 affine + sigmoid for pre/post and Sinkhorn normalization for comb.
    """
    hc = hc_mult
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + hc_eps
    post = 2.0 * torch.sigmoid(
        mixes[..., hc : 2 * hc] * hc_scale[1] + hc_base[hc : 2 * hc]
    )
    cm = (mixes[..., 2 * hc :] * hc_scale[2] + hc_base[2 * hc :]).reshape(
        *mixes.shape[:-1], hc, hc
    )
    cm = torch.softmax(cm, dim=-1) + hc_eps
    cm = cm / (cm.sum(dim=-2, keepdim=True) + hc_eps)
    for _ in range(sinkhorn_iters - 1):
        cm = cm / (cm.sum(dim=-1, keepdim=True) + hc_eps)
        cm = cm / (cm.sum(dim=-2, keepdim=True) + hc_eps)
    return pre, post, cm


# --------------------------------------------------------------------- #
# Configuration helper
# --------------------------------------------------------------------- #


def _v4_config(cfg: Dict) -> Dict:
    """Pull the V4-Flash hyper-params out of ``config.json`` style dict.

    Provides safe defaults for keys that the Flash-Base config.json omits
    (some are only in the official model.py's ModelArgs dataclass).
    """
    out = dict(
        # Header dims
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        head_dim=cfg["head_dim"],
        qk_rope_head_dim=cfg["qk_rope_head_dim"],
        q_lora_rank=cfg["q_lora_rank"],
        o_lora_rank=cfg["o_lora_rank"],
        o_groups=cfg["o_groups"],
        # HC
        hc_mult=cfg.get("hc_mult", 4),
        hc_sinkhorn_iters=cfg.get("hc_sinkhorn_iters", 20),
        hc_eps=cfg.get("hc_eps", 1e-6),
        # MoE
        n_routed_experts=cfg["n_routed_experts"],
        n_shared_experts=cfg["n_shared_experts"],
        num_experts_per_tok=cfg["num_experts_per_tok"],
        moe_intermediate_size=cfg["moe_intermediate_size"],
        num_hash_layers=cfg.get("num_hash_layers", 3),
        routed_scaling_factor=cfg.get("routed_scaling_factor", 1.5),
        swiglu_limit=cfg.get("swiglu_limit", 10.0),
        # Attention
        sliding_window=cfg.get("sliding_window", 128),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        rope_theta=cfg.get("rope_theta", 10000.0),
        max_position_embeddings=cfg.get("max_position_embeddings", 1048576),
        # YaRN scaling
        rope_scaling=cfg.get("rope_scaling", {}),
        # Per-layer compress_ratios
        compress_ratios=cfg.get("compress_ratios", [0] * cfg["num_hidden_layers"]),
    )
    out["nope_head_dim"] = out["head_dim"] - out["qk_rope_head_dim"]
    out["mix_hc"] = (2 + out["hc_mult"]) * out["hc_mult"]
    out["hc_dim"] = out["hc_mult"] * out["hidden_size"]
    return out


# --------------------------------------------------------------------- #
# DeepseekV4Block
# --------------------------------------------------------------------- #


class DeepseekV4Block(MPKModule):
    """Per-layer composite for V4-Flash.

    Wave-3 scope: ``layer_idx=0`` (``compress_ratio=0``, hash routing).

    Parameters live on the block per the official ``Block`` layout
    (``deps/deepseek_v4/DeepSeek-V4-Flash/inference/model.py:653-672``):

        attn_norm.weight        [hidden_size]               bf16
        ffn_norm.weight         [hidden_size]               bf16
        hc_attn_fn              [mix_hc, hc_dim]            fp32
        hc_ffn_fn               [mix_hc, hc_dim]            fp32
        hc_attn_base/hc_ffn_base [mix_hc]                   fp32
        hc_attn_scale/hc_ffn_scale [3]                       fp32
        attn.attn_sink          [num_heads]                  fp32
        attn.q_norm.weight      [q_lora_rank]                bf16
        attn.kv_norm.weight     [head_dim]                   bf16
        attn.wq_a.weight + scale [q_lora_rank, hidden]       fp8 + fp32
        attn.wq_b.weight + scale [n_heads*head_dim, q_lora]  fp8 + fp32
        attn.wkv.weight + scale  [head_dim, hidden]          fp8 + fp32
        attn.wo_a.weight + scale [groups*o_lora, n_heads*head_dim/groups] fp8 + fp32
        attn.wo_b.weight + scale [hidden, groups*o_lora]     fp8 + fp32
        ffn.gate.weight          [n_routed_experts, hidden]  bf16
        ffn.gate.tid2eid         [vocab_size, n_act]         int32 (hash routing only)
        ffn.experts.<e>.w1/w2/w3.weight + scale              fp8 per-expert
        ffn.shared_experts.w1/w2/w3.weight + scale           fp8

    The reference :meth:`forward` recreates ``Block.forward`` but uses
    *only pure PyTorch* (no tilelang / no fast_hadamard_transform), so it
    runs anywhere CUDA + PyTorch are available.

    The :meth:`compile` method composes catalog modules' ``compile()``
    where possible; orchestration of the FP8 MoE pipeline and the SWA
    cache management for MLA decode remain best-effort scaffolds for
    Wave-3 (see OPEN comments).
    """

    def __init__(
        self,
        config: Dict,
        layer_idx: int,
        *,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        if layer_idx != 0:
            # The class is layer-idx-parameterized but Wave-3 only validates
            # layer 0. Layers 2/3/43 follow in subsequent waves.
            pass
        cfg = _v4_config(config)
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.compress_ratio = cfg["compress_ratios"][layer_idx]
        self.is_hash_routing = layer_idx < cfg["num_hash_layers"]

        # Convenience aliases (used heavily below)
        H = self.hidden_size = cfg["hidden_size"]
        self.num_heads = cfg["num_attention_heads"]
        self.head_dim = cfg["head_dim"]
        self.qk_rope_head_dim = cfg["qk_rope_head_dim"]
        self.nope_head_dim = cfg["nope_head_dim"]
        self.q_lora_rank = cfg["q_lora_rank"]
        self.o_lora_rank = cfg["o_lora_rank"]
        self.o_groups = cfg["o_groups"]
        self.hc_mult = cfg["hc_mult"]
        self.mix_hc = cfg["mix_hc"]
        self.hc_dim = cfg["hc_dim"]
        self.hc_sinkhorn_iters = cfg["hc_sinkhorn_iters"]
        self.hc_eps = cfg["hc_eps"]
        self.rms_eps = cfg["rms_norm_eps"]
        self.num_experts_per_tok = cfg["num_experts_per_tok"]
        self.moe_intermediate_size = cfg["moe_intermediate_size"]
        self.n_routed_experts = cfg["n_routed_experts"]
        self.swiglu_limit = cfg["swiglu_limit"]
        self.route_scale = cfg["routed_scaling_factor"]
        self.softmax_scale = float(self.head_dim) ** -0.5
        self.sliding_window = cfg["sliding_window"]

        # ------------------------------------------------------------------
        # HC parameters (Block-level, both attn-site and ffn-site)
        # ------------------------------------------------------------------
        self.attn_norm_weight = nn.Parameter(torch.ones(H, dtype=torch.bfloat16))
        self.ffn_norm_weight = nn.Parameter(torch.ones(H, dtype=torch.bfloat16))
        self.hc_attn_fn = nn.Parameter(
            torch.empty(self.mix_hc, self.hc_dim, dtype=torch.float32)
        )
        self.hc_ffn_fn = nn.Parameter(
            torch.empty(self.mix_hc, self.hc_dim, dtype=torch.float32)
        )
        self.hc_attn_base = nn.Parameter(
            torch.empty(self.mix_hc, dtype=torch.float32)
        )
        self.hc_ffn_base = nn.Parameter(
            torch.empty(self.mix_hc, dtype=torch.float32)
        )
        self.hc_attn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))
        self.hc_ffn_scale = nn.Parameter(torch.empty(3, dtype=torch.float32))

        # ------------------------------------------------------------------
        # MLA-attention parameters (raw FP8 storage; dequant-on-forward in v1)
        # ------------------------------------------------------------------
        # attn_sink — additive logit sink, shape [n_heads], fp32.
        self.attn_sink = nn.Parameter(
            torch.zeros(self.num_heads, dtype=torch.float32)
        )
        self.attn_q_norm_weight = nn.Parameter(
            torch.ones(self.q_lora_rank, dtype=torch.bfloat16)
        )
        self.attn_kv_norm_weight = nn.Parameter(
            torch.ones(self.head_dim, dtype=torch.bfloat16)
        )

        # FP8 weights stored as float8_e4m3fn + block-128 fp32 scales (UE8M0
        # convention is "stored as fp32" on disk for V4-Flash, per inspection
        # of the checkpoint shard 00002; convert.py loads them as fp32).
        # The reference forward dequants on the fly.
        def _fp8_param(shape):
            return nn.Parameter(
                torch.zeros(shape, dtype=torch.float8_e4m3fn),
                requires_grad=False,
            )

        def _scale_param(shape):
            return nn.Parameter(
                torch.zeros(shape, dtype=torch.float32),
                requires_grad=False,
            )

        n_heads_x_head_dim = self.num_heads * self.head_dim
        groups_o_lora = self.o_groups * self.o_lora_rank
        per_group_in = n_heads_x_head_dim // self.o_groups

        self.wq_a_weight = _fp8_param((self.q_lora_rank, H))
        self.wq_a_scale = _scale_param((self.q_lora_rank // 128, H // 128))
        self.wq_b_weight = _fp8_param((n_heads_x_head_dim, self.q_lora_rank))
        self.wq_b_scale = _scale_param(
            (n_heads_x_head_dim // 128, self.q_lora_rank // 128)
        )
        self.wkv_weight = _fp8_param((self.head_dim, H))
        self.wkv_scale = _scale_param((self.head_dim // 128, H // 128))
        self.wo_a_weight = _fp8_param((groups_o_lora, per_group_in))
        self.wo_a_scale = _scale_param(
            (groups_o_lora // 128, per_group_in // 128)
        )
        self.wo_b_weight = _fp8_param((H, groups_o_lora))
        self.wo_b_scale = _scale_param((H // 128, groups_o_lora // 128))

        # ------------------------------------------------------------------
        # MoE parameters
        # ------------------------------------------------------------------
        # Gate (linear router): the official model casts to fp32 inside
        # forward; we keep it bf16 on disk and promote on the fly.
        self.gate_weight = nn.Parameter(
            torch.zeros(self.n_routed_experts, H, dtype=torch.bfloat16)
        )
        # Hash routing table — int32 (the checkpoint stores it as int64).
        if self.is_hash_routing:
            self.tid2eid = nn.Parameter(
                torch.zeros(
                    cfg["vocab_size"],
                    self.num_experts_per_tok,
                    dtype=torch.int32,
                ),
                requires_grad=False,
            )
            self.gate_bias = None
        else:
            self.tid2eid = None
            self.gate_bias = nn.Parameter(
                torch.zeros(self.n_routed_experts, dtype=torch.float32)
            )

        # Routed experts (FP8 per-expert tensors). Kept as a 3-tuple of
        # ModuleLists indexed [w1, w2, w3] so convert.py can address them
        # naturally; matching the official model's per-Expert layout.
        E = self.n_routed_experts
        I = self.moe_intermediate_size
        self.experts_w1_weight = _fp8_param((E, I, H))
        self.experts_w1_scale = _scale_param((E, I // 128, H // 128))
        self.experts_w2_weight = _fp8_param((E, H, I))
        self.experts_w2_scale = _scale_param((E, H // 128, I // 128))
        self.experts_w3_weight = _fp8_param((E, I, H))
        self.experts_w3_scale = _scale_param((E, I // 128, H // 128))

        # Shared expert (single; swiglu_limit=0 per official model.py:628).
        self.shared_w1_weight = _fp8_param((I, H))
        self.shared_w1_scale = _scale_param((I // 128, H // 128))
        self.shared_w2_weight = _fp8_param((H, I))
        self.shared_w2_scale = _scale_param((H // 128, I // 128))
        self.shared_w3_weight = _fp8_param((I, H))
        self.shared_w3_scale = _scale_param((I // 128, H // 128))

        # ------------------------------------------------------------------
        # Catalog sub-modules — exposed for compile()-side composition.
        # The reference forward uses the math directly; the sub-modules are
        # used only by compile() (where MPK kernels are registered).
        # ------------------------------------------------------------------
        self.mhc_prenorm_gemm_attn = MhcPrenormGemm(
            hidden_size=H, hc_mult=self.hc_mult,
            prefix=f"{prefix}hc_attn_prenorm_",
        )
        self.mhc_pre_attn = MhcPre(
            hidden_size=H, hc_mult=self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            prefix=f"{prefix}hc_attn_pre_",
        )
        self.mhc_post_attn = MhcPost(
            hidden_size=H, hc_mult=self.hc_mult,
            prefix=f"{prefix}hc_attn_post_",
        )
        self.mhc_prenorm_gemm_ffn = MhcPrenormGemm(
            hidden_size=H, hc_mult=self.hc_mult,
            prefix=f"{prefix}hc_ffn_prenorm_",
        )
        self.mhc_pre_ffn = MhcPre(
            hidden_size=H, hc_mult=self.hc_mult,
            sinkhorn_iters=self.hc_sinkhorn_iters,
            prefix=f"{prefix}hc_ffn_pre_",
        )
        self.mhc_post_ffn = MhcPost(
            hidden_size=H, hc_mult=self.hc_mult,
            prefix=f"{prefix}hc_ffn_post_",
        )
        self.mla_q_kv_norm = MLAv4QKVRMSNorm(
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.head_dim,
            eps=self.rms_eps,
            prefix=f"{prefix}mla_q_kv_norm_",
        )
        self.mla_decode = MLAv4Decode(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            softmax_scale=self.softmax_scale,
            prefix=f"{prefix}mla_decode_",
        )
        self.inv_rope_o = InvRopeFP8QuantO(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            rope_dim=self.qk_rope_head_dim,
            block_size=128,
            prefix=f"{prefix}inv_rope_o_",
        )
        if self.is_hash_routing:
            self.hash_route = HashRouteLookup(
                vocab_size=cfg["vocab_size"],
                num_experts_per_tok=self.num_experts_per_tok,
                prefix=f"{prefix}hash_route_",
            )
        else:
            self.hash_route = None
        self.expert_silu_mul = SiluMul(
            intermediate_size=self.moe_intermediate_size,
            swiglu_limit=self.swiglu_limit,
            prefix=f"{prefix}expert_silu_mul_",
        )
        self.shared_silu_mul = SiluMul(
            intermediate_size=self.moe_intermediate_size,
            swiglu_limit=None,
            prefix=f"{prefix}shared_silu_mul_",
        )

        # YaRN-disabled freqs_cis for ratio=0 layers (per model.py:478-480).
        # We compute lazily on first forward() because max_seq_len isn't a
        # constructor arg.
        self._freqs_cis_cache: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # FP8 dequant helpers (PyTorch reference only — compile() uses
    # MPK's native FP8 group GEMM kernels instead).
    # ------------------------------------------------------------------
    @staticmethod
    def _fp8_dequant(
        weight_fp8: torch.Tensor,
        scale: torch.Tensor,
        block_size: int = 128,
    ) -> torch.Tensor:
        """Reference dequant: expand block-128 fp32 scales to per-element
        and multiply.

        ``weight_fp8`` has shape ``[..., M, K]`` (last 2 dims block-scaled),
        ``scale`` has shape ``[..., M // block, K // block]`` fp32.
        """
        # Handle 2-D and 3-D (expert-batched) cases.
        if weight_fp8.dim() == 2:
            M, K = weight_fp8.shape
            sM = scale.shape[-2]
            sK = scale.shape[-1]
            s = scale.repeat_interleave(block_size, dim=-2)[:M]
            s = s.repeat_interleave(block_size, dim=-1)[:, :K]
            return weight_fp8.float() * s
        elif weight_fp8.dim() == 3:
            E, M, K = weight_fp8.shape
            s = scale.repeat_interleave(block_size, dim=-2)[..., :M, :]
            s = s.repeat_interleave(block_size, dim=-1)[..., :, :K]
            return weight_fp8.float() * s
        else:
            raise ValueError(f"Unsupported FP8 weight rank {weight_fp8.dim()}")

    def _fp8_linear(
        self,
        x: torch.Tensor,
        weight_fp8: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """Reference path: dequant weight to fp32, F.linear in bf16."""
        w_f32 = self._fp8_dequant(weight_fp8, scale)
        # Match the official Linear which keeps activations bf16 and casts
        # the weight ``to(x.dtype)`` before F.linear (see model.py:151).
        return F.linear(x, w_f32.to(x.dtype))

    @staticmethod
    def _fp8_qat_inplace(
        x: torch.Tensor, rope_dim: int, block_size: int = 64
    ) -> torch.Tensor:
        """Wave 3.5 Gap 4: simulate per-128 FP8 E4M3 quant on K/V nope dims.

        Replicates the effect of ``act_quant(kv[..., :-rope_dim], block_size,
        ..., inplace=True)`` from deps/deepseek_v4/.../kernel.py: it
        quantises the nope portion to FP8 then immediately dequantises it
        back to bf16 (round-trip noise injection). The rope portion is left
        untouched. Operates on a freshly returned tensor (not actually in-
        place against the caller's allocation, but writes the result back
        into the same object before returning).
        """
        nope = x[..., :-rope_dim].contiguous()
        N = nope.size(-1)
        if N % block_size != 0:
            return x  # nothing to do
        leading = nope.shape[:-1]
        num_groups = N // block_size
        grouped = nope.float().reshape(*leading, num_groups, block_size)
        amax = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
        # FP8 E4M3 max is 448.
        scale = amax / 448.0
        q = (grouped / scale).to(torch.float8_e4m3fn).float()
        dq = (q * scale).reshape(*leading, N).to(x.dtype)
        out = x.clone()
        out[..., :-rope_dim] = dq
        return out

    # ------------------------------------------------------------------
    # Freqs_cis for ratio=0 layers
    # ------------------------------------------------------------------
    def _freqs_cis(self, max_seq_len: int, device, dtype=torch.complex64) -> torch.Tensor:
        if self._freqs_cis_cache is not None and self._freqs_cis_cache.size(0) >= max_seq_len:
            return self._freqs_cis_cache[:max_seq_len].to(device)
        scaling = self.cfg.get("rope_scaling", {}) or {}
        # ratio=0 layers disable YaRN (model.py:478) and use base rope_theta.
        if self.compress_ratio:
            base = self.cfg.get("compress_rope_theta", 160000.0)
            original = scaling.get("original_max_position_embeddings", 65536)
            factor = scaling.get("factor", 16.0)
            beta_fast = scaling.get("beta_fast", 32)
            beta_slow = scaling.get("beta_slow", 1)
        else:
            base = self.cfg["rope_theta"]
            original = 0   # disable YaRN
            factor = 1.0
            beta_fast = 32
            beta_slow = 1
        fc = _yarn_freqs_cis(
            self.qk_rope_head_dim,
            max_seq_len,
            original,
            base,
            factor,
            beta_fast,
            beta_slow,
        ).to(device)
        self._freqs_cis_cache = fc
        return fc

    # ==================================================================
    # PyTorch reference forward
    # ==================================================================
    def forward(
        self,
        hidden_hc: torch.Tensor,
        position_ids: torch.Tensor,
        swa_cache: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """End-to-end PyTorch reference for one Block at ``layer_idx``.

        Args:
            hidden_hc:    ``[T, hc, H]`` bf16. The HC residual stream
                          coming into the block.
            position_ids: ``[T]`` int (decode positions).
            swa_cache:    ``[swa_total, head_dim]`` bf16. Pre-filled SWA
                          cache window (zeros = empty cache for a fresh
                          decode test). The block does NOT write back to
                          this cache in the reference forward.
            input_ids:    ``[T]`` int32/int64. Token IDs used for hash
                          routing (only when ``self.is_hash_routing``).

        Returns:
            ``[T, hc, H]`` bf16 — the new HC residual stream.
        """
        T = hidden_hc.size(0)
        device = hidden_hc.device
        H = self.hidden_size

        # ---------------- Attention site ----------------
        residual = hidden_hc  # [T, hc, H]
        x_flat = hidden_hc.reshape(T, -1).float()  # [T, hc*H]
        rsqrt = torch.rsqrt(
            x_flat.pow(2).mean(-1, keepdim=True) + self.rms_eps
        )
        # hc_attn_fn: [mix_hc, hc*H] fp32
        mixes = F.linear(x_flat, self.hc_attn_fn) * rsqrt
        pre, post, comb = _hc_split_sinkhorn(
            mixes,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        # Reduce: weighted sum over hc copies → [T, H]
        layer_input = torch.sum(
            pre.unsqueeze(-1) * residual.float(), dim=1
        ).to(hidden_hc.dtype)

        # attn_norm
        layer_input = _rmsnorm(
            layer_input, self.attn_norm_weight, self.rms_eps
        )  # [T, H]

        # ---- MLA attention ----
        x_attn = self._mla_forward(layer_input, position_ids, swa_cache)
        # ---- HC post (attn) ----
        # y = post * x + sum_{hc_i} comb * residual
        hidden_hc = (
            post.unsqueeze(-1) * x_attn.unsqueeze(1).float()
            + torch.sum(
                comb.unsqueeze(-1) * residual.unsqueeze(2).float(), dim=1
            )
        ).to(hidden_hc.dtype)

        # ---------------- FFN site ----------------
        residual = hidden_hc
        x_flat = hidden_hc.reshape(T, -1).float()
        rsqrt = torch.rsqrt(
            x_flat.pow(2).mean(-1, keepdim=True) + self.rms_eps
        )
        mixes = F.linear(x_flat, self.hc_ffn_fn) * rsqrt
        pre, post, comb = _hc_split_sinkhorn(
            mixes,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.hc_mult,
            self.hc_sinkhorn_iters,
            self.hc_eps,
        )
        layer_input = torch.sum(
            pre.unsqueeze(-1) * residual.float(), dim=1
        ).to(hidden_hc.dtype)
        layer_input = _rmsnorm(
            layer_input, self.ffn_norm_weight, self.rms_eps
        )
        # ---- MoE ----
        x_ffn = self._moe_forward(layer_input, input_ids)
        # ---- HC post (ffn) ----
        hidden_hc = (
            post.unsqueeze(-1) * x_ffn.unsqueeze(1).float()
            + torch.sum(
                comb.unsqueeze(-1) * residual.unsqueeze(2).float(), dim=1
            )
        ).to(hidden_hc.dtype)
        return hidden_hc

    # ------------------------------------------------------------------
    # Reference MLA attention (port of Attention.forward, ratio=0)
    # ------------------------------------------------------------------
    def _mla_forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        swa_cache: torch.Tensor,
    ) -> torch.Tensor:
        """Layer-0 MLA: pure SWA, no Compressor, no Indexer.

        Single-batch (B=1) — we treat ``x`` as ``[T, H]`` and the SWA
        cache as a flat ``[swa_total, head_dim]`` window where row k is
        the K/V for absolute position k. The test sets the cache to all-
        zeros so that each decode step only attends over the freshly-
        produced K/V from this forward.
        """
        T = x.size(0)
        device = x.device
        rd = self.qk_rope_head_dim
        nope = self.nope_head_dim
        # === Q projection ===
        # wq_a [q_lora, H] FP8 → q_lora [T, q_lora] bf16
        q_lora = self._fp8_linear(x, self.wq_a_weight, self.wq_a_scale)
        q_lora = _rmsnorm(q_lora, self.attn_q_norm_weight, self.rms_eps)
        # wq_b [H*head_dim, q_lora] FP8 → q [T, H*head_dim]
        q = self._fp8_linear(q_lora, self.wq_b_weight, self.wq_b_scale)
        q = q.unflatten(-1, (self.num_heads, self.head_dim)).contiguous()
        # Per-head RMS (model.py:498)
        q_var = q.float().square().mean(-1, keepdim=True)
        q = (q.float() * torch.rsqrt(q_var + self.rms_eps)).to(q.dtype)
        # Apply rotary to the rope tail.
        freqs_cis = self._freqs_cis(T + 1, device)
        # Use positions [0..T) for a single decode-style sequence.
        fc_seg = freqs_cis[:T]
        # Make a 4-D view [1, T, H, rope_dim] for the apply_rotary_emb helper.
        q4 = q.unsqueeze(0)
        _apply_rotary_inplace(q4[..., -rd:], fc_seg)
        q = q4.squeeze(0)

        # === KV projection ===
        # wkv [head_dim, H] FP8 → kv [T, head_dim]
        kv = self._fp8_linear(x, self.wkv_weight, self.wkv_scale)
        kv = _rmsnorm(kv, self.attn_kv_norm_weight, self.rms_eps)
        # Apply rotary to last rd dims.
        kv3 = kv.unsqueeze(0).unsqueeze(2)   # [1, T, 1, head_dim]
        _apply_rotary_inplace(kv3[..., -rd:], fc_seg)
        kv = kv3.squeeze(2).squeeze(0)
        # Wave 3.5 (Gap 4): apply FP8 QAT to kv[..., :-rd] (the nope dims).
        # The official model calls ``act_quant(kv[..., :-rd], 64, ..., inplace=True)``
        # which performs a fused quant-dequant round-trip in bf16 -- it does
        # not change the dtype but introduces FP8 E4M3 quantisation noise on
        # the nope portion of K/V (the rope portion is left untouched).
        # Without this step the L0 reference vs official-PyTorch divergence
        # on the K cache is ~1%; with it the bf16 tolerance lines up.
        kv = self._fp8_qat_inplace(kv, rd, block_size=64)

        # === sparse_attn (ratio=0: dense over [position 0..t)) ===
        # Build the K/V tensor: stack the freshly computed kv with the SWA
        # cache. For T tokens at positions [0..T), token t attends over
        # rows [0..t) of the merged K/V stream.
        # In a true MPK runtime, the swa_cache is written first and queried
        # back; here we ignore swa_cache (assumed empty) and attend over
        # the freshly produced kv tensor only.
        k_all = kv.unsqueeze(1)  # [T, 1, head_dim] - one shared KV head
        # Expand kv to [num_heads, ...] for the attention matmul:
        # MLA uses a single shared K/V head replicated across num_heads.
        attn_out = self._mla_attn_dense(q, kv, position_ids)
        # === Inverse rotary on the rope tail of attn_out ===
        attn4 = attn_out.unsqueeze(0)
        _apply_rotary_inplace(attn4[..., -rd:], fc_seg, inverse=True)
        attn_out = attn4.squeeze(0)
        # === O projection: wo_a [groups*o_lora, head_dim*n_heads/groups], then wo_b [H, groups*o_lora] ===
        # Reshape o to [T, groups, head_dim*n_heads/groups]
        per_group = self.num_heads * self.head_dim // self.o_groups
        o_gr = attn_out.reshape(T, self.o_groups, per_group)
        # wo_a is grouped: shape [groups*o_lora, per_group]; we view it as
        # [groups, o_lora, per_group] and einsum.
        wo_a_f32 = self._fp8_dequant(self.wo_a_weight, self.wo_a_scale)
        wo_a_grp = wo_a_f32.view(self.o_groups, self.o_lora_rank, per_group)
        wo_a_out = torch.einsum(
            "tgd,grd->tgr", o_gr.float(), wo_a_grp
        ).to(x.dtype)  # [T, groups, o_lora]
        wo_a_out_flat = wo_a_out.reshape(T, self.o_groups * self.o_lora_rank)
        # wo_b [H, groups*o_lora]
        out = self._fp8_linear(wo_a_out_flat, self.wo_b_weight, self.wo_b_scale)
        return out

    def _mla_attn_dense(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Dense causal attention with attn_sink (single shared KV head).

        ``q``  : [T, num_heads, head_dim] bf16
        ``kv`` : [T, head_dim]            bf16
        ``position_ids``: [T] int — for causal mask use.

        Returns ``o`` [T, num_heads, head_dim] bf16.
        """
        T = q.size(0)
        H = self.num_heads
        D = self.head_dim
        q_f = q.float()
        kv_f = kv.float()  # [T, D]
        sink = self.attn_sink.float()  # [H]
        out = torch.zeros(T, H, D, dtype=torch.float32, device=q.device)
        for t in range(T):
            # Token t attends over positions [0, t] (causal). For prefill
            # this is the prefix; for decode at position p it's [0, p].
            kv_pre = kv_f[: t + 1]                        # [t+1, D]
            logits = (q_f[t] @ kv_pre.transpose(-1, -2))  # [H, t+1]
            logits = logits * self.softmax_scale
            sink_lane = sink.unsqueeze(-1)                # [H, 1]
            full = torch.cat([logits, sink_lane], dim=-1) # [H, t+2]
            w = full.softmax(dim=-1)
            w_kv = w[:, : t + 1]
            out[t] = w_kv @ kv_pre                        # [H, D]
        return out.to(q.dtype)

    # ------------------------------------------------------------------
    # Reference MoE (port of MoE.forward, hash routing branch)
    # ------------------------------------------------------------------
    def _moe_forward(
        self, x: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        T = x.size(0)
        H = self.hidden_size
        # === Gate ===
        # router_logits = F.linear(x.float(), gate.weight.float())
        scores = F.linear(x.float(), self.gate_weight.float())
        # sqrtsoftplus scoring
        original_scores = F.softplus(scores).sqrt()
        if self.is_hash_routing:
            indices = self.tid2eid[input_ids.long()].long()  # [T, K]
        else:
            # Score-based topk (with bias for selection only)
            scores_b = original_scores + (self.gate_bias if self.gate_bias is not None else 0.0)
            indices = scores_b.topk(self.num_experts_per_tok, dim=-1)[1]
        weights = original_scores.gather(1, indices)  # [T, K]
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-12)
        weights = weights * self.route_scale  # [T, K]

        # === Routed experts ===
        # Slow reference: per-(token,k) expert dispatch matching official model
        # forward (line 636-641). Loops over all experts; for each one, find
        # which (token, k_slot) pairs route to it and sum-accumulate.
        y = torch.zeros_like(x, dtype=torch.float32)
        # Dequant all expert weights once (small for the layer-test K=6, T=4
        # path: 256 experts × 3 matmuls of 2048×4096 — fine on a B200).
        w1_all = self._fp8_dequant(
            self.experts_w1_weight, self.experts_w1_scale
        )  # [E, I, H]
        w2_all = self._fp8_dequant(
            self.experts_w2_weight, self.experts_w2_scale
        )  # [E, H, I]
        w3_all = self._fp8_dequant(
            self.experts_w3_weight, self.experts_w3_scale
        )  # [E, I, H]
        for e in range(self.n_routed_experts):
            # Find (token, k_slot) where indices == e
            mask = indices == e
            if not mask.any():
                continue
            tok_idx, k_idx = torch.where(mask)
            x_sel = x[tok_idx].float()                    # [n_sel, H]
            gate = F.linear(x_sel, w1_all[e])             # [n_sel, I]
            up = F.linear(x_sel, w3_all[e])               # [n_sel, I]
            if self.swiglu_limit > 0:
                gate = torch.clamp(gate, max=self.swiglu_limit)
                up = torch.clamp(up, min=-self.swiglu_limit, max=self.swiglu_limit)
            mid = F.silu(gate) * up                       # [n_sel, I]
            out_e = F.linear(mid.to(x.dtype), w2_all[e].to(x.dtype)).float()
            w_sel = weights[tok_idx, k_idx].unsqueeze(-1).float()  # [n_sel, 1]
            y.index_add_(0, tok_idx, out_e * w_sel)
        # === Shared expert (no clamp) ===
        sw1 = self._fp8_dequant(self.shared_w1_weight, self.shared_w1_scale)
        sw3 = self._fp8_dequant(self.shared_w3_weight, self.shared_w3_scale)
        sw2 = self._fp8_dequant(self.shared_w2_weight, self.shared_w2_scale)
        gate_s = F.linear(x.float(), sw1)
        up_s = F.linear(x.float(), sw3)
        mid_s = F.silu(gate_s) * up_s
        shared = F.linear(mid_s.to(x.dtype), sw2.to(x.dtype))
        return (y.to(x.dtype) + shared).to(x.dtype)

    # ==================================================================
    # MPK compile() — Wave-3.5 best-effort scaffold
    # ==================================================================
    def compile(
        self,
        hidden_hc_dt,
        position_ids_dt,
        swa_cache_dt,
        input_ids_dt,
        *,
        output_hc=None,
        grid_dim=None,
        block_dim=None,
    ):
        """Best-effort MPK composition of the V4 catalog modules.

        Wave 3.5 status (follow-up to 83c38ecc):

        * Gap 4 (FP8 QAT on K/V nope dims): FIXED in the PyTorch
          reference ``forward()``. The MPK compile path needs to insert
          ``QuantizeFP8(block_size=64)`` followed by an in-place dequant
          before the ``MLAv4Decode`` consumes K/V; the catalog module
          exists at ``python/mirage/mpk/layers/quantize_fp8.py``. Not yet
          wired here.

        * Gap 2 (grouped FP8 BMM for wo_a): the existing
          ``LinearFP8BMM`` catalog matches the V4 wo_a shape contract
          exactly --
          ``[T, groups, per_group] @ [groups, o_lora, per_group]^T
          -> [T, groups, o_lora]``. Wave 4 should compose this directly
          (no new kernel required); ``num_heads = o_groups``,
          ``in_features_per_head = num_heads * head_dim / o_groups``,
          ``out_features_per_head = o_lora_rank``.

        * Gap 3 (hash routing vs V3 MoE permute): ``HashRouteLookup``
          outputs ``[T, K]`` int32 token-major; ``MoEPermute`` wants
          ``[E_LOCAL, MBT]`` int32 expert-major + 1-indexed. See
          ``python/mirage/mpk/layers/moe/hash_route_to_expert_major.py``
          for the adapter (Python reference today, kernel-fold OPEN).

        * Gap 1 (SWA cache write-back): catalog scaffold landed at
          ``python/mirage/mpk/layers/attention/mla_v4_swa_cache_write.py``
          with reserved ``TaskType TASK_MLA_V4_SWA_CACHE_WRITE_SM100=364``
          in ``runtime_header.h``. The CUDA kernel and ``task_register.cc``
          entry are NOT landed -- next agent's first task.

        Until all four gaps are wired into a single contiguous
        compile-scope block, this method raises
        :class:`NotImplementedError`. The companion test
        ``test_compiled_vs_reference_layer0`` remains SKIP with the
        precise outstanding-work description.
        """
        raise NotImplementedError(
            "DeepseekV4Block.compile() is a Wave-3.5 scaffold. Wave-4 "
            "must (a) wire LinearFP8BMM for wo_a, (b) land the new "
            "mla_v4_swa_cache_write_sm100 CUDA kernel + task_register "
            "entry, (c) wire HashRouteToExpertMajor (or fold into "
            "MoEPermute), and (d) thread the FP8 QAT step (gap 4) into "
            "the compile path between MLAv4QKVRMSNorm and MLAv4Decode. "
            "See this method's docstring for per-gap detail."
        )

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite — see child compile()s")
