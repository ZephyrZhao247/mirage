"""DeepSeek V3 model defined against the new ``mirage.mpk.layers`` catalog.

This is the v1 catalog-based implementation of DeepSeek V3 (companion to the
existing :mod:`mirage.mpk.models.deepseek_v3.builder` which uses the
direct-pk path). It mirrors the structure of
:mod:`mirage.mpk.models.qwen3.modeling`:

  * One :class:`MPKModule` subclass per architectural block (MLA, dense MLP,
    MoE MLP, decoder layer, the model, and the LM head wrapper).
  * Each block implements ``compile()`` which registers MPK tasks via the
    catalog modules from :mod:`mirage.mpk.layers`. The PyTorch ``forward()``
    paths are stubbed (``NotImplementedError``) because the MLA / paged-KV /
    MoE-routing references depend on MPK runtime state that has no eager
    counterpart; the official HF reference at
    ``transformers/models/deepseek_v3/modeling_deepseek_v3.py`` is the
    correctness oracle.
  * HF state_dict loading goes through a custom ``_load_from_state_dict``
    on each block that maps HF keys to the un-fused ``nn.Parameter`` names
    used here. The driver
    (``demo/deepseek_v3/demo_new.py``) is responsible for KV-absorption and
    W_UV→o_proj fusion **before** ``load_state_dict()``.

Scope (deliberately reduced for v1)
-----------------------------------

* **BF16 only.** No FP8 paths — the catalog ``Linear`` /
  ``LinearWithResidual`` / ``MoEW13(bf16)`` / ``MoEW2(bf16)`` modules are
  used. FP8 catalog modules (``LinearFP8`` etc.) are deferred.
* **Single GPU only.** ``world_size=1``, ``ep_size=1``, no NVShmem.
* **Decode-only.** ``max_num_batched_tokens<=8``. Uses ``MLADecode`` +
  ``MLAReduce`` (with ``num_splits=1`` when ``max_seq_length/page_size<=1``).
  No prefill path, no chunked prefill, no MTP.
* **No BMM Q, no qb_fused, no direct_paged_decode_kv.** Always:
  ``MLAKVGather(variant="standard")`` + ``contiguous_kv``.
* **Per-layer intermediates.** Every ``pk.new_tensor`` allocation is
  per-decoder-layer (no sharing across layers).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...context import current_pk
from ...layers import (
    AllReduce,
    ArgmaxPartial,
    ArgmaxReduce,
    AssembleQDecode,
    Embed,
    FP8GroupGEMMSmallM,
    FusedRMSNormQuantizeFP8,
    Linear,
    LinearWithResidual,
    MPKModule,
    MLADecode,
    MLAKVGather,
    MLAReduce,
    MLARopeK,
    MLARopeQ,
    MoeMulSumAdd,
    MoEPermute,
    MoESiluMul,
    MoETopkRouting,
    MoEUnpermute,
    MoEW13,
    MoEW2,
    QuantizeFP8F32Scale,
    QuantizeFP8UE8M0,
    RMSNorm,
    RotaryEmbedding,
    TransposeScale,
)
from ...layers.linear.linear_fp8 import _dequant_fp8


def _packed_scale_k(reduction_size: int) -> int:
    """UE8M0 packed-scale K count: ceil(ceil(K/128)/4) (4 bytes/uint32)."""
    num_groups = (reduction_size + 127) // 128
    return (num_groups + 3) // 4


def _dequant_fp8_blockwise(w_fp8, scale_f32, block: int = 128):
    """Dequant an FP8 weight ``(N, K)`` with a 128x128-block f32 scale
    ``(N//block, K//block)`` to fp32 ``(N, K)`` — the layout
    ``fp8_gemm_dense_smallm`` expects for ``weight_scale`` (DeepSeek V3
    native FP8 weight quantization)."""
    w = (w_fp8.view(torch.float8_e4m3fn) if w_fp8.dtype == torch.uint8
         else w_fp8).float()
    N, K = w.shape
    nb, kb = N // block, K // block
    return (w.reshape(nb, block, kb, block)
            * scale_f32.reshape(nb, 1, kb, 1)).reshape(N, K)


def _swapab_grid(out_features: int, num_workers: int) -> int:
    """grid.x for an FP8 swapAB GEMM: largest divisor of ``out_features//128``
    that is ``<= num_workers``, so per-task output is a 128-multiple
    (MMA_M=128) and grid.x divides the output cleanly.
    """
    if out_features % 128 != 0:
        raise ValueError(f"swapAB out_features={out_features} must be %128==0")
    blocks = out_features // 128
    g = 1
    for d in range(1, blocks + 1):
        if blocks % d == 0 and d <= num_workers:
            g = d
    return g


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grid_for_linear(size: int) -> int:
    """Mirror ``grid_for_rmsnorm_linear_layer`` from demo/deepseek_v3/demo.py.

    Picks the tile divisor that the kernel's task atom expects. Order of
    preference matches both qwen3 and deepseek demos.
    """
    if size / 96 > 400:
        assert size % 256 == 0, f"linear size not supported: {size}"
        return size // 256
    if size % 96 == 0:
        return 96
    if size % 64 == 0:
        return 64
    raise ValueError(f"linear out-dim {size} not divisible by 96 or 64")


def _moe_hidden_split(hidden_size: int, preferred: int = 56) -> int:
    """Pick a valid hidden-dimension split for the MoE mul_sum_add epilogue.

    Mirrors :func:`_moe_hidden_split` in
    ``python/mirage/mpk/models/deepseek_v3/builder.py``. Must be a divisor
    of ``hidden_size`` AND the per-CTA slab (``hidden_size // y``) must be
    a 128-multiple (the underlying kernel's epilogue tile).
    """
    max_y = min(preferred, max(1, hidden_size // 128))
    for y in range(max_y, 0, -1):
        if hidden_size % y == 0 and (hidden_size // y) % 128 == 0:
            return y
    return 1


# ---------------------------------------------------------------------------
# DeepseekV3MLA
# ---------------------------------------------------------------------------


class DeepseekV3MLA(MPKModule):
    """Multi-head Latent Attention (decode-only, BF16, single GPU).

    Pipeline:
        1. ``q_a_proj``  : Linear ``(hidden -> q_lora_rank)``, BF16.
        2. ``q_a_layernorm`` : RMSNorm over ``q_a_proj`` output.
        3. ``q_b_proj``  : Linear ``(q_lora_rank -> H * (kv_lora_rank +
           qk_rope_head_dim))`` — KV-absorbed; the driver MUST apply
           absorption via ``absorb_kv_into_q`` before ``load_state_dict()``.
        4. ``kv_a_proj_with_mqa`` : Linear ``(hidden -> kv_lora_rank +
           qk_rope_head_dim)``, BF16. Output split into ``c_latent`` and
           ``k_pe``.
        5. ``kv_a_layernorm`` : RMSNorm over the ``c_latent`` half only.
        6. RoPE on Q (fused per-head NoPE-PE layout) and K (PE only).
        7. ``mla_kv_gather`` (standard) : appends to per-layer paged
           ``ckv_kpe_cache`` and materialises a contiguous ``(R * S, D_K)``
           slab for the decode kernel.
        8. ``mla_decode`` + ``mla_reduce`` (split-K = 1 when the per-request
           KV length fits a single 128-token tile; otherwise the catalog
           default split-K applies).
        9. ``o_proj`` : Linear ``(H * kv_lora_rank -> hidden)`` with residual
           — the W_UV absorption into o_proj is performed by the driver at
           load time (the resulting fused weight is ``(hidden,
           H * kv_lora_rank)``, NOT the HF-native ``(hidden, H * v_head_dim)``).
    """

    def __init__(self, config, layer_idx: int, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        # MLA per-token KV-latent width after absorption.
        # 576 = kv_lora_rank(512) + qk_rope_head_dim(64) for DeepSeek V3.
        self.qk_head_dim = self.kv_lora_rank + self.qk_rope_head_dim
        # v_head_dim ABSORBED == kv_lora_rank (the absorbed attention emits
        # output of width kv_lora_rank).
        self.v_head_dim = self.kv_lora_rank

        # ---- Layernorm scales ----------------------------------------
        self.q_a_layernorm = nn.Parameter(torch.empty(self.q_lora_rank))
        self.kv_a_layernorm = nn.Parameter(torch.empty(self.kv_lora_rank))

        # ---- Linear weights (raw nn.Parameter; the v1 path does NOT
        #     fuse qkv_a at compile time — keeping the wiring minimal).
        # q_a_proj: (q_lora_rank, hidden)
        self.q_a_proj_weight = nn.Parameter(
            torch.empty(self.q_lora_rank, self.hidden_size)
        )
        # kv_a_proj_with_mqa: (kv_lora_rank + qk_rope_head_dim, hidden)
        self.kv_a_proj_with_mqa_weight = nn.Parameter(
            torch.empty(
                self.kv_lora_rank + self.qk_rope_head_dim,
                self.hidden_size,
            )
        )
        # q_b_proj (KV-absorbed): (H * (kv_lora_rank + qk_rope_head_dim),
        # q_lora_rank) — driver applies absorption before load_state_dict.
        self.q_b_proj_weight = nn.Parameter(
            torch.empty(self.num_heads * self.qk_head_dim, self.q_lora_rank)
        )
        # o_proj (W_UV-fused): (hidden, H * kv_lora_rank). HF native is
        # (hidden, H * v_head_dim); the driver fuses W_UV in at load time.
        self.o_proj_weight = nn.Parameter(
            torch.empty(self.hidden_size, self.num_heads * self.kv_lora_rank)
        )

        # ---- Catalog leaves (no parameters in catalog; just dispatch) ----
        self.rope_q = MLARopeQ(num_heads=self.num_heads, variant="fused")
        self.rope_k = MLARopeK()
        self.kv_gather = MLAKVGather(
            d_k=self.qk_head_dim,
            d_v=self.kv_lora_rank,
            page_size=getattr(config, "page_size", 128),
            variant="standard",
        )
        # decode/reduce concrete params are filled in by compile() (they
        # depend on pk.max_seq_length and pk.page_size at compile time).
        self._decode = None
        self._reduce = None

    # ------------------------------------------------------------------
    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        # Map HF keys to our parameters. The driver is responsible for
        # producing the absorbed q_b_proj and fused o_proj before calling
        # load_state_dict (see demo_new.py::_load_hf_weights_with_absorption).
        for hf_name, param in [
            ("q_a_proj.weight", self.q_a_proj_weight),
            ("q_b_proj.weight", self.q_b_proj_weight),
            ("kv_a_proj_with_mqa.weight", self.kv_a_proj_with_mqa_weight),
            ("o_proj.weight", self.o_proj_weight),
            ("q_a_layernorm.weight", self.q_a_layernorm),
            ("kv_a_layernorm.weight", self.kv_a_layernorm),
        ]:
            hf_key = prefix + hf_name
            if hf_key in state_dict:
                with torch.no_grad():
                    param.copy_(state_dict.pop(hf_key))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs
        )

    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        # The MLA pipeline (paged KV, RoPE on a slice, group-limited routing,
        # split-K decode) is intrinsically tied to MPK runtime state. The
        # official HF reference at transformers/models/deepseek_v3/
        # modeling_deepseek_v3.py is the correctness oracle for forward().
        raise NotImplementedError(
            "DeepseekV3MLA.forward() is not implemented in the MPK catalog. "
            "Use transformers.DeepseekV3ForCausalLM for eager-mode reference."
        )

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError(
            "composite module — see child compile()s"
        )

    # ------------------------------------------------------------------
    def compile(self, x_dt, cos_dt, sin_dt, *, residual_dt, output):
        """Build the MLA task graph for one decoder layer.

        Args:
            x_dt: input DTensor of shape ``(mbt, hidden)`` (post-input-RMSnorm).
            cos_dt / sin_dt: RoPE tables, shape ``(max_seq_len, D_PE)``.
            residual_dt: residual DTensor for the o_proj+residual epilogue.
            output: destination DTensor for ``attn_proj_out`` (the output
                of o_proj+residual). Shape ``(mbt, hidden)``.
        """
        pk = current_pk()
        from ....core import bfloat16 as _mi_bf16, float32 as _mi_f32

        mbt = pk.max_num_batched_tokens
        mbr = pk.max_num_batched_requests
        H = self.num_heads
        D_K = self.qk_head_dim          # 576
        D_V = self.kv_lora_rank          # 512
        D_PE = self.qk_rope_head_dim     # 64
        D_NOPE = self.kv_lora_rank       # in absorbed path, "NoPE" half == c_latent width
        page_size = pk.page_size
        kv_len_max = pk.max_seq_length

        # ---- Per-layer intermediate tensors --------------------------
        # q_a_out (mbt, q_lora_rank)
        per_layer_q_a_out = pk.new_tensor(
            dims=(mbt, self.q_lora_rank),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_q_a_out",
        )
        # q_nope_pe (mbt, H * (kv_lora_rank + qk_rope_head_dim))
        per_layer_q_nope_pe = pk.new_tensor(
            dims=(mbt, H * D_K),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_q_nope_pe",
        )
        # kv_a_out (mbt, kv_lora_rank + qk_rope_head_dim) — c_latent + k_pe.
        per_layer_kv_a_out = pk.new_tensor(
            dims=(mbt, self.kv_lora_rank + self.qk_rope_head_dim),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_kv_a_out",
        )
        # contiguous_kv: gather destination, shape (R*S_max, D_K).
        per_layer_contig_kv = pk.new_tensor(
            dims=(mbr * kv_len_max, D_K),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_contiguous_kv",
        )
        # MLA decode partial outputs. Use a single split when the per-
        # request KV range fits one 128-token tile (max_seq_len/page_size
        # <= 1 in MLA terms). Otherwise the natural deepseek scheme
        # of num_splits = ceil(kv_len_max / 128) applies.
        max_kv_tiles = (kv_len_max + 127) // 128
        # Force num_splits=1 for v1 (only valid when max_kv_tiles <= 1).
        # Fall back to the deepseek default split count otherwise.
        if max_kv_tiles <= 1:
            num_splits = 1
        else:
            num_splits = max_kv_tiles
        per_layer_partial_o = pk.new_tensor(
            dims=(mbr * 1 * num_splits, H * D_V),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_partial_o",
        )
        per_layer_partial_lse = pk.new_tensor(
            dims=(mbr * 1 * num_splits, H),
            dtype=_mi_f32,
            name=f"{self.prefix}per_layer_partial_lse",
        )
        # attn_out (mbt, H * kv_lora_rank)
        per_layer_attn_out = pk.new_tensor(
            dims=(mbt, H * D_V),
            dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_attn_out",
        )

        # ---- Attach raw weights as DTensors --------------------------
        w_q_a_dt = pk.attach_input(
            self.q_a_proj_weight, name=f"{self.prefix}q_a_proj_weight"
        )
        w_q_b_dt = pk.attach_input(
            self.q_b_proj_weight, name=f"{self.prefix}q_b_proj_weight"
        )
        w_kv_a_dt = pk.attach_input(
            self.kv_a_proj_with_mqa_weight,
            name=f"{self.prefix}kv_a_proj_with_mqa_weight",
        )
        w_q_a_ln_dt = pk.attach_input(
            self.q_a_layernorm, name=f"{self.prefix}q_a_layernorm"
        )
        w_kv_a_ln_dt = pk.attach_input(
            self.kv_a_layernorm, name=f"{self.prefix}kv_a_layernorm"
        )
        w_o_dt = pk.attach_input(
            self.o_proj_weight, name=f"{self.prefix}o_proj_weight"
        )

        # ---- 1. q_a_proj : Linear (BF16) -----------------------------
        q_a_grid = _grid_for_linear(self.q_lora_rank)
        pk.linear_layer(
            input=x_dt,
            weight=w_q_a_dt,
            output=per_layer_q_a_out,
            grid_dim=(q_a_grid, 1, 1),
            block_dim=(128, 1, 1),
        )

        # ---- 2. q_a_layernorm : RMSNorm (in-place) -------------------
        pk.rmsnorm_layer(
            input=per_layer_q_a_out,
            weight=w_q_a_ln_dt,
            output=per_layer_q_a_out,
            grid_dim=(mbt, 1, 1),
            block_dim=(128, 1, 1),
        )

        # ---- 3. q_b_proj : Linear ------------------------------------
        q_b_grid = _grid_for_linear(H * D_K)
        pk.linear_layer(
            input=per_layer_q_a_out,
            weight=w_q_b_dt,
            output=per_layer_q_nope_pe,
            grid_dim=(q_b_grid, 1, 1),
            block_dim=(128, 1, 1),
        )

        # ---- 4. kv_a_proj_with_mqa : Linear --------------------------
        kv_a_grid = _grid_for_linear(self.kv_lora_rank + self.qk_rope_head_dim)
        pk.linear_layer(
            input=x_dt,
            weight=w_kv_a_dt,
            output=per_layer_kv_a_out,
            grid_dim=(kv_a_grid, 1, 1),
            block_dim=(128, 1, 1),
        )

        # ---- 5. kv_a_layernorm : RMSNorm over the c_latent slice -----
        # The kv_a_out row layout is [c_latent (kv_lora_rank) | k_pe (D_PE)].
        # rmsnorm operates on the [0:kv_lora_rank) slice in place.
        pk.rmsnorm_layer(
            input=per_layer_kv_a_out,
            weight=w_kv_a_ln_dt,
            output=per_layer_kv_a_out,
            grid_dim=(mbt, 1, 1),
            block_dim=(128, 1, 1),
            process_dim=self.kv_lora_rank,
            in_offset_elems=0,
            out_offset_elems=0,
        )

        # ---- 6a. RoPE on Q (fused per-head [NoPE | PE]) -------------
        # The fused Q tensor's per-row layout is
        # [h0_nope (D_NOPE=512) | h0_pe (D_PE=64) | h1_nope | h1_pe | ...].
        # The MLA fused-RoPE kernel applies rotate-half to the PE slice
        # of each head in place.
        self.rope_q.compile(
            q_pe=per_layer_q_nope_pe,
            cos_pos_embed=cos_dt,
            sin_pos_embed=sin_dt,
        )

        # ---- 6b. RoPE on K (in-place on the k_pe slice of kv_a_out) -
        self.rope_k.compile(
            k_pe=per_layer_kv_a_out,
            cos_pos_embed=cos_dt,
            sin_pos_embed=sin_dt,
            # The k_pe slice lives at [kv_lora_rank : kv_lora_rank + D_PE)
            # within each row of the (kv_lora_rank + D_PE)-wide kv_a_out.
            k_pe_row_stride=self.kv_lora_rank + self.qk_rope_head_dim,
            k_pe_offset=self.kv_lora_rank,
        )

        # ---- 7. MLA KV gather (standard variant) --------------------
        # Attaches the per-layer paged KV cache pool, appends the new
        # c_latent / k_pe rows from kv_a_out (the row stride / offset
        # kwargs let the gather kernel read the c_latent slice from the
        # combined kv_a_out buffer), and materialises a contiguous KV
        # slab for the decode kernel.
        k_cache_torch, _ = pk.get_kv_cache(self.layer_idx)
        layer_cache_dt = pk.attach_input(
            k_cache_torch, name=f"{self.prefix}ckv_kpe_cache"
        )
        # c_latent_new and k_pe_new are both views into per_layer_kv_a_out.
        # We pass kv_a_out as BOTH inputs; the kernel uses the (row_stride,
        # offset) kwargs to address the c_latent slice [0:kv_lora_rank)
        # and the k_pe slice [kv_lora_rank:kv_lora_rank+D_PE) inside it.
        self.kv_gather.compile(
            c_latent_new=per_layer_kv_a_out,
            k_pe_new=per_layer_kv_a_out,
            paged_cache=layer_cache_dt,
            contiguous_kv=per_layer_contig_kv,
            c_latent_row_stride=self.kv_lora_rank + self.qk_rope_head_dim,
            c_latent_offset_elems=0,
            k_pe_row_stride=self.kv_lora_rank + self.qk_rope_head_dim,
            k_pe_offset_elems=self.kv_lora_rank,
        )

        # ---- 8. MLA decode + reduce ---------------------------------
        # Instantiate the catalog modules lazily (kv_len / num_splits
        # depend on pk fields that aren't known at __init__ time).
        if self._decode is None:
            self._decode = MLADecode(
                num_heads=H,
                d_k=D_K,
                d_v=D_V,
                num_splits=num_splits,
                kv_len=kv_len_max,
                q_len=1,
                prefix=self.prefix,
            )
        if self._reduce is None:
            self._reduce = MLAReduce(
                num_heads=H,
                d_v=D_V,
                num_splits=num_splits,
                d_start=0,
                d_count=2,
                q_len=1,
                prefix=self.prefix,
            )

        self._decode.compile(
            q_input=per_layer_q_nope_pe,
            kv_input=per_layer_contig_kv,
            output_partial=per_layer_partial_o,
            output_lse=per_layer_partial_lse,
        )
        self._reduce.compile(
            input_partial=per_layer_partial_o,
            input_lse=per_layer_partial_lse,
            output=per_layer_attn_out,
        )

        # ---- 9. o_proj + residual -----------------------------------
        # Output shape: (mbt, hidden_size). The fused W_UV * W_o weight
        # has shape (hidden, H * kv_lora_rank); produced by the driver.
        pk.linear_with_residual_layer(
            input=per_layer_attn_out,
            weight=w_o_dt,
            residual=residual_dt,
            output=output,
            grid_dim=(self.hidden_size // 64, 1, 1),
            block_dim=(128, 1, 1),
        )
        return output


# ---------------------------------------------------------------------------
# DeepseekV3MLP (dense MLP for layers 0..first_k_dense_replace-1)
# ---------------------------------------------------------------------------


class DeepseekV3MLP(MPKModule):
    """Dense gated MLP in FP8 (decode), HF-faithful reference.

    Mirrors HF ``DeepseekV3MLP.forward``: ``down(silu(gate(x)) * up(x))``
    (+ residual). The ``compile`` path is FP8 on SM100:

      1. ``QuantizeFP8UE8M0`` the input → ``(x_fp8, x_scale)``.
      2. ONE FP8 swapAB GEMM over the **concat-fused** ``gate_up`` weight
         (rows ``[0:I]`` = gate, ``[I:2I]`` = up) → ``(mbt, 2I)`` whose
         columns are therefore ``[gate(I) | up(I)]``.
      3. ``silu_mul`` with ``grid.x = 1`` (plain halved ``[gate|up]`` layout,
         no per-task interleave) → ``(mbt, I)``.
      4. ``QuantizeFP8UE8M0`` the silu output, then a swapAB GEMM over the
         ``down`` weight with the bf16 residual fused → ``(mbt, hidden)``.

    Used both as the dense MLP (``intermediate_size``) for layers
    ``0..first_k_dense_replace-1`` and as the shared expert
    (``moe_intermediate_size * n_shared_experts``) inside the MoE block.

    Weights are stored as raw FP8 ``uint8`` + UE8M0-packed ``uint32`` scales;
    the driver produces the concat-fused ``gate_up`` weight at load time.
    """

    def __init__(self, config, *, intermediate_size=None, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.hidden_size = config.hidden_size
        self.intermediate_size = (
            intermediate_size if intermediate_size is not None
            else config.intermediate_size
        )
        H, I = self.hidden_size, self.intermediate_size

        # Concat-fused gate_up: rows [0:I]=gate, [I:2I]=up. FP8 E4M3 (uint8)
        # + 128x128-block f32 weight scale (DeepSeek V3 native FP8 layout,
        # consumed by fp8_gemm_dense_smallm).
        self.gate_up_weight = nn.Parameter(
            torch.empty(2 * I, H, dtype=torch.uint8), requires_grad=False
        )
        self.gate_up_scale = nn.Parameter(
            torch.empty(2 * I // 128, H // 128, dtype=torch.float32),
            requires_grad=False,
        )
        self.down_weight = nn.Parameter(
            torch.empty(H, I, dtype=torch.uint8), requires_grad=False
        )
        self.down_scale = nn.Parameter(
            torch.empty(H // 128, I // 128, dtype=torch.float32),
            requires_grad=False,
        )

        # Activation quantizers (f32 1x128 scale → fp8_gemm_dense_smallm).
        self.q_in = QuantizeFP8F32Scale(H, prefix=f"{prefix}q_in_")
        self.q_silu = QuantizeFP8F32Scale(I, prefix=f"{prefix}q_silu_")

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        # The driver pre-builds the concat-fused FP8 gate_up + 128x128 scale
        # and the FP8 down + scale under these keys.
        for hf_name, param in [
            ("gate_up_proj.weight", self.gate_up_weight),
            ("gate_up_proj.weight_scale_inv", self.gate_up_scale),
            ("down_proj.weight", self.down_weight),
            ("down_proj.weight_scale_inv", self.down_scale),
        ]:
            hf_key = prefix + hf_name
            if hf_key in state_dict:
                with torch.no_grad():
                    param.copy_(state_dict.pop(hf_key))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs
        )

    def forward(self, x, residual):
        """HF reference, dequantizing the stored FP8 weights (fp32 math)."""
        I = self.intermediate_size
        gu = _dequant_fp8_blockwise(self.gate_up_weight, self.gate_up_scale)
        gate = F.linear(x.float(), gu[:I])
        up = F.linear(x.float(), gu[I:])
        silu = F.silu(gate) * up
        dn = _dequant_fp8_blockwise(self.down_weight, self.down_scale)
        out = F.linear(silu, dn) + residual.float()
        return out.to(x.dtype)

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite — see child compile()s")

    def compile(self, x_dt, residual_dt, *, output):
        pk = current_pk()
        from ....core import bfloat16 as _mi_bf16

        mbt = pk.max_num_batched_tokens
        H, I = self.hidden_size, self.intermediate_size
        nw = pk.num_workers

        # 1. quantize input → FP8 + f32 1x128 scale (M, H/128).
        x_fp8, x_scale = self.q_in.compile(x_dt)

        # 2. gate_up dense GEMM (concat-fused weight) → (mbt, 2I) = [gate|up].
        w_gu = pk.attach_input(
            self.gate_up_weight, name=f"{self.prefix}gate_up_weight"
        )
        ws_gu = pk.attach_input(
            self.gate_up_scale, name=f"{self.prefix}gate_up_scale"
        )
        mlp_mid = pk.new_tensor(
            dims=(mbt, 2 * I), dtype=_mi_bf16, name=f"{self.prefix}mlp_mid",
        )
        pk.fp8_gemm_dense_smallm_layer(
            input_fp8=x_fp8, weight_fp8=w_gu,
            input_scale=x_scale, weight_scale=ws_gu,
            output=mlp_mid, num_workers=nw,
        )

        # 3. silu_mul (grid.x=1 → plain [gate|up] halved layout).
        silu_out = pk.new_tensor(
            dims=(mbt, I), dtype=_mi_bf16, name=f"{self.prefix}silu_out",
        )
        pk.silu_mul_layer(
            input=mlp_mid, output=silu_out,
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
        )

        # 4. quantize silu output, down dense GEMM → partial, + residual.
        silu_fp8, silu_scale = self.q_silu.compile(silu_out)
        w_dn = pk.attach_input(
            self.down_weight, name=f"{self.prefix}down_weight"
        )
        ws_dn = pk.attach_input(
            self.down_scale, name=f"{self.prefix}down_scale"
        )
        partial = pk.new_tensor(
            dims=(mbt, H), dtype=_mi_bf16, name=f"{self.prefix}down_partial",
        )
        pk.fp8_gemm_dense_smallm_layer(
            input_fp8=silu_fp8, weight_fp8=w_dn,
            input_scale=silu_scale, weight_scale=ws_dn,
            output=partial, num_workers=nw,
        )
        pk.elementwise_add_layer(
            input_a=partial, input_b=residual_dt, output=output,
            grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
        )
        return output


# ---------------------------------------------------------------------------
# DeepseekV3MoEMLP (MoE MLP for layers first_k_dense_replace..)
# ---------------------------------------------------------------------------


class DeepseekV3MoE(MPKModule):
    """DeepSeek V3 MoE block (HF ``DeepseekV3MoE``), FP8 permute-based path.

    ``forward`` mirrors HF: router gate -> per-expert routed loop ->
    ``+ shared_experts(x)``, with the transformer residual folded into the
    shared-expert output (so the decoder layer must NOT re-add it).
    ``compile`` builds the validated grouped-GEMM permute pipeline (see
    ``tests/runtime_python/test_mode/test_dsv3_moe_permute_chain_testmode.py``):

        router GEMM (bf16) -> MoETopkRouting(sigmoid) -> QuantizeFP8(ue8m0)
        -> MoEPermute -> FP8GroupGEMM(w13) -> MoESiluMul -> QuantizeFP8(ue8m0)
        -> FP8GroupGEMM(w2) -> [shared expert = DeepseekV3MLP] ->
        MoEUnpermute(routed combine + shared_out).

    TP/EP-aware: ``ep_size`` (from the parallel config) shards the routed
    experts (``local_num_experts = n_routed_experts // ep_size``). v1 is
    tested at ``ep_size == 1`` (single GPU).
    """

    def __init__(self, config, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        pc = current_pk().parallel_config
        self.ep_size = getattr(pc, "ep_size", 1)
        ep_rank = getattr(pc, "ep_rank", 0)
        self.hidden_size = H = config.hidden_size
        self.moe_intermediate_size = I = config.moe_intermediate_size
        self.num_experts = E = config.n_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_shared_experts = getattr(config, "n_shared_experts", 1)
        self.num_groups = getattr(config, "n_group", 8)
        self.topk_group = getattr(config, "topk_group", 4)
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 2.5)
        self.bm_padding = 128
        self.local_num_experts = E // self.ep_size
        self.local_expert_start = ep_rank * self.local_num_experts
        self._m_total = self.local_num_experts * self.bm_padding

        # Router (bf16) + group-limited sigmoid routing (owns e_score bias).
        self.gate_weight = nn.Parameter(torch.empty(E, H))
        self.routing = MoETopkRouting(
            num_experts=E, num_experts_per_tok=self.num_experts_per_tok,
            variant="sigmoid", num_groups=self.num_groups,
            topk_group=self.topk_group,
            routed_scaling_factor=self.routed_scaling_factor,
            local_num_experts=self.local_num_experts,
            local_expert_start=self.local_expert_start, prefix=f"{prefix}gate_")
        # Routed experts: own FP8 (E_local, N, K) weights + K-outer scales.
        self.experts_w13 = FP8GroupGEMMSmallM(
            self.local_num_experts, in_features=H, out_features=2 * I,
            prefix=f"{prefix}experts_w13_")
        self.experts_silu = MoESiluMul(I, prefix=f"{prefix}experts_silu_")
        self.experts_w2 = FP8GroupGEMMSmallM(
            self.local_num_experts, in_features=I, out_features=H,
            prefix=f"{prefix}experts_w2_")
        self.permute = MoEPermute(
            self.local_num_experts, H, self.num_experts_per_tok,
            bm_padding=self.bm_padding, prefix=f"{prefix}permute_")
        self.unpermute = MoEUnpermute(H, prefix=f"{prefix}unpermute_")
        # Shared expert (DeepseekV3MLP with moe_int * n_shared).
        self.shared_experts = DeepseekV3MLP(
            config, intermediate_size=I * self.num_shared_experts,
            prefix=f"{prefix}shared_experts_")
        self.register_buffer(
            "m_indices",
            torch.arange(self._m_total, dtype=torch.int32) // self.bm_padding,
            persistent=False)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        gk = prefix + "gate.weight"
        if gk in state_dict:
            with torch.no_grad():
                self.gate_weight.copy_(state_dict.pop(gk))
        bk = prefix + "gate.e_score_correction_bias"
        if bk in state_dict:
            with torch.no_grad():
                self.routing.bias.copy_(state_dict.pop(bk).to(torch.float32))
        # Driver pre-stacks + UE8M0-packs the routed expert weights.
        for hf, p in [
            ("experts.w13.weight", self.experts_w13.weight),
            ("experts.w13.weight_scale", self.experts_w13.weight_scale),
            ("experts.w2.weight", self.experts_w2.weight),
            ("experts.w2.weight_scale", self.experts_w2.weight_scale),
        ]:
            k = prefix + hf
            if k in state_dict:
                with torch.no_grad():
                    p.copy_(state_dict.pop(k))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def _dequant_group(self, gg, block: int = 128):
        """Dequant a FP8GroupGEMMSmallM's owned (E,N,K) weight to fp32 — same
        math as :meth:`FP8GroupGEMMSmallM.forward`'s SFB path."""
        E, N, K = gg.num_experts, gg.out_features, gg.in_features
        nk = K // block
        num_sf_k = (nk + 3) // 4
        w = gg.weight.view(torch.float8_e4m3fn).float()
        sfb = gg.weight_scale.view(num_sf_k, E, N).permute(1, 2, 0).contiguous()
        sfb_bytes = sfb.view(torch.uint8).reshape(E, N, num_sf_k * 4)[..., :nk]
        sfb_f32 = torch.pow(torch.tensor(2.0, device=w.device),
                            sfb_bytes.float() - 127.0)
        sfb_exp = sfb_f32.repeat_interleave(block, dim=-1)[..., :K]
        return w * sfb_exp

    def forward(self, x, residual):
        """HF reference: gate -> routed loop -> + shared(x) (+ residual)."""
        H, I = self.hidden_size, self.moe_intermediate_size
        topk = self.num_experts_per_tok
        x2 = x.view(-1, H)
        res2 = residual.view(-1, H)
        M = x2.shape[0]
        logits = F.linear(x2.float(), self.gate_weight.float())
        topk_w, routing_indices, _ = self.routing.forward(logits)
        ri = routing_indices  # (E_local, M)
        w13 = self._dequant_group(self.experts_w13)  # (E_local, 2I, H)
        w2 = self._dequant_group(self.experts_w2)    # (E_local, H, I)
        routed = torch.zeros(M, H, dtype=torch.float32, device=x.device)
        for e in range(self.local_num_experts):
            for t in range(M):
                s = int(ri[e, t].item())
                if s <= 0:
                    continue
                gu = x2[t].float() @ w13[e].t()
                act = F.silu(gu[:I]) * gu[I:]
                routed[t] += topk_w[t, s - 1].item() * (act @ w2[e].t())
        shared = self.shared_experts.forward(x2, res2)  # shared(x) + residual
        return (routed.to(x.dtype) + shared).view_as(x)

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite — see child compile()s")

    def compile(self, x_dt, residual_dt, *, output):
        pk = current_pk()
        from ....core import (bfloat16 as _bf16, float8_e4m3 as _fp8,
                              float32 as _f32, uint32 as _u32, int32 as _i32)
        mbt = pk.max_num_batched_tokens
        H, I, E = self.hidden_size, self.moe_intermediate_size, self.num_experts
        E_local = self.local_num_experts
        topk = self.num_experts_per_tok
        m_total = self._m_total
        K_PACKED = _packed_scale_k(H)
        K_PACKED_I = _packed_scale_k(I)
        nw = pk.num_workers
        p = self.prefix

        # 1. router GEMM (bf16).
        w_gate = pk.attach_input(self.gate_weight, name=f"{p}gate_weight")
        logits = pk.new_tensor(dims=(mbt, E), dtype=_bf16, name=f"{p}router_logits")
        rg = max(1, min(_grid_for_linear(E) if E % 64 == 0 else 1, max(1, E // 8)))
        pk.linear_layer(input=x_dt, weight=w_gate, output=logits,
                        grid_dim=(rg, 1, 1), block_dim=(128, 1, 1))
        # 2. routing.
        topk_w = pk.new_tensor(dims=(mbt, topk), dtype=_f32, name=f"{p}topk_w")
        routing_idx = pk.new_tensor(dims=(E_local, mbt), dtype=_i32, name=f"{p}routing_idx")
        moe_mask = pk.new_tensor(dims=(E_local + 1,), dtype=_i32, name=f"{p}moe_mask")
        self.routing.compile(logits, topk_w, routing_idx, moe_mask)
        # 3. quantize input (UE8M0).
        in_fp8 = pk.new_tensor(dims=(mbt, H), dtype=_fp8, name=f"{p}in_fp8")
        in_scale = pk.new_tensor(dims=(mbt, K_PACKED), dtype=_u32, name=f"{p}in_scale")
        pk.quantize_fp8_layer(input=x_dt, output_fp8=in_fp8, output_scale=in_scale,
                              grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
                              scale_ue8m0=True)
        # 4. meta + permute.
        meta = pk.new_tensor(dims=(2, m_total + mbt * topk), dtype=_i32, name=f"{p}meta")
        pk.tensor_init_layer(target=meta, dummy=in_fp8, grid_dim=(1, 1, 1),
                             block_dim=(128, 1, 1), dummy_input_map=(-1, -1, -1),
                             target_input_map=(-1, -1, -1))
        perm_fp8 = pk.new_tensor(dims=(m_total, H), dtype=_fp8, name=f"{p}perm_fp8")
        perm_scale = pk.new_tensor(dims=(K_PACKED, m_total), dtype=_u32, name=f"{p}perm_scale")
        self.permute.compile(in_fp8, in_scale, topk_w, routing_idx,
                             perm_fp8, perm_scale, meta)
        # 5. w13 group GEMM.
        m_idx = pk.attach_input(self.m_indices, name=f"{p}m_indices")
        w13_out = pk.new_tensor(dims=(m_total, 2 * I), dtype=_bf16, name=f"{p}w13_out")
        self.experts_w13.compile(perm_fp8, perm_scale, m_idx, w13_out, num_workers=nw)
        # 6. silu_mul.
        silu_out = pk.new_tensor(dims=(m_total, I), dtype=_bf16, name=f"{p}silu_out")
        self.experts_silu.compile(w13_out, output=silu_out)
        # 7. quantize silu (UE8M0, K-outer (K_PACKED_I, m_total)).
        silu_fp8 = pk.new_tensor(dims=(m_total, I), dtype=_fp8, name=f"{p}silu_fp8")
        silu_scale = pk.new_tensor(dims=(K_PACKED_I, m_total), dtype=_u32, name=f"{p}silu_scale")
        pk.quantize_fp8_layer(input=silu_out, output_fp8=silu_fp8,
                              output_scale=silu_scale, grid_dim=(m_total, 1, 1),
                              block_dim=(128, 1, 1), scale_ue8m0=True,
                              process_all_rows=True)
        # 8. w2 group GEMM.
        w2_out = pk.new_tensor(dims=(m_total, H), dtype=_bf16, name=f"{p}w2_out")
        self.experts_w2.compile(silu_fp8, silu_scale, m_idx, w2_out, num_workers=nw)
        # 9. shared expert (DeepseekV3MLP: shared(x) + residual).
        shared_out = pk.new_tensor(dims=(mbt, H), dtype=_bf16, name=f"{p}shared_out")
        self.shared_experts.compile(x_dt, residual_dt, output=shared_out)
        # 10. unpermute: output = shared_out + routed-combine.
        self.unpermute.compile(permuted_output=w2_out, meta=meta,
                               residual=shared_out, output=output)
        return output


# Backward-compat alias (decoder layer references the old name until rewritten).
DeepseekV3MoEMLP = DeepseekV3MoE


class DeepseekV3DecoderLayer(MPKModule):
    """One decoder layer = input-RMSnorm → MLA → post-attn-RMSnorm → MLP.

    The MLP is dense (:class:`DeepseekV3MLP`) for ``layer_idx <
    first_k_dense_replace`` and MoE (:class:`DeepseekV3MoEMLP`) thereafter.
    """

    def __init__(self, config, layer_idx: int, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.layer_idx = layer_idx
        self.input_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            prefix=f"{prefix}input_layernorm_",
        )
        self.self_attn = DeepseekV3MLA(
            config, layer_idx, prefix=f"{prefix}self_attn_"
        )
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            prefix=f"{prefix}post_attention_layernorm_",
        )
        first_moe = getattr(config, "first_k_dense_replace", 3)
        if layer_idx < first_moe:
            self.mlp = DeepseekV3MLP(config, prefix=f"{prefix}mlp_")
            self.is_moe = False
        else:
            self.mlp = DeepseekV3MoEMLP(config, prefix=f"{prefix}mlp_")
            self.is_moe = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepseekV3DecoderLayer.forward() not implemented. "
            "Use transformers reference."
        )

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite — see child compile()s")

    def compile(self, x_dt, cos_dt, sin_dt):
        pk = current_pk()
        from ....core import bfloat16 as _mi_bf16

        hidden = self.input_layernorm.hidden_size

        per_layer_rmsnorm_attn_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_rmsnorm_attn_out",
        )
        per_layer_attn_proj_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_attn_proj_out",
        )
        per_layer_rmsnorm_mlp_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_rmsnorm_mlp_out",
        )
        per_layer_mlp_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}per_layer_mlp_out",
        )

        # Input RMSNorm → MLA (with residual fused into o_proj).
        self.input_layernorm.compile(
            x_dt,
            output=per_layer_rmsnorm_attn_out,
            grid_dim=(pk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        self.self_attn.compile(
            per_layer_rmsnorm_attn_out, cos_dt, sin_dt,
            residual_dt=x_dt,
            output=per_layer_attn_proj_out,
        )

        # Post-attention RMSNorm → MLP. Dense MLP fuses residual into
        # down_proj. MoE MLP fuses residual into the shared-expert path
        # and the final mul_sum_add reduces over the routed expert outputs
        # (so per_layer_mlp_out is the post-MLP, post-residual hidden state).
        self.post_attention_layernorm.compile(
            per_layer_attn_proj_out,
            output=per_layer_rmsnorm_mlp_out,
            grid_dim=(pk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        self.mlp.compile(
            per_layer_rmsnorm_mlp_out,
            residual_dt=per_layer_attn_proj_out,
            output=per_layer_mlp_out,
        )
        return per_layer_mlp_out


# ---------------------------------------------------------------------------
# DeepseekV3Model
# ---------------------------------------------------------------------------


class DeepseekV3Model(MPKModule):
    def __init__(self, config, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.config = config
        self.embed_tokens = Embed(
            config.vocab_size, config.hidden_size,
            prefix=f"{prefix}embed_tokens_",
        )
        # RoPE on the qk_rope_head_dim (=64) channels — NOT head_dim.
        # We use the plain RotaryEmbedding from the catalog; if the model
        # config has rope_scaling/yarn, the proper YARN-aligned cos/sin
        # would go through builder._precompute_rope_embeddings. v1 falls
        # back to plain RoPE so the modeling can stand alone for the
        # smoke-test, with a recorded TODO to plumb YARN later.
        rope_max = min(4096, getattr(config, "max_position_embeddings", 4096))
        rope_theta = getattr(config, "rope_theta", 10000.0)
        self.rotary_emb = RotaryEmbedding(
            head_dim=config.qk_rope_head_dim,
            max_position_embeddings=rope_max,
            base=rope_theta,
            prefix=f"{prefix}rotary_emb_",
        )
        self.layers = nn.ModuleList([
            DeepseekV3DecoderLayer(
                config, layer_idx=i, prefix=f"{prefix}layers_{i}_"
            )
            for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
            prefix=f"{prefix}norm_",
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepseekV3Model.forward() not implemented. Use transformers."
        )

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite — see child compile()s")

    def compile(self, input_tokens_dt):
        pk = current_pk()
        from ....core import bfloat16 as _mi_bf16

        cos_dt, sin_dt = self.rotary_emb.compile()
        hidden = self.config.hidden_size
        embed_out_dt = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}embed_out",
        )
        self.embed_tokens.compile(
            input_tokens_dt,
            input_source=1,
            output=embed_out_dt,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
        h_dt = embed_out_dt
        for layer in self.layers:
            h_dt = layer.compile(h_dt, cos_dt, sin_dt)
        final_rmsnorm_out = pk.new_tensor(
            dims=(pk.max_num_batched_tokens, hidden), dtype=_mi_bf16,
            name=f"{self.prefix}final_rmsnorm_out",
        )
        self.norm.compile(
            h_dt,
            output=final_rmsnorm_out,
            grid_dim=(pk.max_num_batched_tokens, 1, 1),
            block_dim=(128, 1, 1),
        )
        return final_rmsnorm_out


# ---------------------------------------------------------------------------
# DeepseekV3ForCausalLM
# ---------------------------------------------------------------------------


class DeepseekV3ForCausalLM(MPKModule):
    """Full DeepSeek V3 + lm_head + split-reduce argmax (greedy decode).

    Driver responsibilities (see ``demo/deepseek_v3/demo_new.py``):
      * Allocate the per-layer combined CKV/KPE cache pool and pass it via
        ``PersistentKernel(kv_cache=...)``.
      * Pre-pad ``lm_head.weight`` to a 256-multiple vocab.
      * Pass ``output_tokens`` torch tensor through ``model.compile()``.
      * Perform KV absorption + W_UV→o_proj fusion + expert stacking BEFORE
        ``load_state_dict()``.
    """

    def __init__(self, config, *, prefix: str = ""):
        super().__init__(prefix=prefix)
        self.config = config
        self.model = DeepseekV3Model(config, prefix=f"{prefix}model_")
        self.lm_head = Linear(
            config.hidden_size, config.vocab_size,
            prefix=f"{prefix}lm_head_",
        )
        # Greedy-decode head — split-reduce so the large vocab fans out.
        self.argmax_partial = ArgmaxPartial(
            vocab_size=config.vocab_size,
            num_partial_tasks=1,  # overwritten in compile()
            prefix=f"{prefix}argmax_partial_",
        )
        self.argmax_reduce = ArgmaxReduce(
            num_partial_tasks=1,
            prefix=f"{prefix}argmax_reduce_",
        )

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "DeepseekV3ForCausalLM.forward() not implemented in MPK catalog."
        )

    def auto_grid_dim(self, *args, **kwargs):
        raise NotImplementedError("composite — see child compile()s")

    def compile(self, input_tokens_dt, *, output_tokens=None,
                lm_head_padded_vocab: Optional[int] = None):
        pk = current_pk()
        h_dt = self.model.compile(input_tokens_dt)

        logits_dt = self.lm_head.compile(
            h_dt,
            grid_dim=(pk.num_workers, 1, 1),
            block_dim=(128, 1, 1),
        )

        self.argmax_partial.num_partial_tasks = pk.num_workers
        self.argmax_reduce.num_partial_tasks = pk.num_workers
        part_val_dt, part_idx_dt = self.argmax_partial.compile(
            logits_dt,
            grid_dim=(pk.num_workers, 1, 1),
            block_dim=(128, 1, 1),
        )
        return self.argmax_reduce.compile(
            part_val_dt, part_idx_dt,
            output=output_tokens,
            grid_dim=(1, 1, 1),
            block_dim=(128, 1, 1),
        )
