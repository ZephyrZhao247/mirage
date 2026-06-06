"""Test: DeepseekV3DecoderLayer.compile() vs a COMPOSED oracle.

Verifies one full decoder layer = input_layernorm -> MLA -> post_attention
_layernorm -> MLP, for both the dense (layer_idx < first_k_dense_replace) and
MoE (layer_idx >= first_k_dense_replace) MLP variants.

The layer's child modules are each already numerically verified bottom-up:
  * DeepseekV3MLA   -> test_dsv3_mla_testmode.py        (rel 0.0075)
  * DeepseekV3MLP   -> test_dsv3_mlp_fp8_testmode.py    (rel 0.039)
  * DeepseekV3MoE   -> test_dsv3_moe_testmode.py        (rel ~)
This test composes their validated oracles into the decoder-layer reference:

    h0          = RMSNorm(x,  input_layernorm.weight)
    attn_out    = MLA_oracle(h0) + x         # residual fused into o_proj
    h1          = RMSNorm(attn_out, post_attention_layernorm.weight)
    layer_out   = mlp.forward(h1, residual=attn_out)   # validated module fwd

Residuals are FUSED into the kernel (o_proj adds x; the dense MLP's down_proj
and the MoE's shared-expert path add attn_out), so the oracle adds each
residual exactly once, where the module does.

The MLA oracle reuses test_dsv3_mla_testmode.py's single-split SDPA (num_splits
=1, kv_len=128, mbt=mbr=1). RoPE uses interleaved (GPT-J) cos/sin tables so the
eager oracle and the interleaved MLA kernel agree.

hidden_size is forced to 7168 by the MLA dims (H=128, D_K=576, D_V=512).

Run:
  CUDA_VISIBLE_DEVICES=<free> python tests/runtime_python/test_mode/test_dsv3_decoder_layer_testmode.py
"""

import math
import os
import sys
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.models.deepseek_v3.modeling import DeepseekV3DecoderLayer
from mirage.mpk.layers.mla.rope import _rotate_interleaved

# Re-use the validated bottom-up helpers from the child tests.
from test_dsv3_mla_testmode import (
    _build_cos_sin, _deepseek_softmax_scale, _rmsnorm, _rope_apply,
    H, HIDDEN, Q_LORA, KV_LORA, D_PE, D_K, EPS,
)
from test_dsv3_mlp_fp8_testmode import _qfp8
from test_dsv3_moe_permute_chain_testmode import _pack_group_weight


def _rmsnorm_ref(x_f32, w_bf16, eps):
    """Token-wise RMSNorm matching layers.norm.RMSNorm.forward (fp32 reduce)."""
    var = x_f32.pow(2).mean(-1, keepdim=True)
    return x_f32 * torch.rsqrt(var + eps) * w_bf16.float()


def _mla_oracle(h0_bf16, residual_bf16, mla, prior_kv, cos, sin, step_pos):
    """Replay test_dsv3_mla_testmode.py's single-split SDPA on input h0.

    `mla` is the (weight-filled) DeepseekV3MLA module; `residual_bf16` is what
    the decoder feeds as residual_dt (the layer input x, NOT h0). Returns the
    o_proj+residual output (1, HIDDEN) in fp32.
    """
    ss = _deepseek_softmax_scale()
    cos_p = cos[step_pos].float()
    sin_p = sin[step_pos].float()
    Wqa = mla.q_a_proj_weight.float()
    Wqb = mla.q_b_proj_weight.float()
    Wkva = mla.kv_a_proj_with_mqa_weight.float()
    Wo = mla.o_proj_weight.float()
    q_a_ln = mla.q_a_layernorm
    kv_a_ln = mla.kv_a_layernorm

    h = h0_bf16[0].float()
    q_a = _rmsnorm(h @ Wqa.t(), q_a_ln)                       # (Q_LORA,)
    q = (q_a @ Wqb.t()).reshape(H, D_K)                       # (H, 576)
    q_pe = _rope_apply(q[:, KV_LORA:].clone(), cos_p, sin_p)  # (H, 64)
    q = torch.cat([q[:, :KV_LORA], q_pe], dim=-1)             # (H, 576)
    kv_a = h @ Wkva.t()                                       # (576,)
    c_latent = _rmsnorm(kv_a[:KV_LORA], kv_a_ln)              # (512,)
    k_pe = _rope_apply(kv_a[KV_LORA:], cos_p, sin_p)          # (64,)
    new_kv = torch.cat([c_latent, k_pe])                      # (576,)
    kv_slab = torch.cat([prior_kv.float(), new_kv[None, :]], dim=0)  # (128,576)
    scores = (q @ kv_slab.t()) * ss                          # (H,128)
    attn_w = torch.softmax(scores, dim=-1)
    attn = attn_w @ kv_slab[:, :KV_LORA]                      # (H,512)
    o = attn.reshape(-1) @ Wo.t() + residual_bf16[0].float()  # (HIDDEN,)
    return o[None, :]


def _run(layer_idx, intermediate_size, num_pages_for_cache, tol, label):
    device = "cuda"
    torch.manual_seed(0)

    kv_len = 128
    page_size = 128
    max_num_pages = num_pages_for_cache
    max_seq_length = 128
    step_pos = kv_len - 1
    eps = 1e-6

    # MoE-reduced fields (only used when layer is MoE); harmless for dense.
    cfg = SimpleNamespace(
        hidden_size=HIDDEN, num_attention_heads=H, q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA, qk_nope_head_dim=128, qk_rope_head_dim=D_PE,
        page_size=page_size, rms_norm_eps=eps, first_k_dense_replace=3,
        intermediate_size=intermediate_size,
        # MoE config (reduced like test_dsv3_moe_testmode.py). NB: topk=8 (not
        # 2) because the MoEPermute meta buffer is sized (2, m_total+mbt*topk)
        # and tensor_init requires that 1-D size be a multiple of 8 (16B vec).
        # With mbt=1 (MLA single-token decode) and m_total=16*128, topk must be
        # a multiple of 8 → topk=8 selects all 8 experts of the one chosen
        # group (n_group=2, topk_group=1 ⇒ EXPERTS_PER_GROUP=8).
        moe_intermediate_size=256, n_routed_experts=16,
        num_experts_per_tok=8, n_shared_experts=1, n_group=2, topk_group=1,
        routed_scaling_factor=2.5, norm_topk_prob=True,
    )

    # ---- RoPE tables: interleaved (GPT-J) so oracle/kernel agree -----------
    cos, sin = _build_cos_sin(max_seq_length, device, torch.bfloat16)

    # ---- KV cache pool (num_layers=max_num_pages? no: dim0==num_layers) -----
    # get_kv_cache(layer_idx) indexes dim-0 by layer; the pool needs at least
    # layer_idx+1 layers. We size it (layer_idx+1, 1, page_size, D_K).
    num_layers_cache = layer_idx + 1
    ckv_kpe_cache = torch.zeros(
        num_layers_cache, 1, page_size, D_K, dtype=torch.bfloat16, device=device)
    prior_kv = torch.randn(kv_len - 1, D_K, dtype=torch.bfloat16, device=device) * 0.1
    ckv_kpe_cache[layer_idx, 0, :kv_len - 1, :] = prior_kv

    # ---- Meta tensors (1 new decode token, seq_len=128) --------------------
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    paged_kv_indices = torch.tensor([0], dtype=torch.int32, device=device)
    paged_kv_last = torch.tensor([page_size], dtype=torch.int32, device=device)
    step = torch.tensor([step_pos], dtype=torch.int32, device=device)
    tokens = torch.zeros(1, max_seq_length, dtype=torch.int64, device=device)
    prompt_lengths = torch.tensor([kv_len], dtype=torch.int32, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True, num_workers=num_workers,
        num_local_schedulers=num_schedulers, mpi_rank=0, world_size=1,
        max_num_batched_tokens=1, max_num_batched_requests=1,
        max_seq_length=max_seq_length, max_num_pages=1, page_size=page_size,
        meta_tensors={
            "step": step, "tokens": tokens, "prompt_lengths": prompt_lengths,
            "qo_indptr_buffer": qo_indptr,
            "paged_kv_indptr_buffer": paged_kv_indptr,
            "paged_kv_indices_buffer": paged_kv_indices,
            "paged_kv_last_page_len_buffer": paged_kv_last,
        },
        kv_cache={"k_cache": ckv_kpe_cache, "v_cache": ckv_kpe_cache},
    )
    pk = PersistentKernel(**params)

    x = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device=device) * 0.5
    out = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=device)

    x_dt = pk.attach_input(x, name="x")
    cos_dt = pk.attach_input(cos, name="cos")
    sin_dt = pk.attach_input(sin, name="sin")

    # ---- Random weights ----------------------------------------------------
    Wqa = torch.randn(Q_LORA, HIDDEN, dtype=torch.bfloat16, device=device) * 0.02
    Wqb = torch.randn(H * D_K, Q_LORA, dtype=torch.bfloat16, device=device) * 0.02
    Wkva = torch.randn(KV_LORA + D_PE, HIDDEN, dtype=torch.bfloat16, device=device) * 0.02
    Wo = torch.randn(HIDDEN, H * KV_LORA, dtype=torch.bfloat16, device=device) * 0.02
    q_a_ln = torch.randn(Q_LORA, dtype=torch.bfloat16, device=device) * 0.1 + 1.0
    kv_a_ln = torch.randn(KV_LORA, dtype=torch.bfloat16, device=device) * 0.1 + 1.0
    in_ln = torch.randn(HIDDEN, dtype=torch.bfloat16, device=device) * 0.1 + 1.0
    post_ln = torch.randn(HIDDEN, dtype=torch.bfloat16, device=device) * 0.1 + 1.0

    with pk.compile_scope():
        layer = DeepseekV3DecoderLayer(cfg, layer_idx=layer_idx, prefix="L_").to(device)
        with torch.no_grad():
            # The catalog RMSNorm declares its weight as fp32 (torch.ones); the
            # rmsnorm_hopper kernel reads the scale as bf16. In a real run
            # model.to(bfloat16) casts it; here we set the .data to bf16
            # directly (a fp32 weight would be read 4-bytes-as-two-bf16 and
            # garble the norm). NB: do NOT layer.to(bfloat16) — that would also
            # cast the fp32 FP8 weight-scales the GEMM reads as fp32.
            layer.input_layernorm.weight.data = in_ln.clone()
            layer.post_attention_layernorm.weight.data = post_ln.clone()
            mla = layer.self_attn
            mla.q_a_proj_weight.copy_(Wqa)
            mla.q_b_proj_weight.copy_(Wqb)
            mla.kv_a_proj_with_mqa_weight.copy_(Wkva)
            mla.o_proj_weight.copy_(Wo)
            mla.q_a_layernorm.copy_(q_a_ln)
            mla.kv_a_layernorm.copy_(kv_a_ln)

            if not layer.is_moe:
                I = intermediate_size
                gate = torch.randn(I, HIDDEN, dtype=torch.bfloat16, device=device) * 0.05
                up = torch.randn(I, HIDDEN, dtype=torch.bfloat16, device=device) * 0.05
                down = torch.randn(HIDDEN, I, dtype=torch.bfloat16, device=device) * 0.05
                gu_fp8, gu_scale = _qfp8(torch.cat([gate, up], dim=0))
                dn_fp8, dn_scale = _qfp8(down)
                layer.mlp.gate_up_weight.copy_(gu_fp8)
                layer.mlp.gate_up_scale.copy_(gu_scale)
                layer.mlp.down_weight.copy_(dn_fp8)
                layer.mlp.down_scale.copy_(dn_scale)
            else:
                moe = layer.mlp
                I = cfg.moe_intermediate_size
                E = cfg.n_routed_experts
                w13 = torch.randn(E, 2 * I, HIDDEN, dtype=torch.bfloat16, device=device) * 0.05
                w2 = torch.randn(E, HIDDEN, I, dtype=torch.bfloat16, device=device) * 0.05
                w13_fp8, w13_scale, _ = _pack_group_weight(w13)
                w2_fp8, w2_scale, _ = _pack_group_weight(w2)
                sg = torch.randn(I, HIDDEN, dtype=torch.bfloat16, device=device) * 0.05
                su = torch.randn(I, HIDDEN, dtype=torch.bfloat16, device=device) * 0.05
                sd = torch.randn(HIDDEN, I, dtype=torch.bfloat16, device=device) * 0.05
                sgu_fp8, sgu_scale = _qfp8(torch.cat([sg, su], dim=0))
                sd_fp8, sd_scale = _qfp8(sd)
                gate_w = torch.randn(E, HIDDEN, dtype=torch.bfloat16, device=device) * 0.2
                moe.gate_weight.copy_(gate_w)
                moe.routing.bias.zero_()
                moe.experts_w13.weight.copy_(w13_fp8)
                moe.experts_w13.weight_scale.copy_(w13_scale)
                moe.experts_w2.weight.copy_(w2_fp8)
                moe.experts_w2.weight_scale.copy_(w2_scale)
                moe.shared_experts.gate_up_weight.copy_(sgu_fp8)
                moe.shared_experts.gate_up_scale.copy_(sgu_scale)
                moe.shared_experts.down_weight.copy_(sd_fp8)
                moe.shared_experts.down_scale.copy_(sd_scale)

        # ---- Oracle (composed from validated child oracles) ----------------
        h0 = _rmsnorm_ref(x.float(), in_ln, eps).to(torch.bfloat16)
        attn_out = _mla_oracle(h0, x, mla, prior_kv, cos, sin, step_pos)   # (1,H) f32
        attn_out_bf16 = attn_out.to(torch.bfloat16)
        h1 = _rmsnorm_ref(attn_out, post_ln, eps).to(torch.bfloat16)
        ref = layer.mlp.forward(h1, attn_out_bf16).float()                 # (1,H)

        # ---- Build the decoder-layer task graph ----------------------------
        layer_out_dt = layer.compile(x_dt, cos_dt, sin_dt)
        pk.identity_layer(
            input=layer_out_dt, output=pk.attach_input(out, name="out"),
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
        )

    print(f"[{label}] Compiling DeepseekV3DecoderLayer (layer_idx={layer_idx})...")
    pk.compile(output_dir=_HERE)
    pk()
    torch.cuda.synchronize()

    if out.isnan().any() or out.isinf().any():
        print(f"[{label}] FAILED: NaN/Inf (out[0,:6]={out[0,:6]})")
        pk.finalize()
        return False, float("nan")
    max_abs = (out.float() - ref).abs().max().item()
    max_rel = max_abs / max(ref.abs().max().item(), 1e-6)
    print(f"[{label}] out[0,:6]: {out[0,:6]}")
    print(f"[{label}] ref[0,:6]: {ref[0,:6]}")
    print(f"[{label}] max abs {max_abs:.5f} rel {max_rel:.5f} (tol {tol})")
    ok = max_rel < tol
    print(f"[{label}] {'PASSED' if ok else 'FAILED'}")
    pk.finalize()
    return ok, max_rel


def test_dense():
    ok, rel = _run(layer_idx=0, intermediate_size=512, num_pages_for_cache=1,
                   tol=0.10, label="dense")
    assert ok, f"dense decoder layer rel {rel} >= 0.10"


# The MoE routed permute path was previously blocked by a kernel defect at
# hidden>=2048 (moe_permute_sm100 read the input FP8 scale row-major instead of
# column-major). That .cuh bug is now FIXED, so the MoE decoder layer runs as a
# hard assertion at the production config (hidden=7168, layer_idx=3).
def test_moe():
    ok, rel = _run(layer_idx=3, intermediate_size=512, num_pages_for_cache=4,
                   tol=0.15, label="moe")
    assert ok, f"moe decoder layer rel {rel} >= 0.15"


if __name__ == "__main__":
    d_ok, d_rel = _run(layer_idx=0, intermediate_size=512, num_pages_for_cache=1,
                       tol=0.10, label="dense")
    m_ok, m_rel = _run(layer_idx=3, intermediate_size=512, num_pages_for_cache=4,
                       tol=0.15, label="moe")
    print(f"\n=== SUMMARY ===")
    print(f"dense: rel {d_rel:.5f} {'PASS' if d_ok else 'FAIL'}")
    print(f"moe:   rel {m_rel:.5f} {'PASS' if m_ok else 'FAIL'}")
    if not (d_ok and m_ok):
        sys.exit(1)
