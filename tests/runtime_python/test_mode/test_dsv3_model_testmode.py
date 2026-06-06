"""Test: DeepseekV3Model.compile() (4 layers) vs a COMPOSED oracle.

Verifies the model-level plumbing of the new-API DeepSeek V3:

    Embed(tokens) -> 4 x DeepseekV3DecoderLayer -> final RMSNorm

against an oracle that composes the already-validated per-block oracles
(embed lookup + 4 x [input_norm -> MLA -> post_norm -> MLP] + final norm).
Each layer's MLA oracle is the single-split SDPA proven by
test_dsv3_mla_testmode.py; the per-layer composition matches
test_dsv3_decoder_layer_testmode.py.

Config: real DSv3 attention dims (hidden=7168, H=128, D_K=576), but only 4
hidden layers and reduced MLP/MoE widths (mirroring the demo's layer
reduction). mbt=mbr=1, kv_len=128. A 4-layer ckv_kpe_cache pool is allocated;
each layer appends/gathers from its own slice (layer_idx).

RoPE caveat (per task spec): DeepseekV3Model builds plain HF cat-convention
RotaryEmbedding cos/sin tables, but the MLA kernel applies GPT-J *interleaved*
RoPE. To keep this test about plumbing/shape (not YARN numerics), we replace
the model's rotary cos/sin buffers with INTERLEAVED tables (the same
_build_cos_sin the MLA test uses) so the kernel and the oracle agree. Real
YARN numerics are a later end-to-end milestone.

Two variants:
  * first_k_dense_replace=4 (ALL DENSE): PASSES — verifies the multi-layer
    pipeline, embed, 4-layer KV cache pool, and final norm end to end.
  * first_k_dense_replace=3 (layer 3 = MoE): PASSES — exercises the routed
    MoE path (moe_permute_sm100 column-major scale defect at hidden>=2048 now
    fixed) composed into the full multi-layer pipeline. Run via
    _run_model(first_moe=3).

Run:
  CUDA_VISIBLE_DEVICES=<free> python tests/runtime_python/test_mode/test_dsv3_model_testmode.py
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
from mirage.mpk.models.deepseek_v3.modeling import DeepseekV3Model

from test_dsv3_mla_testmode import (
    _build_cos_sin, _deepseek_softmax_scale, _rmsnorm, _rope_apply,
    H, HIDDEN, Q_LORA, KV_LORA, D_PE, D_K,
)
from test_dsv3_mlp_fp8_testmode import _qfp8
from test_dsv3_moe_permute_chain_testmode import _pack_group_weight

VOCAB = 1024            # tiny padded vocab (multiple of 256)


def _rmsnorm_ref(x_f32, w_bf16, eps):
    var = x_f32.pow(2).mean(-1, keepdim=True)
    return x_f32 * torch.rsqrt(var + eps) * w_bf16.float()


def _mla_layer_oracle(h0_bf16, residual_bf16, mw, prior_kv, cos, sin, step_pos):
    """Single-split SDPA MLA oracle (see test_dsv3_mla_testmode.py).

    mw is a dict of this layer's MLA weights (fp32 / bf16 tensors). Returns
    o_proj(attn) + residual_bf16 in fp32, shape (1, HIDDEN).
    """
    ss = _deepseek_softmax_scale()
    cos_p = cos[step_pos].float()
    sin_p = sin[step_pos].float()
    Wqa, Wqb = mw["Wqa"].float(), mw["Wqb"].float()
    Wkva, Wo = mw["Wkva"].float(), mw["Wo"].float()
    h = h0_bf16[0].float()
    q_a = _rmsnorm(h @ Wqa.t(), mw["q_a_ln"])
    q = (q_a @ Wqb.t()).reshape(H, D_K)
    q_pe = _rope_apply(q[:, KV_LORA:].clone(), cos_p, sin_p)
    q = torch.cat([q[:, :KV_LORA], q_pe], dim=-1)
    kv_a = h @ Wkva.t()
    c_latent = _rmsnorm(kv_a[:KV_LORA], mw["kv_a_ln"])
    k_pe = _rope_apply(kv_a[KV_LORA:], cos_p, sin_p)
    new_kv = torch.cat([c_latent, k_pe])
    kv_slab = torch.cat([prior_kv.float(), new_kv[None, :]], dim=0)
    scores = (q @ kv_slab.t()) * ss
    attn = torch.softmax(scores, dim=-1) @ kv_slab[:, :KV_LORA]
    o = attn.reshape(-1) @ Wo.t() + residual_bf16[0].float()
    return o[None, :]


def _make_mla_weights(device):
    """Random MLA weights (same scale as test_dsv3_mla_testmode.py)."""
    return dict(
        Wqa=torch.randn(Q_LORA, HIDDEN, dtype=torch.bfloat16, device=device) * 0.02,
        Wqb=torch.randn(H * D_K, Q_LORA, dtype=torch.bfloat16, device=device) * 0.02,
        Wkva=torch.randn(KV_LORA + D_PE, HIDDEN, dtype=torch.bfloat16, device=device) * 0.02,
        Wo=torch.randn(HIDDEN, H * KV_LORA, dtype=torch.bfloat16, device=device) * 0.02,
        q_a_ln=torch.randn(Q_LORA, dtype=torch.bfloat16, device=device) * 0.1 + 1.0,
        kv_a_ln=torch.randn(KV_LORA, dtype=torch.bfloat16, device=device) * 0.1 + 1.0,
    )


def _run_model(first_moe, num_layers=4, dense_I=512, tol=0.15, label="model"):
    device = "cuda"
    torch.manual_seed(0)

    kv_len = 128
    page_size = 128
    max_seq_length = 128
    step_pos = kv_len - 1
    eps = 1e-6
    moe_I = 256
    moe_E = 16

    cfg = SimpleNamespace(
        vocab_size=VOCAB, hidden_size=HIDDEN, num_hidden_layers=num_layers,
        num_attention_heads=H, q_lora_rank=Q_LORA, kv_lora_rank=KV_LORA,
        qk_nope_head_dim=128, qk_rope_head_dim=D_PE, page_size=page_size,
        rms_norm_eps=eps, first_k_dense_replace=first_moe,
        intermediate_size=dense_I,
        max_position_embeddings=4096, rope_theta=10000.0,
        # MoE (reduced); topk=8 so the permute meta size (m_total+mbt*topk) is
        # a multiple of 8 at mbt=1 (m_total=16*128).
        moe_intermediate_size=moe_I, n_routed_experts=moe_E,
        num_experts_per_tok=8, n_shared_experts=1, n_group=2, topk_group=1,
        routed_scaling_factor=2.5, norm_topk_prob=True,
    )

    # ---- interleaved RoPE tables (kernel == GPT-J interleaved) -------------
    cos, sin = _build_cos_sin(max_seq_length, device, torch.bfloat16)

    # ---- 4-layer KV cache pool; seed each layer's prior context ------------
    ckv = torch.zeros(num_layers, 1, page_size, D_K, dtype=torch.bfloat16, device=device)
    prior_kv = [
        torch.randn(kv_len - 1, D_K, dtype=torch.bfloat16, device=device) * 0.1
        for _ in range(num_layers)
    ]
    for li in range(num_layers):
        ckv[li, 0, :kv_len - 1, :] = prior_kv[li]

    # ---- meta tensors (1 decode token, seq_len=128) -----------------------
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
        kv_cache={"k_cache": ckv, "v_cache": ckv},
    )
    pk = PersistentKernel(**params)

    # ---- model inputs ------------------------------------------------------
    token_id = 7
    input_tokens = torch.tensor([[token_id]], dtype=torch.int64, device=device)
    out = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=device)
    tok_dt = pk.attach_input(input_tokens, name="input_tokens")

    # ---- random weights ----------------------------------------------------
    embed_w = torch.randn(VOCAB, HIDDEN, dtype=torch.bfloat16, device=device) * 0.05
    final_ln = torch.randn(HIDDEN, dtype=torch.bfloat16, device=device) * 0.1 + 1.0
    layer_w = []  # per-layer dict
    for li in range(num_layers):
        is_moe = li >= first_moe
        lw = dict(
            in_ln=torch.randn(HIDDEN, dtype=torch.bfloat16, device=device) * 0.1 + 1.0,
            post_ln=torch.randn(HIDDEN, dtype=torch.bfloat16, device=device) * 0.1 + 1.0,
            mla=_make_mla_weights(device), is_moe=is_moe,
        )
        if not is_moe:
            I = dense_I
            g = torch.randn(I, HIDDEN, dtype=torch.bfloat16, device=device) * 0.05
            u = torch.randn(I, HIDDEN, dtype=torch.bfloat16, device=device) * 0.05
            d = torch.randn(HIDDEN, I, dtype=torch.bfloat16, device=device) * 0.05
            lw["gu_fp8"], lw["gu_scale"] = _qfp8(torch.cat([g, u], dim=0))
            lw["dn_fp8"], lw["dn_scale"] = _qfp8(d)
        else:
            I, E = moe_I, moe_E
            w13 = torch.randn(E, 2 * I, HIDDEN, dtype=torch.bfloat16, device=device) * 0.05
            w2 = torch.randn(E, HIDDEN, I, dtype=torch.bfloat16, device=device) * 0.05
            lw["w13_fp8"], lw["w13_scale"], _ = _pack_group_weight(w13)
            lw["w2_fp8"], lw["w2_scale"], _ = _pack_group_weight(w2)
            sg = torch.randn(I, HIDDEN, dtype=torch.bfloat16, device=device) * 0.05
            su = torch.randn(I, HIDDEN, dtype=torch.bfloat16, device=device) * 0.05
            sd = torch.randn(HIDDEN, I, dtype=torch.bfloat16, device=device) * 0.05
            lw["sgu_fp8"], lw["sgu_scale"] = _qfp8(torch.cat([sg, su], dim=0))
            lw["sd_fp8"], lw["sd_scale"] = _qfp8(sd)
            lw["gate_w"] = torch.randn(E, HIDDEN, dtype=torch.bfloat16, device=device) * 0.2
        layer_w.append(lw)

    with pk.compile_scope():
        model = DeepseekV3Model(cfg, prefix="M_").to(device)
        # Replace cat-convention rotary tables with interleaved ones (see
        # RoPE caveat in the docstring) so kernel == oracle.
        model.rotary_emb.cos = cos
        model.rotary_emb.sin = sin
        with torch.no_grad():
            model.embed_tokens.weight.data = embed_w.clone()
            model.norm.weight.data = final_ln.clone()
            for li, layer in enumerate(model.layers):
                lw = layer_w[li]
                layer.input_layernorm.weight.data = lw["in_ln"].clone()
                layer.post_attention_layernorm.weight.data = lw["post_ln"].clone()
                mla = layer.self_attn
                mla.q_a_proj_weight.copy_(lw["mla"]["Wqa"])
                mla.q_b_proj_weight.copy_(lw["mla"]["Wqb"])
                mla.kv_a_proj_with_mqa_weight.copy_(lw["mla"]["Wkva"])
                mla.o_proj_weight.copy_(lw["mla"]["Wo"])
                mla.q_a_layernorm.copy_(lw["mla"]["q_a_ln"])
                mla.kv_a_layernorm.copy_(lw["mla"]["kv_a_ln"])
                if not lw["is_moe"]:
                    layer.mlp.gate_up_weight.copy_(lw["gu_fp8"])
                    layer.mlp.gate_up_scale.copy_(lw["gu_scale"])
                    layer.mlp.down_weight.copy_(lw["dn_fp8"])
                    layer.mlp.down_scale.copy_(lw["dn_scale"])
                else:
                    moe = layer.mlp
                    moe.gate_weight.copy_(lw["gate_w"])
                    moe.routing.bias.zero_()
                    moe.experts_w13.weight.copy_(lw["w13_fp8"])
                    moe.experts_w13.weight_scale.copy_(lw["w13_scale"])
                    moe.experts_w2.weight.copy_(lw["w2_fp8"])
                    moe.experts_w2.weight_scale.copy_(lw["w2_scale"])
                    moe.shared_experts.gate_up_weight.copy_(lw["sgu_fp8"])
                    moe.shared_experts.gate_up_scale.copy_(lw["sgu_scale"])
                    moe.shared_experts.down_weight.copy_(lw["sd_fp8"])
                    moe.shared_experts.down_scale.copy_(lw["sd_scale"])

        # ---- Oracle ------------------------------------------------------
        h = embed_w[token_id].float()[None, :]           # (1, HIDDEN) f32
        for li, layer in enumerate(model.layers):
            lw = layer_w[li]
            h_bf = h.to(torch.bfloat16)
            h0 = _rmsnorm_ref(h, lw["in_ln"], eps).to(torch.bfloat16)
            attn = _mla_layer_oracle(h0, h_bf, lw["mla"], prior_kv[li],
                                     cos, sin, step_pos)             # (1,H) f32
            attn_bf = attn.to(torch.bfloat16)
            h1 = _rmsnorm_ref(attn, lw["post_ln"], eps).to(torch.bfloat16)
            mlp_out = layer.mlp.forward(h1, attn_bf).float()         # validated fwd
            h = mlp_out
        ref = _rmsnorm_ref(h, final_ln, eps)                         # (1, HIDDEN)

        # ---- Build the model task graph ----------------------------------
        h_dt = model.compile(tok_dt)
        pk.identity_layer(
            input=h_dt, output=pk.attach_input(out, name="out"),
            grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
        )

    print(f"[{label}] Compiling DeepseekV3Model ({num_layers} layers, "
          f"first_moe={first_moe})...")
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


def test_model_dense():
    """4-layer model, ALL layers dense (first_k_dense_replace=4). Verifies the
    embed -> 4 decoder layers -> final-norm pipeline + the 4-layer KV cache."""
    ok, rel = _run_model(first_moe=4, tol=0.15, label="model-dense")
    assert ok, f"4-layer dense model rel {rel} >= 0.15"


# The spec'd variant (layers 0-2 dense + layer 3 MoE) was previously blocked by
# the moe_permute_sm100 column-major scale defect at hidden>=2048. That .cuh bug
# is now FIXED, so the layer-3-MoE 4-layer model runs as a hard assertion.
def test_model_with_moe():
    ok, rel = _run_model(first_moe=3, tol=0.15, label="model-moe")
    assert ok, f"4-layer model with layer-3 MoE rel {rel} >= 0.15"


if __name__ == "__main__":
    d_ok, d_rel = _run_model(first_moe=4, tol=0.15, label="model-dense")
    print("\n--- 4-layer model with layer-3 MoE ---")
    m_ok, m_rel = _run_model(first_moe=3, tol=0.15, label="model-moe")
    print(f"\n=== SUMMARY ===")
    print(f"model-dense (4L all-dense): rel {d_rel:.5f} {'PASS' if d_ok else 'FAIL'}")
    print(f"model-moe   (L3 MoE):       rel {m_rel:.5f} {'PASS' if m_ok else 'FAIL'}")
    if not (d_ok and m_ok):
        sys.exit(1)
