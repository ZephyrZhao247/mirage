"""Layer-by-layer MPK-vs-HF hidden-state comparison for new-API DeepSeek V3.

Isolates the RoPE/YARN correctness of the new-API model from the prefix/decode
loop by running a SINGLE token through both stacks where the token attends only
to itself (kv_len=1), so both sides see identical context:

  * MPK: build DeepseekV3Model (N layers) with the SAME real HF weights as
    demo_new.py (KV absorption + W_UV fusion), run a single decode step at
    absolute position ``--position`` with a 1-row KV cache. Read the
    pre-lm_head hidden state (final RMSNorm output).
  * HF: transformers DeepseekV3Model, feed the same single token id with
    position_ids=[[position]] and an empty past, read last_hidden_state[0,0].

At ``--position 0`` RoPE is the identity (cos=1, sin=0), so a match there proves
the NON-RoPE math (embed, MLA absorption, FP8 MLP/MoE, residuals, norms) is
aligned. At a non-zero position the comparison additionally exercises RoPE/YARN.

Run on a GPU node::

  CUDA_VISIBLE_DEVICES=0 python compare_hf_layerwise.py \
      --model-path /mnt/shared/models/DeepSeek-V3 --layers 0-3 \
      --position 0 --token-id 1538
"""
from __future__ import annotations

import argparse
import sys

import torch

import mirage as mi
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.models.deepseek_v3.modeling import DeepseekV3Model

import demo_new as D
from transformers import AutoConfig
from transformers.models.deepseek_v3.modeling_deepseek_v3 import DeepseekV3Model as HFModel


def _parse_layers(spec):
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--layers", default="0-3")
    ap.add_argument("--position", type=int, default=0)
    ap.add_argument("--token-id", type=int, default=1538)
    ap.add_argument("--max-seq-length", type=int, default=256,
                    help="128 => num_splits=1 (WRITE_FINAL); 256 => split-K + reduce")
    ap.add_argument("--embed-only", action="store_true",
                    help="compare only the embedding output (no decoder layers)")
    ap.add_argument("--mla-only", action="store_true",
                    help="compare embed -> input_norm -> MLA(+residual) for layer 0 "
                         "(post-attention hidden, before MLP)")
    ap.add_argument("--mlp-only", action="store_true",
                    help="compare layer-0 dense MLP output (MPK FP8 vs HF) on a "
                         "fixed random input, no residual")
    args = ap.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    torch.cuda.set_device(0)
    layer_indices = _parse_layers(args.layers)
    num_layers = max(layer_indices) + 1

    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=False)
    cfg.num_hidden_layers = num_layers
    cfg._attn_implementation = "eager"

    HIDDEN = cfg.hidden_size
    D_K = cfg.kv_lora_rank + cfg.qk_rope_head_dim
    page_size = 128
    max_seq_length = args.max_seq_length
    kv_len = 1
    pos = args.position
    tok = args.token_id

    # ---------------- MPK side ----------------
    num_workers, num_sched = mi.get_configurations_from_gpu(0)
    ckv = torch.zeros(num_layers, 1, page_size, D_K, dtype=torch.bfloat16, device="cuda")

    # 1 decode token at absolute position `pos`; it attends only to itself, so
    # kv_len=1 with the cache row at index `pos % page_size`.
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    paged_kv_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    # last_page_len = number of valid rows in the (single) page = pos+1
    paged_kv_last = torch.tensor([pos + 1], dtype=torch.int32, device="cuda")
    step = torch.tensor([pos], dtype=torch.int32, device="cuda")
    tokens = torch.zeros(1, max_seq_length, dtype=torch.int64, device="cuda")
    prompt_lengths = torch.tensor([pos + 1], dtype=torch.int32, device="cuda")

    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True, num_workers=num_workers, num_local_schedulers=num_sched,
        mpi_rank=0, world_size=1, max_num_batched_tokens=1,
        max_num_batched_requests=1, max_seq_length=max_seq_length,
        max_num_pages=1, page_size=page_size,
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

    input_tokens = torch.tensor([[tok]], dtype=torch.int64, device="cuda")
    tok_dt = pk.attach_input(input_tokens, name="input_tokens")
    out = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device="cuda")

    with pk.compile_scope():
        with torch.device("cuda"):
            model = DeepseekV3Model(cfg, prefix="").to("cuda", dtype=torch.bfloat16)
        sd = D._load_hf_weights_with_absorption(args.model_path, cfg, layer_indices)
        # strip the "model." prefix produced by the loader (this builds the bare
        # DeepseekV3Model whose params have no leading "model.").
        sd = {k[len("model."):] if k.startswith("model.") else k: v
              for k, v in sd.items() if not k.startswith("lm_head")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"[mpk] load: {len(unexpected)} unexpected keys (sample {unexpected[:4]})")
        if args.embed_only:
            # Compile ONLY the embedding (token -> hidden); no decoder layers.
            model.embed_tokens.compile(
                tok_dt, input_source=1,
                output=pk.attach_input(out, name="out"),
                grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
            )
        elif args.mlp_only:
            # Feed a fixed random input through layer-0 dense MLP (residual=0).
            from mirage.core import bfloat16 as _bf16
            torch.manual_seed(1234)
            mlp_in = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1
            mlp_in_dt = pk.attach_input(mlp_in, name="mlp_in")
            zero_res = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device="cuda")
            zero_res_dt = pk.attach_input(zero_res, name="zero_res")
            model.layers[0].mlp.compile(
                mlp_in_dt, residual_dt=zero_res_dt,
                output=pk.attach_input(out, name="out"))
            globals()["_MLP_IN"] = mlp_in  # stash for HF side
        elif args.mla_only:
            # embed -> input_norm -> MLA(+residual fused into o_proj); read the
            # post-attention hidden (before the MLP block).
            from mirage.core import bfloat16 as _bf16
            layer0 = model.layers[0]
            embed_out = pk.new_tensor(dims=(1, HIDDEN), dtype=_bf16, name="emb")
            model.embed_tokens.compile(
                tok_dt, input_source=1, output=embed_out,
                grid_dim=(1, 1, 1), block_dim=(128, 1, 1))
            rms_attn = pk.new_tensor(dims=(1, HIDDEN), dtype=_bf16, name="rms_attn")
            layer0.input_layernorm.compile(
                embed_out, output=rms_attn,
                grid_dim=(1, 1, 1), block_dim=(128, 1, 1))
            cos_dt, sin_dt = model.rotary_emb.compile()
            layer0.self_attn.compile(
                rms_attn, cos_dt, sin_dt, residual_dt=embed_out,
                output=pk.attach_input(out, name="out"))
        else:
            h_dt = model.compile(tok_dt)
            pk.identity_layer(
                input=h_dt, output=pk.attach_input(out, name="out"),
                grid_dim=(1, 1, 1), block_dim=(128, 1, 1),
            )
    print("[mpk] compiling megakernel...")
    pk.compile()
    pk()
    torch.cuda.synchronize()
    mpk_h = out[0].float().clone()
    pk.finalize()
    print(f"[mpk] hidden norm={mpk_h.norm().item():.4f}  [:6]={mpk_h[:6]}")

    # ---------------- HF side ----------------
    sd_full = D._selectively_load_layers(args.model_path, layer_indices)
    # dequant HF weights to bf16 for an eager HF model (use the maintained impl).
    hf = HFModel(cfg).to("cuda", dtype=torch.bfloat16).eval()
    hf_sd = {}
    for k, v in sd_full.items():
        if k.startswith("model."):
            kk = k[len("model."):]
        else:
            kk = k
        if k.endswith("weight_scale_inv"):
            continue
        if D.is_fp8(v):
            v = D._maybe_dequant(k, v, sd_full)
        hf_sd[kk] = v.to(torch.bfloat16)
    miss, unexp = hf.load_state_dict(hf_sd, strict=False)
    print(f"[hf] load: {len(miss)} missing, {len(unexp)} unexpected "
          f"(missing sample {miss[:4]}, unexpected sample {unexp[:4]})")

    input_ids = torch.tensor([[tok]], dtype=torch.long, device="cuda")
    if args.mlp_only:
        with torch.no_grad():
            hf_h = hf.layers[0].mlp(globals()["_MLP_IN"])[0].float()
    elif args.embed_only:
        with torch.no_grad():
            hf_h = hf.embed_tokens(input_ids)[0, 0].float()
    elif args.mla_only:
        # Capture (residual + attn_out) = the input to post_attention_layernorm.
        captured = {}
        h = hf.layers[0].post_attention_layernorm.register_forward_pre_hook(
            lambda m, inp: captured.setdefault("x", inp[0].detach()))
        position_ids = torch.tensor([[pos]], dtype=torch.long, device="cuda")
        with torch.no_grad():
            hf(input_ids=input_ids, position_ids=position_ids, use_cache=False)
        h.remove()
        hf_h = captured["x"][0, 0].float()
    else:
        position_ids = torch.tensor([[pos]], dtype=torch.long, device="cuda")
        with torch.no_grad():
            hf_out = hf(input_ids=input_ids, position_ids=position_ids,
                        use_cache=False, output_hidden_states=False)
        hf_h = hf_out.last_hidden_state[0, 0].float()
    print(f"[hf]  hidden norm={hf_h.norm().item():.4f}  [:6]={hf_h[:6]}")

    diff = (mpk_h - hf_h).abs()
    rel = diff.max().item() / max(hf_h.abs().max().item(), 1e-6)
    cos = torch.nn.functional.cosine_similarity(mpk_h, hf_h, dim=0).item()
    print(f"[diff] pos={pos}: max_abs={diff.max().item():.5f} "
          f"rel={rel:.5f} cosine={cos:.6f}")


if __name__ == "__main__":
    main()
