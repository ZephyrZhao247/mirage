"""Precision comparison: MPK demo_new.py vs HuggingFace DeepSeek V3 reference.

Loads the HF DeepseekV3ForCausalLM with the SAME reduced config (a subset of
layers) and the SAME real weights for those layers, greedy-decodes the same
prompt, and diffs the generated token IDs against the MPK output dumped by
``demo_new.py --save-tokens``.

KNOWN CAVEAT (per task spec): the MPK model's RoPE may be GPT-J-interleaved
and/or lack YARN long-rope scaling, while HF uses the YARN cat-convention. If
the tokens diverge, this script reports the first divergence index so the
non-RoPE alignment can be assessed separately.

Usage::

    python compare_hf_precision.py \
        --model-path /mnt/shared/models/DeepSeek-V3 \
        --layers 0-3 --prompt "The capital of France is" \
        --mpk-tokens /tmp/mpk_dsv3_tokens.json --max-new-tokens 32
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def _parse_layers(spec: str):
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
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--mpk-tokens", default=None,
                    help="JSON dumped by demo_new.py --save-tokens")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    torch.set_default_dtype(torch.bfloat16)
    layer_indices = _parse_layers(args.layers)
    num_layers = max(layer_indices) + 1

    # Use the MAINTAINED transformers DeepSeek V3 implementation (NOT the
    # checkpoint's bundled remote code, whose cache uses a stale
    # past_key_values.seen_tokens API incompatible with the installed
    # transformers DynamicCache).
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=False)
    config.num_hidden_layers = num_layers
    config.architectures = ["DeepseekV3ForCausalLM"]
    config.auto_map = {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_path,
                                              trust_remote_code=True)

    print(f"[hf] building reduced HF model: num_hidden_layers={num_layers}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, config=config, trust_remote_code=False,
        dtype=torch.bfloat16, device_map="cuda",
        low_cpu_mem_usage=True,
    )
    model.eval()

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")
    prompt_len = inputs.input_ids.shape[-1]

    with torch.no_grad():
        gen = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens,
            do_sample=False, num_beams=1,
            pad_token_id=tokenizer.eos_token_id
            if isinstance(tokenizer.eos_token_id, int) else 0,
        )
    hf_new = gen[0, prompt_len:].tolist()
    print(f"[hf] prompt_len={prompt_len}, generated {len(hf_new)} tokens")
    print(f"[hf] first 20 new token ids: {hf_new[:20]}")
    print(f"[hf] decoded: {tokenizer.decode(hf_new, skip_special_tokens=True)!r}")

    if args.mpk_tokens:
        with open(args.mpk_tokens) as f:
            mpk = json.load(f)
        mpk_new = mpk["token_ids"]
        print(f"[mpk] first 20 new token ids: {mpk_new[:20]}")
        n = min(len(hf_new), len(mpk_new))
        first_div = None
        match = 0
        for i in range(n):
            if hf_new[i] == mpk_new[i]:
                match += 1
            elif first_div is None:
                first_div = i
        print(f"[diff] compared {n} tokens: {match} match "
              f"({100.0 * match / max(n, 1):.1f}%)")
        if first_div is None:
            print("[diff] FULL TOKEN MATCH over the compared range.")
        else:
            print(f"[diff] first divergence at index {first_div}: "
                  f"hf={hf_new[first_div]} vs mpk={mpk_new[first_div]}")


if __name__ == "__main__":
    main()
