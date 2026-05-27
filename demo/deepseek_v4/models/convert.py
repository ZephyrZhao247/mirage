"""Weight conversion for DeepSeek V4-Flash from the official checkpoint.

For Wave-3 (Layer 0 only), this module:

  1. Reads the safetensors index at
     ``/raid/catalyst/models/DeepSeek-V4-Flash-Base/model.safetensors.index.json``.
  2. Identifies which shards contain ``layers.0.*`` keys (Layer 0 lives
     entirely in shard 2 of 46).
  3. Loads only the needed shards with ``safetensors.load_file()``
     (~6 GiB instead of the full 275 GiB).
  4. Maps the HF ``layers.0.<name>`` keys to attributes on
     :class:`DeepseekV4Block` per the convention documented inline.

The mapping is deliberately a flat dict from
``hf_suffix -> mpk_attr_name`` so it is easy to audit. Any
mismatch / missing key is reported via stdout.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Iterable, List, Set, Tuple

import torch

try:
    from safetensors import safe_open
    from safetensors.torch import load_file
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "safetensors required: pip install safetensors"
    ) from exc


# --------------------------------------------------------------------- #
# HF key suffix -> (MPK attribute name on DeepseekV4Block, transform)
# --------------------------------------------------------------------- #
#
# ``transform`` is one of:
#   "as_is"   : ``param.data.copy_(tensor)`` after a cast to the param's dtype.
#   "i64_to_i32": ``tensor.to(torch.int32)`` then copy_.
#   "f8_keep" : keep as float8_e4m3fn (target param dtype matches).
#
# Some FP8 scales arrive on disk as fp32 with shape ``[M/128, K/128]``; we
# keep them as-is.  Some checkpoints (vLLM) use UE8M0 packed scales; the
# Flash-Base checkpoint stores fp32 (verified by `safe_open` inspection
# on shard 00002).
#
_HF_TO_MPK_LAYER0 = {
    # HC params
    "hc_attn_fn": ("hc_attn_fn", "as_is"),
    "hc_ffn_fn": ("hc_ffn_fn", "as_is"),
    "hc_attn_base": ("hc_attn_base", "as_is"),
    "hc_ffn_base": ("hc_ffn_base", "as_is"),
    "hc_attn_scale": ("hc_attn_scale", "as_is"),
    "hc_ffn_scale": ("hc_ffn_scale", "as_is"),
    # RMSNorms
    "attn_norm.weight": ("attn_norm_weight", "as_is"),
    "ffn_norm.weight": ("ffn_norm_weight", "as_is"),
    # MLA attn
    "attn.attn_sink": ("attn_sink", "as_is"),
    "attn.q_norm.weight": ("attn_q_norm_weight", "as_is"),
    "attn.kv_norm.weight": ("attn_kv_norm_weight", "as_is"),
    "attn.wq_a.weight": ("wq_a_weight", "f8_keep"),
    "attn.wq_a.scale": ("wq_a_scale", "as_is"),
    "attn.wq_b.weight": ("wq_b_weight", "f8_keep"),
    "attn.wq_b.scale": ("wq_b_scale", "as_is"),
    "attn.wkv.weight": ("wkv_weight", "f8_keep"),
    "attn.wkv.scale": ("wkv_scale", "as_is"),
    "attn.wo_a.weight": ("wo_a_weight", "f8_keep"),
    "attn.wo_a.scale": ("wo_a_scale", "as_is"),
    "attn.wo_b.weight": ("wo_b_weight", "f8_keep"),
    "attn.wo_b.scale": ("wo_b_scale", "as_is"),
    # Gate (router)
    "ffn.gate.weight": ("gate_weight", "as_is"),
    # hash routing: I64 on disk → I32 on the MPK side (matches the
    # HashRouteLookup kernel's int32 input).
    "ffn.gate.tid2eid": ("tid2eid", "i64_to_i32"),
    # gate.bias only exists for layer_id >= num_hash_layers; for Layer 0
    # there is no such key in the checkpoint.
}


def _resolve_layer0_shards(index_path: str) -> List[str]:
    """Return absolute paths of the safetensors shards that contain
    ``layers.0.*`` keys."""
    with open(index_path) as f:
        idx = json.load(f)
    weight_map = idx["weight_map"]
    shards: Set[str] = set()
    for k, v in weight_map.items():
        if k.startswith("layers.0.") or k.startswith("model.layers.0."):
            shards.add(v)
    base = os.path.dirname(index_path)
    return [os.path.join(base, s) for s in sorted(shards)]


def _load_layer0_tensors(checkpoint_dir: str) -> Dict[str, torch.Tensor]:
    """Load every ``layers.0.*`` tensor from the relevant shards."""
    index_path = os.path.join(
        checkpoint_dir, "model.safetensors.index.json"
    )
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Missing safetensors index at {index_path}"
        )
    shards = _resolve_layer0_shards(index_path)
    if not shards:
        raise RuntimeError(
            "No safetensors shard contains keys with prefix 'layers.0.'. "
            "Has the index been regenerated?"
        )
    out: Dict[str, torch.Tensor] = {}
    for shard in shards:
        # ``load_file`` would load the entire shard; that's still 6 GiB
        # for shard 2 of Flash-Base which is fine for a one-off test.
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                # Accept both 'layers.0.X' and 'model.layers.0.X' prefixes;
                # the Flash-Base checkpoint uses the former.
                if key.startswith("layers.0.") or key.startswith(
                    "model.layers.0."
                ):
                    suffix = key.split(".", 2)[2] if key.startswith(
                        "model.layers.0."
                    ) else key[len("layers.0."):]
                    out[suffix] = f.get_tensor(key)
    return out


def load_layer0_weights(
    checkpoint_dir: str,
    layer,
    *,
    verbose: bool = True,
) -> Tuple[Set[str], Set[str]]:
    """Populate ``layer``'s parameters from a Flash-Base checkpoint.

    Returns ``(consumed_hf_suffixes, unmapped_hf_suffixes)``.
    """
    tensors = _load_layer0_tensors(checkpoint_dir)

    # ---- Block-level non-expert tensors ----
    consumed: Set[str] = set()
    unmapped: Set[str] = set()

    for hf_suffix, tensor in tensors.items():
        # Expert tensors handled separately below.
        if hf_suffix.startswith("ffn.experts.") or hf_suffix.startswith(
            "ffn.shared_experts."
        ):
            continue
        if hf_suffix not in _HF_TO_MPK_LAYER0:
            unmapped.add(hf_suffix)
            continue
        attr_name, transform = _HF_TO_MPK_LAYER0[hf_suffix]
        param = getattr(layer, attr_name, None)
        if param is None:
            unmapped.add(hf_suffix)
            continue
        try:
            if transform == "as_is":
                target_dtype = param.dtype
                src = tensor.to(target_dtype)
                if src.shape != param.shape:
                    if verbose:
                        print(
                            f"  WARN shape mismatch {hf_suffix}: "
                            f"ckpt {tuple(src.shape)} vs param "
                            f"{tuple(param.shape)} — skipping"
                        )
                    unmapped.add(hf_suffix)
                    continue
                param.data.copy_(src)
            elif transform == "i64_to_i32":
                src = tensor.to(torch.int32)
                if src.shape != param.shape:
                    if verbose:
                        print(
                            f"  WARN tid2eid shape mismatch: "
                            f"ckpt {tuple(src.shape)} vs param "
                            f"{tuple(param.shape)}"
                        )
                    unmapped.add(hf_suffix)
                    continue
                param.data.copy_(src)
            elif transform == "f8_keep":
                # FP8 tensor: bit-cast preserved via view-as-int8-then-back.
                if param.dtype != torch.float8_e4m3fn:
                    raise ValueError(
                        f"{attr_name} must be float8_e4m3fn target dtype"
                    )
                if tensor.dtype != torch.float8_e4m3fn:
                    if verbose:
                        print(
                            f"  WARN {hf_suffix}: expected fp8 on disk, "
                            f"got {tensor.dtype} — coercing"
                        )
                    src = tensor.to(torch.float8_e4m3fn)
                else:
                    src = tensor
                if src.shape != param.shape:
                    if verbose:
                        print(
                            f"  WARN shape mismatch {hf_suffix}: "
                            f"ckpt {tuple(src.shape)} vs param "
                            f"{tuple(param.shape)} — skipping"
                        )
                    unmapped.add(hf_suffix)
                    continue
                # We need to copy without going through a regular cast
                # (FP8 → FP8 copy is fine).
                param.data.copy_(src)
            consumed.add(hf_suffix)
        except Exception as exc:
            if verbose:
                print(f"  ERROR loading {hf_suffix} -> {attr_name}: {exc}")
            unmapped.add(hf_suffix)

    # ---- Routed experts (256 of them, w1/w2/w3 each FP8 + scale) ----
    n_experts = layer.n_routed_experts
    for e in range(n_experts):
        for w_name in ("w1", "w2", "w3"):
            wsuf = f"ffn.experts.{e}.{w_name}.weight"
            ssuf = f"ffn.experts.{e}.{w_name}.scale"
            target_w = getattr(layer, f"experts_{w_name}_weight")
            target_s = getattr(layer, f"experts_{w_name}_scale")
            if wsuf in tensors:
                wt = tensors[wsuf]
                if wt.shape == target_w[e].shape:
                    target_w.data[e].copy_(wt)
                    consumed.add(wsuf)
                else:
                    unmapped.add(wsuf)
            if ssuf in tensors:
                st = tensors[ssuf].to(torch.float32)
                if st.shape == target_s[e].shape:
                    target_s.data[e].copy_(st)
                    consumed.add(ssuf)
                else:
                    unmapped.add(ssuf)

    # ---- Shared expert ----
    for w_name in ("w1", "w2", "w3"):
        wsuf = f"ffn.shared_experts.{w_name}.weight"
        ssuf = f"ffn.shared_experts.{w_name}.scale"
        target_w = getattr(layer, f"shared_{w_name}_weight")
        target_s = getattr(layer, f"shared_{w_name}_scale")
        if wsuf in tensors:
            wt = tensors[wsuf]
            if wt.shape == target_w.shape:
                target_w.data.copy_(wt)
                consumed.add(wsuf)
            else:
                unmapped.add(wsuf)
        if ssuf in tensors:
            st = tensors[ssuf].to(torch.float32)
            if st.shape == target_s.shape:
                target_s.data.copy_(st)
                consumed.add(ssuf)
            else:
                unmapped.add(ssuf)

    if verbose:
        all_ckpt_keys = set(tensors.keys())
        missing = all_ckpt_keys - consumed - unmapped
        print(
            f"\n[convert] Loaded {len(consumed)} keys, "
            f"{len(unmapped)} unmapped, "
            f"{len(missing)} unaccounted-for ckpt keys."
        )
        for u in sorted(unmapped):
            print(f"  unmapped HF key (no MPK target): {u}")
        for m in sorted(missing):
            print(f"  unaccounted ckpt key: {m}")
    return consumed, unmapped


__all__ = ["load_layer0_weights"]
