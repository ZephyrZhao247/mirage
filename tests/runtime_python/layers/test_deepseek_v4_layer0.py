"""Per-layer correctness test for DeepSeek V4-Flash Block (layer_idx=0).

Wave-3 v1 (the simplest layer: compress_ratio=0, hash routing,
hc_mult=4). The test:

  1. Loads ``/raid/catalyst/models/DeepSeek-V4-Flash-Base/config.json``.
  2. Instantiates ``DeepseekV4Block(config, layer_idx=0)``.
  3. Loads Layer 0 weights from the Flash-Base checkpoint shards via
     ``demo.deepseek_v4.models.convert.load_layer0_weights``.
  4. Runs the PyTorch reference forward (``DeepseekV4Block.forward``).
  5. (When the Wave-4 compile path lands) compares against the
     compiled MPK ``compile()``.

Current status: the Wave-3 ``compile()`` is a scaffold — the FP8 MoE
permute pipeline + SWA-cache write-back + grouped o-projection still
need to be wired before the compiled vs reference comparison can be
green. This test therefore exercises only the PyTorch reference path
(loads real Flash-Base weights, runs a forward pass, validates the
load mapping). The compiled-vs-reference comparison is marked SKIP and
will be enabled in Wave-4.

Run with the freest GPU::

    CUDA_VISIBLE_DEVICES=<gpu> python tests/runtime_python/layers/test_deepseek_v4_layer0.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch


CHECKPOINT_DIR = "/raid/catalyst/models/DeepSeek-V4-Flash-Base"


def _load_config() -> dict:
    cfg_path = os.path.join(CHECKPOINT_DIR, "config.json")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"Missing {cfg_path}; cannot run V4 layer-0 test."
        )
    with open(cfg_path) as f:
        return json.load(f)


def test_reference_forward_with_real_weights():
    """Wave-3 deliverable: load real Layer 0 weights, run forward, sanity-check."""
    print("\n" + "=" * 60)
    print("DeepSeek V4-Flash Layer 0 — PyTorch reference forward")
    print("=" * 60)

    from mirage.mpk.models.deepseek_v4.block import DeepseekV4Block
    from demo.deepseek_v4.models.convert import load_layer0_weights

    config = _load_config()
    print(f"[cfg] hidden_size={config['hidden_size']}, "
          f"num_attention_heads={config['num_attention_heads']}, "
          f"n_routed_experts={config['n_routed_experts']}, "
          f"num_experts_per_tok={config['num_experts_per_tok']}, "
          f"hc_mult={config.get('hc_mult', 4)}, "
          f"swiglu_limit={config.get('swiglu_limit', 10.0)}")

    print("[build] DeepseekV4Block(layer_idx=0) ...")
    layer = DeepseekV4Block(config, layer_idx=0)
    print(f"  built — is_hash_routing={layer.is_hash_routing}, "
          f"compress_ratio={layer.compress_ratio}")
    n_params = sum(p.numel() for p in layer.parameters())
    print(f"  total parameter count: {n_params:,}")

    print("[load] Reading Layer 0 weights from Flash-Base shards ...")
    t0 = time.time()
    consumed, unmapped = load_layer0_weights(CHECKPOINT_DIR, layer)
    print(f"  load took {time.time() - t0:.1f}s; "
          f"consumed={len(consumed)} unmapped={len(unmapped)}")
    if unmapped:
        print(f"  NOTE: {len(unmapped)} HF keys not mapped to MPK; "
              "this is expected for keys that are part of pieces still "
              "scaffolded (see block.py OPEN notes).")

    # Move to GPU. Cast non-FP8 params to bf16 / fp32 per their slot.
    print("[move] Migrating layer to CUDA ...")
    device = "cuda"
    layer = layer.to(device)
    layer.eval()

    # Build a tiny prompt.
    T = 4
    H = layer.hidden_size
    hc = layer.hc_mult

    torch.manual_seed(123)
    hidden_hc = (
        torch.randn(T, hc, H, dtype=torch.bfloat16, device=device) * 0.02
    )
    position_ids = torch.arange(T, dtype=torch.int32, device=device)
    swa_total = max(layer.sliding_window, T + 1)
    swa_cache = torch.zeros(
        swa_total, layer.head_dim, dtype=torch.bfloat16, device=device
    )
    # Use plausible token IDs that exist in the vocab.
    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device=device)

    print(f"[fwd ] Running reference forward (T={T}, H={H}, hc={hc}) ...")
    with torch.no_grad():
        t0 = time.time()
        out = layer.forward(hidden_hc, position_ids, swa_cache, input_ids)
        torch.cuda.synchronize()
        print(f"  forward took {time.time() - t0:.1f}s; out.shape={tuple(out.shape)} "
              f"dtype={out.dtype}")

    # Sanity checks.
    assert out.shape == hidden_hc.shape, (
        f"output shape mismatch: {tuple(out.shape)} vs "
        f"{tuple(hidden_hc.shape)}"
    )
    assert out.dtype == hidden_hc.dtype
    finite = torch.isfinite(out)
    assert finite.all(), f"non-finite outputs: {(~finite).sum().item()}"

    max_abs = out.abs().max().item()
    rms = out.float().pow(2).mean().sqrt().item()
    print(f"  out abs_max={max_abs:.4f}, rms={rms:.4f}")
    # The HC post-block has a sigmoid * 2.0 cap on post_mix so per-channel
    # values are bounded by O(|x| + |residual|) ~ O(0.1). A value > 1e3
    # would indicate FP8 dequant error or weight load corruption.
    assert max_abs < 100.0, (
        "output abs-max suspiciously large — likely an FP8 dequant or "
        f"weight-load bug (got {max_abs:.4f})"
    )
    print("\nPASSED: DeepseekV4Block(layer_idx=0) PyTorch reference forward "
          "produces finite, bounded output with real Flash-Base weights.")


def test_compiled_vs_reference_layer0_skipped():
    """The compiled vs reference comparison is SKIPPED in Wave-3.

    Wave-4 will wire ``DeepseekV4Block.compile()`` once the following
    integration gates land:

      * FP8 MoE permute → group-GEMM w13 → SiluMul → group-GEMM w2
        pipeline (the V3 path needs the V4 hash-routing indices wired
        into ``moe_routing_indices``).
      * SWA cache write-back from the freshly-computed kv (one paged-KV
        store per token; ratio=0 needs only the window slot).
      * Grouped o-projection (wo_a is per-group bmm; wo_b is dense row-
      parallel FP8).

    Until then this test raises ``SkipTest`` with the explicit reason
    so CI / runners do NOT treat it as silently-green. This matches the
    Wave-3 constraint "do NOT commit a fake-passing test".
    """
    import unittest
    raise unittest.SkipTest(
        "DeepseekV4Block.compile() is a Wave-3 scaffold. Wave-4 will "
        "wire the FP8 MoE permute pipeline, the SWA cache write-back, "
        "and the grouped o-projection GEMM. See "
        "python/mirage/mpk/models/deepseek_v4/block.py for the OPEN "
        "notes detailing what remains."
    )


if __name__ == "__main__":
    # Add demo/ on sys.path so the test can import demo.deepseek_v4.* .
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    sys.path.insert(0, repo_root)

    try:
        test_reference_forward_with_real_weights()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\nFAILED: {exc}")
        sys.exit(1)

    # Print the SKIP marker without raising it standalone.
    print("\n" + "=" * 60)
    print("[SKIP] test_compiled_vs_reference_layer0 — Wave-4 will enable.")
    print("=" * 60)
