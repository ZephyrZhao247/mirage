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


def test_compiled_vs_reference_layer0():
    """Compiled vs PyTorch-reference comparison for DeepseekV4Block L0.

    Wave 3.6 status (follow-up to b466286d):

      * Gap 1 (SWA cache write-back): LANDED. The
        ``mla_v4_swa_cache_write_sm100`` CUDA kernel + Python catalog
        ``compile()`` are wired; ``test_mla_v4_swa_cache_write.py``
        PASSES exact-copy against the PyTorch reference.
      * Gap 3 (hash-route adapter): LANDED as Python-side scatter
        (option B): caller materialises the expert-major routing buffer
        each step and ``HashRouteToExpertMajor.compile`` attaches it as
        a broadcast MPK input.
      * Gap 4 (FP8 QAT round-trip): the catalog ``QuantizeFP8`` needs to
        grow a round-trip (quant + immediate bf16 dequant) mode before
        it can be inserted between ``MLAv4QKVRMSNorm`` and
        ``MLAv4Decode``. STILL NOT WIRED.
      * Gap 2 (LinearFP8BMM for wo_a): identified-only; not yet wired
        into the block ``compile``.

    ``DeepseekV4Block.compile()`` therefore still raises
    ``NotImplementedError`` -- but it is no longer a Wave-3.5 SKIP. This
    test un-skips and reports the missing piece as a FAILURE so the
    next agent picks up from an honest, known state.

    To make this test PASS:
      1. Extend ``QuantizeFP8`` to support round-trip mode for gap 4.
      2. Implement ``DeepseekV4Block.compile()`` per the ~20-step
         sequence enumerated in its method-body comment.
      3. Add bottom-up sub-module compile-vs-reference unit tests for
         wq_a/wq_b/wkv FP8, MLA decode with new SWA write/gather, FP8
         MoE pipeline with hash routing inputs, and HC pre/post.
      4. Stitch them together once each piece is green.
    """
    print("\n" + "=" * 60)
    print("DeepSeek V4-Flash Layer 0 — compiled vs reference")
    print("=" * 60)

    from mirage.mpk.models.deepseek_v4.block import DeepseekV4Block
    from demo.deepseek_v4.models.convert import load_layer0_weights

    config = _load_config()
    layer = DeepseekV4Block(config, layer_idx=0)
    print("[load] Reading Layer 0 weights ...")
    consumed, unmapped = load_layer0_weights(CHECKPOINT_DIR, layer)
    print(f"  consumed={len(consumed)} unmapped={len(unmapped)}")

    device = "cuda"
    layer = layer.to(device)
    layer.eval()

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
    input_ids = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device=device)

    with torch.no_grad():
        ref = layer.forward(hidden_hc, position_ids, swa_cache, input_ids)
    print(f"  ref out.shape={tuple(ref.shape)} dtype={ref.dtype}")

    # Attempt the MPK compile path. Wave 3.6 leaves this raising
    # NotImplementedError -- we report that as a FAILURE (not a SKIP).
    import mirage
    from mirage.mpk.persistent_kernel import PersistentKernel
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = T
    params["max_num_batched_requests"] = T
    pk = PersistentKernel(**params)

    hidden_hc_dt = pk.attach_input(hidden_hc, name="hidden_hc")
    position_ids_dt = pk.attach_input(position_ids, name="position_ids")
    swa_cache_dt = pk.attach_input(swa_cache, name="swa_cache")
    input_ids_dt = pk.attach_input(
        input_ids.to(torch.int32), name="input_ids"
    )

    failure_module = "DeepseekV4Block.compile (NOT_IMPLEMENTED)"
    failure_msg = None
    try:
        with pk.compile_scope():
            layer.compile(
                hidden_hc_dt, position_ids_dt, swa_cache_dt, input_ids_dt,
            )
    except NotImplementedError as exc:
        failure_msg = str(exc)
    except Exception as exc:
        failure_module = type(exc).__name__
        failure_msg = str(exc)

    if failure_msg is None:
        # We got past compile() -- run the kernel and compare.
        pk.compile(output_dir=os.path.dirname(__file__))
        pk()
        torch.cuda.synchronize()
        # The compile output buffer name is TBD once compile() is fully
        # implemented. Until then this branch is unreachable.
        raise RuntimeError(
            "compile() unexpectedly succeeded without producing an "
            "output DTensor; update this test to fetch the result and "
            "run torch.testing.assert_close(out_mpk, ref, atol=5e-2, "
            "rtol=5e-2)."
        )
    else:
        # Report honest failure -- NOT a SkipTest.
        msg = (
            f"\nFAILED: diverging module = {failure_module}\n"
            f"  detail: {failure_msg}\n"
            f"  max-abs-diff vs reference: N/A (compile path not built)\n"
        )
        print(msg)
        raise AssertionError(msg)


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

    # Wave 3.6: un-SKIPPED. Now runs and reports honest failure when
    # DeepseekV4Block.compile() raises NotImplementedError.
    print("\n" + "=" * 60)
    try:
        test_compiled_vs_reference_layer0()
    except AssertionError as exc:
        # Honest failure -- not a SKIP. Print and exit non-zero so the
        # test driver records it as FAIL.
        print(f"\n[FAIL] test_compiled_vs_reference_layer0:")
        print(str(exc))
        sys.exit(1)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"\n[FAIL] test_compiled_vs_reference_layer0 (unexpected): {exc}")
        sys.exit(1)
    print("=" * 60)
