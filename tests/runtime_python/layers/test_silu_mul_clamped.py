"""Tests for ``layers.SiluMul`` with V4-Flash clamped SwiGLU.

Two test functions:

* ``test_silu_mul_unclamped`` — regression test that the V3 path
  (``swiglu_limit=None``) is byte-identical to the existing behavior.
* ``test_silu_mul_clamped`` — exercises the new V4-Flash clamp path
  (``swiglu_limit=10.0``) with hand-tuned inputs that put both branches
  of the clamp into play.

Both use ``grid_dim=(1, 1, 1)`` so the per-task halved layout matches the
whole-tensor layout (gate || up). See ``test_silu_mul.py`` for the
existing test on which this builds.

Run:
    python tests/runtime_python/layers/test_silu_mul_clamped.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.layers.activation.silu_mul import SiluMul
from mirage.mpk.persistent_kernel import PersistentKernel


def _make_pk(batch_size: int) -> PersistentKernel:
    """Construct a tiny test-mode PersistentKernel.

    Pre-seeds ``qo_indptr_buffer[batch_size] = batch_size`` so the silu_mul
    kernel processes ``batch_size`` tokens (the kernel reads
    ``num_active_tokens`` from this slot — see register_silu_mul_task).
    Without the seed the kernel iterates zero times and out_buf stays zero.
    Same pattern as ``tests/runtime_python/test_mode/test_qwen3_mlp_testmode.py``.
    """
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    qo_indptr_buffer = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
    qo_indptr_buffer[batch_size] = batch_size
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    params["meta_tensors"] = {"qo_indptr_buffer": qo_indptr_buffer}
    return PersistentKernel(**params)


def _run_silu_mul_case(batch_size, intermediate_size, gateup, swiglu_limit, seed):
    """Shared driver for both test cases."""
    device = "cuda"
    dtype = gateup.dtype

    m = SiluMul(
        intermediate_size=intermediate_size, swiglu_limit=swiglu_limit
    ).to(device=device, dtype=dtype)

    # PyTorch reference: same clamp logic as the kernel.
    ref = m.forward(gateup)

    out_buf = torch.zeros(
        batch_size, intermediate_size, dtype=dtype, device=device
    )

    pk = _make_pk(batch_size)
    gateup_dt = pk.attach_input(gateup, name=f"silu_mul_gateup_{seed}")

    print(f"\n{'=' * 60}")
    print(
        f"Test: SiluMul  B={batch_size}, intermediate={intermediate_size}, "
        f"swiglu_limit={swiglu_limit}"
    )
    print("Building module inside compile_scope ...")
    with pk.compile_scope():
        out_dt = m.compile(gateup_dt, output=out_buf, grid_dim=(1, 1, 1))
    assert out_dt is not None, "SiluMul.compile returned None"

    print("Compiling test kernel ...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running test kernel ...")
    pk()
    torch.cuda.synchronize()

    print(f"out_buf[0, :8]: {out_buf[0, :8]}")
    print(f"ref[0, :8]:     {ref[0, :8]}")

    max_diff = (out_buf.float() - ref.float()).abs().max().item()
    print(f"Max absolute diff: {max_diff:.6f}")

    try:
        torch.testing.assert_close(out_buf, ref, atol=1e-2, rtol=1e-2)
    except AssertionError as exc:
        print(f"FAILED: torch.testing.assert_close raised: {exc}")
        pk.finalize()
        sys.exit(1)

    label = "clamped" if swiglu_limit is not None else "unclamped"
    print(f"PASSED: SiluMul ({label}) matches PyTorch reference")
    pk.finalize()


def test_silu_mul_unclamped():
    """Regression: V3 path (swiglu_limit=None) unchanged."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    batch_size = 8
    intermediate_size = 2048
    fused_outdim = 2 * intermediate_size

    gateup = torch.randn(batch_size, fused_outdim, dtype=dtype, device=device)
    _run_silu_mul_case(
        batch_size, intermediate_size, gateup, swiglu_limit=None, seed=0
    )
    print("Test (unclamped) completed successfully!")


def test_silu_mul_clamped():
    """V4-Flash: gate = clamp(max=L), up = clamp(-L, L), then silu*mul."""
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(1)

    batch_size = 4
    intermediate_size = 2048
    fused_outdim = 2 * intermediate_size  # 4096

    # Hand-tuned inputs that exercise BOTH branches of the clamp:
    #   gate: linspace(-15, 15) -> values above L=10 get max-clamped down,
    #         values below -L pass through unchanged (asymmetric clamp).
    #   up:   linspace(-20, 20) -> both -L and +L bounds are hit.
    gate_row = torch.linspace(-15.0, 15.0, intermediate_size, dtype=dtype, device=device)
    up_row = torch.linspace(-20.0, 20.0, intermediate_size, dtype=dtype, device=device)
    gateup = torch.empty(batch_size, fused_outdim, dtype=dtype, device=device)
    # Layout per token: [gate | up].  Add tiny per-row jitter so each batch
    # row is not identical (catches any inadvertent row-broadcast bug).
    for b in range(batch_size):
        jitter = 0.05 * b
        gateup[b, :intermediate_size] = gate_row + jitter
        gateup[b, intermediate_size:] = up_row - jitter

    _run_silu_mul_case(
        batch_size,
        intermediate_size,
        gateup,
        swiglu_limit=10.0,
        seed=1,
    )
    print("Test (clamped) completed successfully!")


if __name__ == "__main__":
    test_silu_mul_unclamped()
    test_silu_mul_clamped()
