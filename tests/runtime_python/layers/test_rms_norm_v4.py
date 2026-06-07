"""V4-Flash ``rms_norm`` test (multi-batch).

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/rms_norm.md``.
Catalog: ``mirage.mpk.layers.deepseek_v4.std.V4RMSNorm`` (REUSE alias of
the existing :class:`mirage.mpk.layers.RMSNorm`).

This test runs the V4 RMSNorm catalog through the full MPK compile +
execute pipeline in ``test_mode=True`` on a multi-batch input, and
compares the kernel output against the catalog's PyTorch ``forward()``
reference.

Run on a free GPU:
    CUDA_VISIBLE_DEVICES=0 python tests/runtime_python/layers/test_rms_norm_v4.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.layers.deepseek_v4.std import V4RMSNorm
from mirage.mpk.persistent_kernel import PersistentKernel


def test_v4_rms_norm_multibatch():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # V4-Flash trunk hidden dim is 4096; use that exact shape so the
    # kernel's HIDDEN_DIM constraint (multiple of NUM_THREADS=256 on
    # Blackwell, with hidden*2/NUM_THREADS >= 4 → hidden >= 512) is
    # comfortably satisfied. Multi-batch from day 1: batch_size >= 2.
    batch_size = 4
    hidden_size = 4096
    eps = 1e-6  # matches V4-Flash rms_norm_eps and the hard-coded 1e-6f
                # in the kernel codegen.

    # Build the catalog module and a random per-channel scale.
    m = V4RMSNorm(hidden_size=hidden_size, eps=eps, prefix="v4rms_")
    w = torch.randn(hidden_size, dtype=dtype, device=device)
    m.weight.data = m.weight.data.to(device=device, dtype=dtype)
    m.weight.data.copy_(w)

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    out_buf = torch.zeros(batch_size, hidden_size, dtype=dtype, device=device)

    ref = m.forward(x)

    # Build PersistentKernel in test mode.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size  # >= 2: multi-batch.
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x, name="x_v4rms")
    with pk.compile_scope():
        _ = m.compile(x_dt, output=out_buf)

    print("Compiling V4 rms_norm test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4 rms_norm test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"out_buf[:2, :8]: {out_buf[:2, :8]}")
    print(f"ref[:2, :8]:     {ref[:2, :8]}")

    max_diff = (out_buf.float() - ref.float()).abs().max().item()
    print(f"Max absolute difference: {max_diff}")

    try:
        # bf16 RMSNorm output tolerance (matches test_rmsnorm.py).
        torch.testing.assert_close(out_buf, ref, atol=0.05, rtol=0.05)
        print("PASSED: V4RMSNorm compile() matches forward() (multi-batch).")
    except AssertionError as e:
        print(f"FAILED: V4RMSNorm disagrees with reference\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_v4_rms_norm_multibatch()
