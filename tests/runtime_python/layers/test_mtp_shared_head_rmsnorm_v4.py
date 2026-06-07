"""V4-Flash ``mtp_shared_head_rmsnorm`` (REUSE alias) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/mtp_shared_head_rmsnorm.md``.

This is the REUSE-decision test: :class:`V4MTPSharedHeadRMSNorm` is a
zero-behavior-delta subclass of :class:`mirage.mpk.layers.RMSNorm`,
which is in turn backed by the Hopper/Blackwell task
``rmsnorm_hopper``. We just confirm the alias still produces the right
numbers end-to-end through the MPK pipeline.

Multi-batch from day 1 (``max_num_batched_requests = 4``), matching
the V4-Flash MTP draft-step shape ``[T, H=4096]``.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.mtp import V4MTPSharedHeadRMSNorm


def test_mtp_shared_head_rmsnorm_v4_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # V4-Flash MTP shared-head RMSNorm shape (per spec):
    #   hidden_states: [num_tokens, HIDDEN=4096] bf16
    # Multi-batch from day 1: 4 tokens.
    batch_size = 4
    hidden_size = 4096
    eps = 1e-6  # matches the kernel's hard-coded 1e-6f

    # ------------------------------------------------------------------
    # Build the alias module + reference
    # ------------------------------------------------------------------
    m = V4MTPSharedHeadRMSNorm(
        hidden_size=hidden_size, eps=eps, prefix="v4_mtp_head_"
    )
    # Match the V4-Flash kernel's RMSNorm.weight dtype (bf16). Random
    # weight (not the all-ones default) so the scale path is exercised.
    w = torch.randn(hidden_size, dtype=dtype, device=device)
    m.weight.data = m.weight.data.to(device=device, dtype=dtype)
    m.weight.data.copy_(w)

    x = torch.randn(batch_size, hidden_size, dtype=dtype, device=device)
    out_buf = torch.zeros(batch_size, hidden_size, dtype=dtype, device=device)

    # PyTorch reference (fp32 reduction, bf16 output, with our random
    # weight). Inherited from layers.RMSNorm.forward().
    ref = m.forward(x)

    # ------------------------------------------------------------------
    # Build PersistentKernel in test mode
    # ------------------------------------------------------------------
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x, name="x")

    with pk.compile_scope():
        _ = m.compile(x_dt, output=out_buf)

    # ------------------------------------------------------------------
    # Compile and run once
    # ------------------------------------------------------------------
    print("Compiling V4MTPSharedHeadRMSNorm test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4MTPSharedHeadRMSNorm test kernel...")
    pk()
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    print(f"out_buf[:2, :8]: {out_buf[:2, :8]}")
    print(f"ref[:2, :8]:     {ref[:2, :8]}")
    max_diff = (out_buf.float() - ref.float()).abs().max().item()
    print(f"Max absolute difference: {max_diff}")

    try:
        torch.testing.assert_close(out_buf, ref, atol=0.05, rtol=0.05)
        print("PASSED: V4MTPSharedHeadRMSNorm compile() matches forward()")
    except AssertionError as e:
        print(
            "FAILED: V4MTPSharedHeadRMSNorm compile() disagrees with forward()\n"
            f"{e}"
        )
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_mtp_shared_head_rmsnorm_v4_testmode()
