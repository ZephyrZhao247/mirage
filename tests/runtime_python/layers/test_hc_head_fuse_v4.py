"""V4-Flash ``hc_head_fuse`` (NEW kernel) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/hc_head_fuse_tilelang.md``.

Terminal mHC kernel collapsing ``[T, HC_MULT, HIDDEN]`` -> ``[T, HIDDEN]``
via flatten-RMS + sigmoid-gated weighted sum. Multi-batch from day 1
(``max_num_batched_requests = 4``).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.hc import V4HcHeadFuse


def test_hc_head_fuse_v4_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # Small shape for fast iteration; the kernel is generic.
    num_tokens = 4
    hidden_size = 256
    hc_mult = 4
    K = hc_mult * hidden_size
    rms_eps = 1e-6
    hc_eps = 1e-6

    m = V4HcHeadFuse(
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        rms_eps=rms_eps,
        hc_eps=hc_eps,
        prefix="v4_hc_head_",
    )
    # fp32 weights (matches reference).
    m.fn.data = m.fn.data.to(device=device, dtype=torch.float32)
    m.fn.data.copy_(
        torch.randn(hc_mult, K, dtype=torch.float32, device=device) * 0.02
    )
    m.hc_scale.data = m.hc_scale.data.to(device=device, dtype=torch.float32)
    m.hc_scale.data.copy_(
        torch.tensor([0.5], dtype=torch.float32, device=device)
    )
    m.hc_base.data = m.hc_base.data.to(device=device, dtype=torch.float32)
    m.hc_base.data.copy_(
        torch.randn(hc_mult, dtype=torch.float32, device=device) * 0.1
    )

    residual = torch.randn(
        num_tokens, hc_mult, hidden_size, dtype=dtype, device=device
    )
    out_buf = torch.zeros(
        num_tokens, hidden_size, dtype=dtype, device=device
    )

    ref_out = m.forward(residual)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = num_tokens
    params["max_num_batched_requests"] = num_tokens
    pk = PersistentKernel(**params)

    residual_dt = pk.attach_input(residual, name="residual")

    with pk.compile_scope():
        _ = m.compile(residual_dt, output=out_buf)

    print("Compiling V4HcHeadFuse test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4HcHeadFuse test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"out_buf[0, :8]: {out_buf[0, :8]}")
    print(f"ref_out[0, :8]: {ref_out[0, :8]}")
    max_o = (out_buf.float() - ref_out.float()).abs().max().item()
    print(f"out max-abs diff: {max_o}")

    try:
        torch.testing.assert_close(out_buf, ref_out, atol=0.05, rtol=0.05)
        print("PASSED: V4HcHeadFuse compile() matches forward()")
    except AssertionError as e:
        print(
            "FAILED: V4HcHeadFuse compile() disagrees with forward()\n"
            f"{e}"
        )
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_hc_head_fuse_v4_testmode()
