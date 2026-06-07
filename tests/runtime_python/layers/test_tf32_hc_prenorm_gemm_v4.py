"""V4-Flash ``tf32_hc_prenorm_gemm`` (NEW kernel, DeepGEMM-sm_100a variant) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/tf32_hc_prenorm_gemm.md``.

DeepGEMM CUDA Class-A locked-alternative of ``hc_prenorm_gemm``; same
outputs at n_splits=1. The naive port aliases to the same device impl.
Multi-batch from day 1 (``max_num_batched_requests = 4``).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.hc import V4Tf32HcPrenormGemm


def test_tf32_hc_prenorm_gemm_v4_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    num_tokens = 4
    hidden_size = 256
    hc_mult = 4
    hc_mult3 = hc_mult * (2 + hc_mult)
    K = hc_mult * hidden_size

    m = V4Tf32HcPrenormGemm(
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        prefix="v4_tf32_hc_gemm_",
    )
    m.fn.data = m.fn.data.to(device=device, dtype=torch.float32)
    m.fn.data.copy_(
        torch.randn(hc_mult3, K, dtype=torch.float32, device=device) * 0.02
    )

    x = torch.randn(num_tokens, K, dtype=dtype, device=device)

    gemm_out_buf = torch.zeros(
        1, num_tokens, hc_mult3, dtype=torch.float32, device=device
    )
    sqrsum_buf = torch.zeros(
        1, num_tokens, dtype=torch.float32, device=device
    )

    ref_gemm, ref_sqr = m.forward(x)

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

    x_dt = pk.attach_input(x, name="x")

    with pk.compile_scope():
        _ = m.compile(x_dt, gemm_out=gemm_out_buf, sqrsum=sqrsum_buf)

    print("Compiling V4Tf32HcPrenormGemm test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4Tf32HcPrenormGemm test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"gemm_out_buf[0, 0, :8]: {gemm_out_buf[0, 0, :8]}")
    print(f"ref_gemm    [0, 0, :8]: {ref_gemm[0, 0, :8]}")
    max_g = (gemm_out_buf - ref_gemm).abs().max().item()
    print(f"gemm_out max-abs diff: {max_g}")
    print(f"sqrsum_buf [0, :]: {sqrsum_buf[0, :]}")
    print(f"ref_sqr    [0, :]: {ref_sqr[0, :]}")
    max_s = (sqrsum_buf - ref_sqr).abs().max().item()
    print(f"sqrsum   max-abs diff: {max_s}")

    try:
        torch.testing.assert_close(gemm_out_buf, ref_gemm, atol=0.05, rtol=0.02)
        torch.testing.assert_close(sqrsum_buf, ref_sqr, atol=0.5, rtol=0.02)
        print("PASSED: V4Tf32HcPrenormGemm compile() matches forward()")
    except AssertionError as e:
        print(
            "FAILED: V4Tf32HcPrenormGemm compile() disagrees with forward()\n"
            f"{e}"
        )
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_tf32_hc_prenorm_gemm_v4_testmode()
