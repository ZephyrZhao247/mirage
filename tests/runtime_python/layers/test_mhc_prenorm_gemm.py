"""Catalog test: ``layers.hc.MhcPrenormGemm`` via PersistentKernel test_mode.

Port of ``tests/runtime_python/test_mode/test_mhc_prenorm_gemm_testmode.py``
to the new ``MPKModule`` API. Validates the v1 decomposed kernel
(``sum_of_squares_sm100``) against the module's eager ``forward()`` oracle.
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_mhc_prenorm_gemm_testmode():
    device = "cuda"
    torch.manual_seed(0)

    N = 4
    HC = 4
    H = 128
    HC3 = (2 + HC) * HC  # 24

    # Inputs.
    residual_3d = torch.randn(N, HC, H, dtype=torch.bfloat16, device=device)
    residual_2d = residual_3d.reshape(N, HC * H).contiguous()
    fn_bf16 = (
        torch.randn(HC3, HC * H, dtype=torch.float32, device=device) * 0.05
    ).to(torch.bfloat16).contiguous()

    # Output buffers — attached so the host can read back.
    gemm_out_mul = torch.zeros(N, HC3, dtype=torch.bfloat16, device=device)
    gemm_out_sqrsum = torch.zeros(N, dtype=torch.float32, device=device)

    m = layers.MhcPrenormGemm(hidden_size=H, hc_mult=HC, prefix="t_")
    ref_mul, ref_sqrsum = m.forward(residual_3d, fn_bf16)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = N
    params["max_num_batched_requests"] = N
    pk = PersistentKernel(**params)

    residual_dt = pk.attach_input(residual_2d, name="residual")
    fn_dt = pk.attach_input(fn_bf16, name="fn")

    with pk.compile_scope():
        m.compile(
            residual_dt,
            fn_dt,
            gemm_out_mul=gemm_out_mul,
            gemm_out_sqrsum=gemm_out_sqrsum,
        )

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    mul_diff = (gemm_out_mul.float() - ref_mul.float()).abs().max().item()
    sqrsum_diff = (gemm_out_sqrsum - ref_sqrsum).abs().max().item()
    print(f"gemm_out_mul max diff (bf16 path):    {mul_diff:.6e}")
    print(f"gemm_out_sqrsum max diff (fp32 path): {sqrsum_diff:.6e}")

    try:
        # bf16 accumulator path is loose, matching the legacy test gate.
        torch.testing.assert_close(
            gemm_out_mul.float(), ref_mul.float(), atol=5e-1, rtol=5e-1
        )
        torch.testing.assert_close(
            gemm_out_sqrsum, ref_sqrsum, atol=5e-1, rtol=5e-2
        )
        print("PASSED: MhcPrenormGemm compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: MhcPrenormGemm compile() disagrees with forward()\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_mhc_prenorm_gemm_testmode()
