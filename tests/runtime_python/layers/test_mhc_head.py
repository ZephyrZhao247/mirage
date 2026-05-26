"""Catalog test: ``layers.hc.MhcHead`` via PersistentKernel test_mode.

Port of ``tests/runtime_python/test_mode/test_mhc_head_testmode.py`` to
the new ``MPKModule`` API.
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_mhc_head_testmode():
    device = "cuda"
    bf16 = torch.bfloat16
    fp32 = torch.float32
    torch.manual_seed(0)

    N = 4
    hc = 4
    H = 128

    residual = torch.randn(N, hc, H, dtype=bf16, device=device) * 0.5
    fn = torch.randn(hc, hc * H, dtype=fp32, device=device) * 0.05
    hc_scale = torch.full((1,), 0.7, dtype=fp32, device=device)
    hc_base = torch.tensor([-0.2, 0.1, 0.0, 0.3], dtype=fp32, device=device)
    out = torch.zeros(N, H, dtype=bf16, device=device)

    m = layers.MhcHead(hidden_size=H, hc_mult=hc, prefix="t_")
    ref = m.forward(residual, fn, hc_scale, hc_base)

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

    residual_dt = pk.attach_input(residual, name="residual")
    fn_dt = pk.attach_input(fn, name="fn")
    hc_scale_dt = pk.attach_input(hc_scale, name="hc_scale")
    hc_base_dt = pk.attach_input(hc_base, name="hc_base")

    with pk.compile_scope():
        m.compile(
            residual_dt, fn_dt, hc_scale_dt, hc_base_dt,
            out=out,
            block_dim=(128, 1, 1),
        )

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"Max absolute difference: {max_diff:.6e}")

    try:
        torch.testing.assert_close(out, ref, atol=5e-3, rtol=5e-3)
        print("PASSED: MhcHead compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: MhcHead compile() disagrees with forward()\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_mhc_head_testmode()
