"""Catalog test: ``layers.hc.MhcPost`` via PersistentKernel test_mode.

Port of ``tests/runtime_python/test_mode/test_mhc_post_testmode.py`` to
the new ``MPKModule`` API.
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_mhc_post_testmode():
    device = "cuda"
    torch.manual_seed(0)

    N = 4
    hc = 4
    H = 128

    x = torch.randn(N, H, dtype=torch.bfloat16, device=device)
    residual = torch.randn(N, hc, H, dtype=torch.bfloat16, device=device)
    post_mix = torch.randn(N, hc, dtype=torch.float32, device=device)
    comb_mix = torch.randn(N, hc, hc, dtype=torch.float32, device=device)
    out = torch.zeros(N, hc, H, dtype=torch.bfloat16, device=device)

    m = layers.MhcPost(hidden_size=H, hc_mult=hc, prefix="t_")
    ref = m.forward(x, residual, post_mix, comb_mix)

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

    x_dt = pk.attach_input(x, name="x")
    residual_dt = pk.attach_input(residual, name="residual")
    post_dt = pk.attach_input(post_mix, name="post_mix")
    comb_dt = pk.attach_input(comb_mix, name="comb_mix")

    with pk.compile_scope():
        m.compile(
            x_dt, residual_dt, post_dt, comb_dt,
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
        torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
        print("PASSED: MhcPost compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: MhcPost compile() disagrees with forward()\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_mhc_post_testmode()
