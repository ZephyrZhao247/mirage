"""Catalog test: ``layers.hc.MhcPre`` via PersistentKernel test_mode.

Port of ``tests/runtime_python/test_mode/test_mhc_pre_testmode.py`` to
the new ``MPKModule`` API. Compares the kernel's outputs (post_mix,
comb_mix, layer_input) against the module's eager ``forward()`` oracle.
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_mhc_pre_testmode():
    device = "cuda"
    torch.manual_seed(0)

    N = 4
    HC = 4
    H = 128
    HC3 = HC * (HC + 2)  # 24

    # Set up inputs from an honest prenorm GEMM so the rsqrt math is stable.
    fn = torch.randn(HC3, HC * H, dtype=torch.float32, device=device) * 0.05
    residual = torch.randn(N, HC, H, dtype=torch.bfloat16, device=device)

    x_flat = residual.reshape(N, HC * H).float()
    gemm_out_mul = torch.nn.functional.linear(x_flat, fn).unsqueeze(0).contiguous()
    gemm_out_sqrsum = x_flat.square().sum(-1).unsqueeze(0).contiguous()

    hc_scale = torch.tensor([0.5, 0.5, 1.0], dtype=torch.float32, device=device)
    hc_base = torch.randn(HC3, dtype=torch.float32, device=device) * 0.1

    # Output buffers — attached so the host can read back.
    post_mix = torch.zeros(N, HC, dtype=torch.float32, device=device)
    comb_mix = torch.zeros(N, HC, HC, dtype=torch.float32, device=device)
    layer_input = torch.zeros(N, H, dtype=torch.bfloat16, device=device)

    m = layers.MhcPre(hidden_size=H, hc_mult=HC, sinkhorn_iters=20, prefix="t_")
    ref_post, ref_comb, ref_layer = m.forward(
        gemm_out_mul, gemm_out_sqrsum, residual, hc_scale, hc_base
    )

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

    gm_dt = pk.attach_input(gemm_out_mul, name="gemm_out_mul")
    gs_dt = pk.attach_input(gemm_out_sqrsum, name="gemm_out_sqrsum")
    hs_dt = pk.attach_input(hc_scale, name="hc_scale")
    hb_dt = pk.attach_input(hc_base, name="hc_base")
    re_dt = pk.attach_input(residual, name="residual")

    with pk.compile_scope():
        m.compile(
            gm_dt, gs_dt, hs_dt, hb_dt, re_dt,
            post_mix=post_mix,
            comb_mix=comb_mix,
            layer_input=layer_input,
            block_dim=(128, 1, 1),
        )

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    post_diff = (post_mix - ref_post).abs().max().item()
    comb_diff = (comb_mix - ref_comb).abs().max().item()
    layer_diff = (layer_input.float() - ref_layer.float()).abs().max().item()
    print(f"post_mix    max diff: {post_diff:.6e}")
    print(f"comb_mix    max diff: {comb_diff:.6e}")
    print(f"layer_input max diff: {layer_diff:.6e}")

    try:
        torch.testing.assert_close(post_mix, ref_post, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(comb_mix, ref_comb, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(
            layer_input.float(), ref_layer.float(), atol=1e-2, rtol=1e-2
        )
        print("PASSED: MhcPre compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: MhcPre compile() disagrees with forward()\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_mhc_pre_testmode()
