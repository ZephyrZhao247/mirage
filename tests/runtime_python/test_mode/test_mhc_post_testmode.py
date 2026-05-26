"""Test for PersistentKernel.mhc_post_layer (DeepSeek V4-Flash, K5 step).

Builds a minimal MPK graph with the mhc_post_layer, runs in test_mode, and
compares against a PyTorch reference extracted from
deps/vllm/.../mhc.py:mhc_post_tilelang (= model.py:684-687).
"""

import os
import sys
import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def torch_mhc_post_ref(x_bf16, residual_bf16, post_mix_f32, comb_mix_f32):
    """Reference implementation of mhc_post.

    Inputs:
      x        : [N, H]      bf16
      residual : [N, hc, H]  bf16
      post_mix : [N, hc]     fp32
      comb_mix : [N, hc, hc] fp32 (indexed as [n, hc_in, hc_out])

    Output:
      out      : [N, hc, H]  bf16
    """
    # post: [N, hc, 1] * x: [N, 1, H] -> [N, hc, H]
    term1 = post_mix_f32.unsqueeze(-1) * x_bf16.unsqueeze(-2).float()
    # comb[n, hc_in, hc_out] * residual[n, hc_in, h], summed over hc_in
    # -> result indexed by [n, hc_out, h].
    # comb_mix: [N, hc_in, hc_out, 1], residual: [N, hc_in, 1, H]
    term2 = torch.sum(
        comb_mix_f32.unsqueeze(-1) * residual_bf16.unsqueeze(-2).float(),
        dim=1,  # contract hc_in (axis 1 of [N, hc_in, hc_out, H])
    )
    return (term1 + term2).to(torch.bfloat16)


def test_mhc_post_testmode():
    device = "cuda"
    N = 4
    hc = 4
    H = 128

    torch.manual_seed(0)
    x = torch.randn(N, H, dtype=torch.bfloat16, device=device)
    residual = torch.randn(N, hc, H, dtype=torch.bfloat16, device=device)
    post_mix = torch.randn(N, hc, dtype=torch.float32, device=device)
    comb_mix = torch.randn(N, hc, hc, dtype=torch.float32, device=device)
    out = torch.zeros(N, hc, H, dtype=torch.bfloat16, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x, name="x")
    residual_dt = pk.attach_input(residual, name="residual")
    post_dt = pk.attach_input(post_mix, name="post_mix")
    comb_dt = pk.attach_input(comb_mix, name="comb_mix")
    out_dt = pk.attach_input(out, name="out")

    # One CTA per token. 128 threads/CTA matches TileLang n_thr=128.
    pk.mhc_post_layer(
        x=x_dt,
        residual=residual_dt,
        post_mix=post_dt,
        comb_mix=comb_dt,
        out=out_dt,
        grid_dim=(N, 1, 1),
        block_dim=(128, 1, 1),
    )

    print("Compiling test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running test kernel...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    ref = torch_mhc_post_ref(x, residual, post_mix, comb_mix)
    print(f"Output[0,0,:8]:    {out[0, 0, :8]}")
    print(f"Reference[0,0,:8]: {ref[0, 0, :8]}")

    # bf16 tolerance per Wave-2 gate.
    ok = torch.allclose(out, ref, rtol=1e-3, atol=1e-3)
    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"Max absolute difference: {max_diff}")

    if not ok:
        print(f"FAILED: outputs differ beyond tolerance (max diff {max_diff})")
        pk.finalize()
        sys.exit(1)

    print("PASSED: mhc_post_layer test_mode produces correct output")
    pk.finalize()


if __name__ == "__main__":
    test_mhc_post_testmode()
