"""V4-Flash ``mhc_post`` (NEW kernel) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/mhc_post_tilelang.md``.

Per token t, per output hc-stream i_hco, per hidden idx h:
    out[t, i_hco, h] = post_mix[t, i_hco] * x_in[t, h]
                       + sum_i comb_mix[t, i, i_hco] * residual_in[t, i, h]

Multi-batch from day 1 (max_num_batched_requests = 4). hc=4 matches
V4-Flash's HC_MULT.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.hc import V4MhcPost


def test_mhc_post_v4_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # V4-Flash HC config: hc=4, hidden=4096. Multi-batch: 4 tokens.
    num_tokens = 4
    hc = 4
    hidden = 4096

    m = V4MhcPost(hc=hc, hidden_size=hidden, prefix="v4_mhc_post_")

    comb_mix = torch.randn(num_tokens, hc, hc, dtype=torch.float32, device=device)
    residual_in = torch.randn(num_tokens, hc, hidden, dtype=dtype, device=device)
    post_mix = torch.randn(num_tokens, hc, dtype=torch.float32, device=device)
    x_in = torch.randn(num_tokens, hidden, dtype=dtype, device=device)

    out_buf = torch.zeros(num_tokens, hc, hidden, dtype=dtype, device=device)

    ref = m.forward(comb_mix, residual_in, post_mix, x_in)

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

    comb_dt = pk.attach_input(comb_mix, name="comb_mix")
    res_in_dt = pk.attach_input(residual_in, name="residual_in")
    post_dt = pk.attach_input(post_mix, name="post_mix")
    x_dt = pk.attach_input(x_in, name="x_in")

    with pk.compile_scope():
        _ = m.compile(
            comb_dt, res_in_dt, post_dt, x_dt,
            residual_out=out_buf,
        )

    print("Compiling V4MhcPost test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4MhcPost test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"out_buf[0, 0, :8]: {out_buf[0, 0, :8]}")
    print(f"ref[0, 0, :8]:     {ref[0, 0, :8]}")
    max_diff = (out_buf.float() - ref.float()).abs().max().item()
    print(f"Max absolute difference: {max_diff}")

    try:
        torch.testing.assert_close(out_buf, ref, atol=0.1, rtol=0.05)
        print("PASSED: V4MhcPost compile() matches forward().")
    except AssertionError as e:
        print(f"FAILED: V4MhcPost disagrees with reference\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_mhc_post_v4_testmode()
