"""V4-Flash ``mhc_fused`` (NEW kernel) test, decode-regime fused
mhc_post + hc_prenorm_gemm.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/mhc_fused_tilelang.md``.

Produces in one kernel:
  * residual_out -- the new hc-stream residual (=hc_post output)
  * gemm_out_mul, gemm_out_sqrsum -- the next-layer hc_pre GEMM
    partials, with SPLIT_K=1 in the naive impl.

Multi-batch from day 1 (max_num_batched_requests = 4, well under the
T<=16 small-token gate).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.hc import V4MhcFused


def test_mhc_fused_v4_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    num_tokens = 4
    hc = 4
    hidden = 4096
    n_out = hc * (2 + hc)  # 24

    m = V4MhcFused(hc=hc, hidden_size=hidden, n_out=n_out, prefix="v4_mhc_fused_")

    comb_mix = torch.randn(num_tokens, hc, hc, dtype=torch.float32, device=device)
    residual_in = torch.randn(num_tokens, hc, hidden, dtype=dtype, device=device)
    post_mix = torch.randn(num_tokens, hc, dtype=torch.float32, device=device)
    x_in = torch.randn(num_tokens, hidden, dtype=dtype, device=device)
    weight_t = torch.randn(
        n_out, hc, hidden, dtype=torch.float32, device=device,
    ) * 0.01

    gemm_mul_buf = torch.zeros(
        1, num_tokens, n_out, dtype=torch.float32, device=device,
    )
    gemm_sqr_buf = torch.zeros(
        1, num_tokens, dtype=torch.float32, device=device,
    )
    residual_out_buf = torch.zeros(
        num_tokens, hc, hidden, dtype=dtype, device=device,
    )

    ref_mul, ref_sqr, ref_res = m.forward(
        comb_mix, residual_in, post_mix, x_in, weight_t,
    )

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
    w_dt = pk.attach_input(weight_t, name="weight_t")

    with pk.compile_scope():
        _ = m.compile(
            comb_dt, res_in_dt, post_dt, x_dt, w_dt,
            gemm_out_mul=gemm_mul_buf,
            gemm_out_sqrsum=gemm_sqr_buf,
            residual_out=residual_out_buf,
        )

    print("Compiling V4MhcFused test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4MhcFused test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"residual_out[0, 0, :8]: {residual_out_buf[0, 0, :8]}")
    print(f"ref_res[0, 0, :8]:      {ref_res[0, 0, :8]}")
    print(f"gemm_out_mul[0, 0, :4]: {gemm_mul_buf[0, 0, :4]}")
    print(f"ref_mul[0, 0, :4]:      {ref_mul[0, 0, :4]}")
    print(f"gemm_out_sqrsum[0]: {gemm_sqr_buf[0]}")
    print(f"ref_sqr[0]:         {ref_sqr[0]}")

    diff_res = (residual_out_buf.float() - ref_res.float()).abs().max().item()
    diff_mul = (gemm_mul_buf - ref_mul).abs().max().item()
    diff_sqr = (gemm_sqr_buf - ref_sqr).abs().max().item()
    # Relative compare on sqr/mul (they grow with HIDDEN).
    rel_mul = diff_mul / (ref_mul.abs().max().item() + 1e-6)
    rel_sqr = diff_sqr / (ref_sqr.abs().max().item() + 1e-6)
    print(
        f"max-abs diff: residual_out={diff_res}, "
        f"gemm_out_mul={diff_mul} (rel={rel_mul:.4g}), "
        f"gemm_out_sqrsum={diff_sqr} (rel={rel_sqr:.4g})"
    )

    try:
        torch.testing.assert_close(
            residual_out_buf, ref_res, atol=0.5, rtol=0.05,
        )
        # Large reductions over HIDDEN*HC: allow generous tolerance.
        torch.testing.assert_close(
            gemm_mul_buf, ref_mul, atol=0.5, rtol=0.05,
        )
        torch.testing.assert_close(
            gemm_sqr_buf, ref_sqr, atol=1.0, rtol=0.02,
        )
        print("PASSED: V4MhcFused compile() matches forward().")
    except AssertionError as e:
        print(f"FAILED: V4MhcFused disagrees with reference\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_mhc_fused_v4_testmode()
