"""V4-Flash ``mhc_pre_big_fuse_with_norm`` (NEW kernel) test.

Spec:
    ``docs/mpk/deepseek_v4/vllm_kernels/mhc_pre_big_fuse_with_norm_tilelang.md``.

Same outputs as :class:`V4MhcPreBigFuse`, with an extra fused second-pass
RMSNorm (with learnable ``norm_weight``) on top of the pre-mix-weighted
sum. The deliberate bf16 round-trip on ``y`` before the second-pass
sumsq matches the spec (``attn_norm(y.to(bf16))`` reference precision).

Multi-batch from day 1 (max_num_batched_requests = 4).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.hc import V4MhcPreBigFuseWithNorm


def test_mhc_pre_big_fuse_with_norm_v4_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    num_tokens = 4
    hc_mult = 4
    hidden = 4096
    hc_mult3 = hc_mult * (2 + hc_mult)  # 24
    n_splits = 2

    m = V4MhcPreBigFuseWithNorm(
        hidden_size=hidden, hc_mult=hc_mult, prefix="v4_mhc_pren_",
    )
    # Random bf16 norm_weight (default init is ones; randn exercises the
    # gamma scaling path).
    m.norm_weight.data = m.norm_weight.data.to(device=device, dtype=dtype)
    m.norm_weight.data.copy_(
        torch.randn(hidden, dtype=dtype, device=device)
    )

    gemm_out_mul = torch.randn(
        n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=device,
    ) * 0.1
    gemm_out_sqrsum = (
        torch.randn(n_splits, num_tokens, dtype=torch.float32, device=device)
        .abs() * 100.0 + 1.0
    )
    hc_scale = torch.rand(3, dtype=torch.float32, device=device) * 0.5 + 0.5
    hc_base = torch.randn(hc_mult3, dtype=torch.float32, device=device) * 0.1
    residual = torch.randn(
        num_tokens, hc_mult, hidden, dtype=dtype, device=device,
    )

    post_mix_buf = torch.zeros(
        num_tokens, hc_mult, dtype=torch.float32, device=device,
    )
    comb_mix_buf = torch.zeros(
        num_tokens, hc_mult * hc_mult, dtype=torch.float32, device=device,
    )
    layer_input_buf = torch.zeros(
        num_tokens, hidden, dtype=dtype, device=device,
    )

    ref_post, ref_comb, ref_li = m.forward(
        gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual,
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

    gemm_mul_dt = pk.attach_input(gemm_out_mul, name="gemm_out_mul")
    gemm_sqr_dt = pk.attach_input(gemm_out_sqrsum, name="gemm_out_sqrsum")
    hc_scale_dt = pk.attach_input(hc_scale, name="hc_scale")
    hc_base_dt = pk.attach_input(hc_base, name="hc_base")
    residual_dt = pk.attach_input(residual, name="residual")

    with pk.compile_scope():
        _ = m.compile(
            gemm_mul_dt, gemm_sqr_dt, hc_scale_dt, hc_base_dt, residual_dt,
            post_mix=post_mix_buf,
            comb_mix=comb_mix_buf,
            layer_input=layer_input_buf,
        )

    print("Compiling V4MhcPreBigFuseWithNorm test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4MhcPreBigFuseWithNorm test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"post_mix[0]: {post_mix_buf[0]}")
    print(f"ref_post[0]: {ref_post[0]}")
    print(f"comb_mix[0, :4]: {comb_mix_buf[0, :4]}")
    print(f"ref_comb[0, :4]: {ref_comb[0, :4]}")
    print(f"layer_input[0, :8]: {layer_input_buf[0, :8]}")
    print(f"ref_li[0, :8]:      {ref_li[0, :8]}")

    diff_post = (post_mix_buf - ref_post).abs().max().item()
    diff_comb = (comb_mix_buf - ref_comb).abs().max().item()
    diff_li = (layer_input_buf.float() - ref_li.float()).abs().max().item()
    print(f"max-abs diff: post={diff_post}, comb={diff_comb}, layer_input={diff_li}")

    try:
        torch.testing.assert_close(post_mix_buf, ref_post, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(comb_mix_buf, ref_comb, atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(layer_input_buf, ref_li, atol=0.1, rtol=0.05)
        print("PASSED: V4MhcPreBigFuseWithNorm compile() matches forward().")
    except AssertionError as e:
        print(f"FAILED: V4MhcPreBigFuseWithNorm disagrees with reference\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_mhc_pre_big_fuse_with_norm_v4_testmode()
