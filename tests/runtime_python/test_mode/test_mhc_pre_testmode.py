"""Test the mhc_pre_sm100 layer through the full MPK pipeline in test_mode.

Compares the kernel's outputs (post_mix, comb_mix, layer_input) against a
PyTorch oracle that implements the math directly from vLLM's
mhc_pre_big_fuse_tilelang.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def torch_mhc_pre_ref(
    gemm_out_mul,   # [splits, N, hc3] fp32
    gemm_out_sqrsum,  # [splits, N]     fp32
    hc_scale,       # [3]               fp32
    hc_base,        # [hc3]             fp32
    residual,       # [N, hc, H]        bf16
    rms_eps=1e-6,
    hc_pre_eps=1e-6,
    hc_sinkhorn_eps=1e-6,
    hc_post_mult_value=2.0,
    sinkhorn_iters=20,
):
    N, hc, H = residual.shape
    hc3 = (2 + hc) * hc
    assert gemm_out_mul.shape[-1] == hc3

    sqrsum_total = gemm_out_sqrsum.sum(0)                         # [N]
    rms = torch.rsqrt(sqrsum_total / (hc * H) + rms_eps)          # [N]
    mixes = gemm_out_mul.sum(0) * rms.unsqueeze(-1)               # [N, hc3]

    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + hc_pre_eps
    post = hc_post_mult_value * torch.sigmoid(
        mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc]
    )
    cm = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).reshape(N, hc, hc)

    # Sinkhorn: initial row-softmax + eps, then col-normalize; then 19 more
    # row/col iterations.
    cm = torch.softmax(cm, dim=-1) + hc_sinkhorn_eps
    cm = cm / (cm.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)
    for _ in range(sinkhorn_iters - 1):
        cm = cm / (cm.sum(dim=-1, keepdim=True) + hc_sinkhorn_eps)
        cm = cm / (cm.sum(dim=-2, keepdim=True) + hc_sinkhorn_eps)

    layer_input = (pre.unsqueeze(-1) * residual.float()).sum(dim=1).to(torch.bfloat16)
    return post.contiguous(), cm.contiguous(), layer_input.contiguous()


def test_mhc_pre_testmode():
    device = "cuda"
    torch.manual_seed(0)

    N = 4
    HC = 4
    H = 128
    HC3 = HC * (HC + 2)  # = 24
    SPLITS = 1

    # Build inputs. We populate gemm_out_* with the result of an honest prenorm
    # GEMM so that the rsqrt math doesn't hit a degenerate value.
    fn = torch.randn(HC3, HC * H, dtype=torch.float32, device=device) * 0.05
    residual = torch.randn(N, HC, H, dtype=torch.bfloat16, device=device)

    x_flat = residual.reshape(N, HC * H).float()  # [N, HC*H]
    gemm_out_mul = torch.nn.functional.linear(x_flat, fn).unsqueeze(0).contiguous()  # [1, N, HC3]
    gemm_out_sqrsum = x_flat.square().sum(-1).unsqueeze(0).contiguous()              # [1, N]

    hc_scale = torch.tensor([0.5, 0.5, 1.0], dtype=torch.float32, device=device)
    hc_base = torch.randn(HC3, dtype=torch.float32, device=device) * 0.1

    # Output buffers.
    post_mix = torch.zeros(N, HC, dtype=torch.float32, device=device)
    comb_mix = torch.zeros(N, HC, HC, dtype=torch.float32, device=device)
    layer_input = torch.zeros(N, H, dtype=torch.bfloat16, device=device)

    # PyTorch oracle.
    ref_post, ref_comb, ref_layer = torch_mhc_pre_ref(
        gemm_out_mul, gemm_out_sqrsum, hc_scale, hc_base, residual)

    # Build kernel.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    pk = PersistentKernel(**params)

    gm_dt = pk.attach_input(gemm_out_mul, name="gemm_out_mul")
    gs_dt = pk.attach_input(gemm_out_sqrsum, name="gemm_out_sqrsum")
    hs_dt = pk.attach_input(hc_scale, name="hc_scale")
    hb_dt = pk.attach_input(hc_base, name="hc_base")
    re_dt = pk.attach_input(residual, name="residual")
    pm_dt = pk.attach_input(post_mix, name="post_mix")
    cm_dt = pk.attach_input(comb_mix, name="comb_mix")
    li_dt = pk.attach_input(layer_input, name="layer_input")

    block_dim = (128, 1, 1)
    pk.mhc_pre_layer(
        gemm_out_mul=gm_dt,
        gemm_out_sqrsum=gs_dt,
        hc_scale=hs_dt,
        hc_base=hb_dt,
        residual=re_dt,
        post_mix=pm_dt,
        comb_mix=cm_dt,
        layer_input=li_dt,
        grid_dim=(N, 1, 1),
        block_dim=block_dim,
    )

    print("Compiling test kernel...")
    folder = os.path.dirname(__file__)
    pk.compile(output_dir=folder)

    print("Running test kernel...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    # Compare.
    post_diff = (post_mix - ref_post).abs().max().item()
    comb_diff = (comb_mix - ref_comb).abs().max().item()
    layer_diff = (layer_input.float() - ref_layer.float()).abs().max().item()
    print(f"post_mix max diff:    {post_diff:.6e}")
    print(f"comb_mix max diff:    {comb_diff:.6e}")
    print(f"layer_input max diff: {layer_diff:.6e}")
    print(f"ref post[0]: {ref_post[0]}")
    print(f"got post[0]: {post_mix[0]}")
    print(f"ref comb[0]:\n{ref_comb[0]}")
    print(f"got comb[0]:\n{comb_mix[0]}")

    ok_post = torch.allclose(post_mix, ref_post, rtol=1e-3, atol=1e-3)
    ok_comb = torch.allclose(comb_mix, ref_comb, rtol=1e-3, atol=1e-3)
    ok_layer = torch.allclose(layer_input.float(), ref_layer.float(),
                              rtol=1e-2, atol=1e-2)

    if not (ok_post and ok_comb and ok_layer):
        print(f"FAILED: ok_post={ok_post} ok_comb={ok_comb} ok_layer={ok_layer}")
        pk.finalize()
        sys.exit(1)

    print("PASSED: mhc_pre test_mode outputs match PyTorch oracle.")
    pk.finalize()


if __name__ == "__main__":
    test_mhc_pre_testmode()
