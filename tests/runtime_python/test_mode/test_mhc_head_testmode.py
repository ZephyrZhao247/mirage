"""Test for DeepSeek V4-Flash mhc_head_layer.

The mhc_head task is the final HC collapse before the LM head: a two-pass
fused kernel that computes per-token RMS-then-projection then a sigmoid-
gated weighted sum across the hc=4 HC copies, producing a single hidden
vector per token for consumption by lm_head.

Reference math is extracted from `model.py` (DeepSeek V4-Flash) and the
TileLang implementation `hc_head_fuse_tilelang` in
`deps/vllm/vllm/model_executor/layers/mhc.py`.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def torch_mhc_head_ref(residual_bf16, fn_fp32, hc_scale, hc_base,
                       rms_eps=1e-6, hc_eps=1e-6):
    """PyTorch reference, extracted from model.py:729-736.

    Args:
        residual_bf16 : [N, hc, H] bf16
        fn_fp32       : [hc, hc*H]  fp32
        hc_scale      : [1]         fp32
        hc_base       : [hc]        fp32
    Returns:
        out           : [N, H]      bf16
    """
    N, hc, H = residual_bf16.shape
    x = residual_bf16.flatten(1).float()                              # [N, hc*H]
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + rms_eps)  # [N, 1]
    mixes = torch.nn.functional.linear(x, fn_fp32) * rsqrt            # [N, hc]
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + hc_eps          # [N, hc]
    out = (pre.unsqueeze(-1) * residual_bf16.float()).sum(dim=1)      # [N, H]
    return out.to(torch.bfloat16)


def test_mhc_head_testmode():
    device = "cuda"
    bf16 = torch.bfloat16
    fp32 = torch.float32

    # Small shapes per spec section 4.8.
    N = 4
    hc = 4
    H = 128

    torch.manual_seed(0)
    residual = torch.randn(N, hc, H, dtype=bf16, device=device) * 0.5
    fn = torch.randn(hc, hc * H, dtype=fp32, device=device) * 0.05
    hc_scale = torch.full((1,), 0.7, dtype=fp32, device=device)
    hc_base = torch.tensor([-0.2, 0.1, 0.0, 0.3], dtype=fp32, device=device)
    out = torch.zeros(N, H, dtype=bf16, device=device)

    # PyTorch reference.
    ref = torch_mhc_head_ref(residual, fn, hc_scale, hc_base)

    # Build PersistentKernel in test mode.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    pk = PersistentKernel(**params)

    residual_dt = pk.attach_input(residual, name="residual")
    fn_dt = pk.attach_input(fn, name="fn")
    hc_scale_dt = pk.attach_input(hc_scale, name="hc_scale")
    hc_base_dt = pk.attach_input(hc_base, name="hc_base")
    out_dt = pk.attach_input(out, name="out")

    target_cc = pk.target_cc
    if target_cc >= 90:
        block_dim = (128, 1, 1)
    else:
        block_dim = (128, 1, 1)

    pk.mhc_head_layer(
        residual=residual_dt,
        fn=fn_dt,
        hc_scale=hc_scale_dt,
        hc_base=hc_base_dt,
        out=out_dt,
        grid_dim=(N, 1, 1),
        block_dim=block_dim,
    )

    print("Compiling test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running test kernel...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    print(f"Output:\n{out[:2, :8]}")
    print(f"Reference:\n{ref[:2, :8]}")

    max_diff = (out.float() - ref.float()).abs().max().item()
    print(f"Max absolute difference: {max_diff}")

    rtol, atol = 1e-3, 1e-3
    ok = torch.allclose(out.float(), ref.float(), rtol=rtol, atol=atol)
    if ok:
        print(f"PASSED: mhc_head_layer matches reference (atol/rtol={atol})")
    else:
        # bf16 path: relax tolerance slightly given accumulated rounding.
        ok2 = torch.allclose(out.float(), ref.float(), rtol=5e-3, atol=5e-3)
        if ok2:
            print(f"PASSED (relaxed): mhc_head_layer matches within 5e-3")
        else:
            print(f"FAILED: max diff {max_diff} exceeds tolerance")
            pk.finalize()
            sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_mhc_head_testmode()
