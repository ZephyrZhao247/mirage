"""Test: DeepseekV3MLP (FP8 dense) compile() vs its HF-faithful forward().

Bottom-up composite test (Stage 1 of the add-mpk-model workflow). Builds the
module standalone, fills its FP8 weights from quantized random bf16, computes
the PyTorch reference via ``module.forward``, then runs the MPK ``compile``
path in test_mode and compares.

Run:
    CUDA_VISIBLE_DEVICES=3 python tests/runtime_python/test_mode/test_dsv3_mlp_fp8_testmode.py
"""

import os
import sys
from types import SimpleNamespace

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.models.deepseek_v3.modeling import DeepseekV3MLP

FP8_MAX = 448.0


def _qfp8(w_bf16, block=128):
    """Quantize a bf16 weight (N, K) to FP8 E4M3 (uint8) + 128x128-block
    f32 scale (N//block, K//block) — the fp8_gemm_dense_smallm layout."""
    N, K = w_bf16.shape
    nb, kb = N // block, K // block
    w = w_bf16.float()
    scale = torch.empty(nb, kb, dtype=torch.float32, device=w.device)
    fp8 = torch.empty(N, K, dtype=torch.float8_e4m3fn, device=w.device)
    for bi in range(nb):
        for ki in range(kb):
            blk = w[bi * block:(bi + 1) * block, ki * block:(ki + 1) * block]
            amax = blk.abs().max().item()
            s = amax / FP8_MAX if amax > 0 else 1.0
            scale[bi, ki] = s
            fp8[bi * block:(bi + 1) * block, ki * block:(ki + 1) * block] = (
                (blk / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            )
    return fp8.view(torch.uint8), scale


def test_dsv3_mlp_fp8():
    device = "cuda"
    torch.manual_seed(0)
    mbt = 8
    H = 512          # hidden  (512/128=4, 4%4==0  → UE8M0 ok)
    I = 1024         # interm  (1024/128=8, 8%4==0 → UE8M0 ok)

    cfg = SimpleNamespace(hidden_size=H, intermediate_size=I)
    m = DeepseekV3MLP(cfg, prefix="mlp_").to(device)

    # Random reference weights, quantized into the module's FP8 params.
    gate = torch.randn(I, H, dtype=torch.bfloat16, device=device) * 0.05
    up = torch.randn(I, H, dtype=torch.bfloat16, device=device) * 0.05
    down = torch.randn(H, I, dtype=torch.bfloat16, device=device) * 0.05
    gate_up = torch.cat([gate, up], dim=0)  # (2I, H), rows [0:I]=gate, [I:2I]=up

    gu_fp8, gu_scale = _qfp8(gate_up)
    dn_fp8, dn_scale = _qfp8(down)
    with torch.no_grad():
        m.gate_up_weight.copy_(gu_fp8)
        m.gate_up_scale.copy_(gu_scale)
        m.down_weight.copy_(dn_fp8)
        m.down_scale.copy_(dn_scale)

    x = torch.randn(mbt, H, dtype=torch.bfloat16, device=device) * 0.5
    residual = torch.randn(mbt, H, dtype=torch.bfloat16, device=device) * 0.1

    ref = m.forward(x, residual)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True, num_workers=num_workers,
        num_local_schedulers=num_schedulers, mpi_rank=0, world_size=1,
        max_num_batched_tokens=mbt, max_num_batched_requests=mbt,
    )
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x, name="x")
    r_dt = pk.attach_input(residual, name="residual")
    out = torch.zeros(mbt, H, dtype=torch.bfloat16, device=device)
    out_dt = pk.attach_input(out, name="out")

    with pk.compile_scope():
        m.compile(x_dt, r_dt, output=out_dt)

    print("Compiling DeepseekV3MLP (FP8)...")
    pk.compile(output_dir=os.path.dirname(__file__))
    pk()
    torch.cuda.synchronize()

    if out.isnan().any() or out.isinf().any():
        print(f"FAILED: NaN/Inf in output (out[0,:8]={out[0,:8]})")
        pk.finalize()
        sys.exit(1)

    max_abs = (out.float() - ref.float()).abs().max().item()
    max_rel = max_abs / max(ref.float().abs().max().item(), 1e-6)
    print(f"out[0,:6]: {out[0,:6]}")
    print(f"ref[0,:6]: {ref[0,:6]}")
    print(f"max abs {max_abs:.5f} rel {max_rel:.5f}")
    ok = max_rel < 0.10
    print("PASSED" if ok else "FAILED")
    pk.finalize()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    test_dsv3_mlp_fp8()
