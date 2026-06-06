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
    # IMPORTANT (2026-06-06 regression guard): cast the module to bf16 — the
    # HF-faithful driver does ``DeepseekV3Model(cfg).to(dtype=torch.bfloat16)``,
    # which would silently downcast the fp32 128x128-block weight_scale params
    # to bf16. The fp8_gemm_dense_smallm kernel reads weight_scale as float*, so
    # a bf16 scale buffer makes each 4-byte read span TWO bf16 scale entries →
    # ~100x-1000x garbage (pre-fix: cosine ~0.03, output norm ~2600x too large).
    # DeepseekV3MLP._apply must restore fp32. This test FAILS pre-fix.
    m = DeepseekV3MLP(cfg, prefix="mlp_").to(device).to(dtype=torch.bfloat16)
    assert m.gate_up_scale.dtype == torch.float32, (
        f"weight_scale must stay fp32 after .to(bfloat16); got "
        f"{m.gate_up_scale.dtype}")

    # Weights with REALISTIC ~100x per-128x128-block scale VARIATION (like real
    # DeepSeek weight_scale_inv, 9e-6..1e-3). Uniform-scale weights (the old
    # randn*0.05) hid the bf16-scale bug because all blocks shared one scale.
    def _blockscaled(N, K, seed):
        g = torch.Generator(device=device).manual_seed(seed)
        nb, kb = N // 128, K // 128
        # per-(128x128)-block magnitude spanning ~100x: 2^(-17 + (bi+ki)%8).
        bi = torch.arange(nb, device=device)[:, None]
        ki = torch.arange(kb, device=device)[None, :]
        blk_mag = torch.pow(2.0, (-17 + (bi + ki) % 8).float())  # (nb,kb)
        w = (torch.randn(nb, 128, kb, 128, generator=g, device=device)
             * blk_mag[:, None, :, None] * 100.0).reshape(N, K).to(torch.bfloat16)
        return w

    gate = _blockscaled(I, H, 1)
    up = _blockscaled(I, H, 2)
    down = _blockscaled(H, I, 3)
    gate_up = torch.cat([gate, up], dim=0)  # (2I, H), rows [0:I]=gate, [I:2I]=up

    gu_fp8, gu_scale = _qfp8(gate_up)
    dn_fp8, dn_scale = _qfp8(down)
    print(f"weight_scale variation: gate_up {gu_scale.min():.2e}..{gu_scale.max():.2e} "
          f"(~{gu_scale.max()/gu_scale.min():.0f}x), down {dn_scale.min():.2e}..{dn_scale.max():.2e}")
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
    cos = torch.nn.functional.cosine_similarity(
        out.float().flatten(), ref.float().flatten(), dim=0).item()
    norm_ratio = out.float().norm().item() / max(ref.float().norm().item(), 1e-9)
    print(f"out[0,:6]: {out[0,:6]}")
    print(f"ref[0,:6]: {ref[0,:6]}")
    print(f"max abs {max_abs:.5f} rel {max_rel:.5f} cosine {cos:.6f} "
          f"norm_ratio {norm_ratio:.3f}")
    # cosine guard catches the bf16-scale bug (pre-fix cosine ~0.03, norm ~2600x);
    # rel guard catches fine-grained drift.
    ok = max_rel < 0.10 and cos > 0.999 and 0.9 < norm_ratio < 1.1
    print("PASSED" if ok else "FAILED")
    pk.finalize()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    test_dsv3_mlp_fp8()
