"""Isolate the fp8_gemm_dense_smallm_sm100 scale-handling defect.

Drives kernel::fp8_gemm_dense_smallm::fp8_gemm_dense_smallm_sm100_task_impl
<128,3> directly (the exact codegen instantiation) with:
  (a) UNIFORM per-128x128-block weight scales  → should be ~perfect.
  (b) HIGH-VARIATION per-block weight scales (~100x across K blocks, like
      real DeepSeek weight_scale_inv)          → reproduces the failure.
Reference: pure-torch 128x128-block dequant + F.linear.
"""

import torch
import runtime_kernel_dense_gemm_smallm as ext

torch.manual_seed(0)
DEV = "cuda"
FP8_MAX = 448.0


def quant_block_scaled(w_bf16, scale):
    """Given a bf16 weight (N,K) and a chosen (N/128, K/128) f32 scale,
    produce the FP8 bytes so that dequant(fp8)*scale ~= w. Returns uint8.
    Vectorized."""
    N, K = w_bf16.shape
    nb, kb = N // 128, K // 128
    blocks = w_bf16.float().reshape(nb, 128, kb, 128)
    fp8 = (blocks / scale.reshape(nb, 1, kb, 1)).clamp(-FP8_MAX, FP8_MAX)
    return fp8.reshape(N, K).to(torch.float8_e4m3fn).view(torch.uint8)


def quant_act(x_bf16):
    """Per-row 1x128-group activation quant → (uint8, f32 scale (M,K/128)).
    Vectorized."""
    M, K = x_bf16.shape
    kb = K // 128
    blocks = x_bf16.float().reshape(M, kb, 128)
    amax = blocks.abs().amax(dim=2)                       # (M, kb)
    scale = (amax / FP8_MAX).clamp_min(1e-12)             # (M, kb)
    fp8 = (blocks / scale[:, :, None]).clamp(-FP8_MAX, FP8_MAX)
    fp8 = fp8.reshape(M, K).to(torch.float8_e4m3fn).view(torch.uint8)
    return fp8, scale.contiguous()


def dequant_w(w_u8, scale):
    w = w_u8.view(torch.float8_e4m3fn).float()
    N, K = w.shape
    nb, kb = N // 128, K // 128
    return (w.reshape(nb, 128, kb, 128) *
            scale.reshape(nb, 1, kb, 1)).reshape(N, K)


def dequant_a(a_u8, sa):
    a = a_u8.view(torch.float8_e4m3fn).float()
    M, K = a.shape
    kb = K // 128
    return (a.reshape(M, kb, 128) * sa.reshape(M, kb, 1)).reshape(M, K)


def cosine(a, b):
    a = a.float().flatten()
    b = b.float().flatten()
    return torch.nn.functional.cosine_similarity(a, b, dim=0).item()


def run_case(name, M, N, K, weight_scale, num_workers=4):
    # The "true" bf16 weight we want to represent: random, then we quantize it
    # with the GIVEN per-block scale. Make the underlying value magnitude track
    # the scale so each block actually uses the FP8 dynamic range.
    nb, kb = N // 128, K // 128
    # underlying magnitude tracks the scale so each block uses the FP8 range.
    w_bf16 = (torch.randn(nb, 128, kb, 128, device=DEV)
              * (weight_scale.reshape(nb, 1, kb, 1) * 100.0)
              ).reshape(N, K).to(torch.bfloat16)
    w_u8 = quant_block_scaled(w_bf16, weight_scale)

    x_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device=DEV) * 0.5
    a_u8, sa = quant_act(x_bf16)

    # Reference: dequant both, F.linear.
    a_f = dequant_a(a_u8, sa)
    w_f = dequant_w(w_u8, weight_scale)
    ref = torch.matmul(a_f, w_f.t()).to(torch.bfloat16)

    out = torch.zeros(M, N, dtype=torch.bfloat16, device=DEV)
    ext.dense_gemm_smallm(a_u8, sa.contiguous(), w_u8,
                          weight_scale.contiguous(), out, num_workers)
    torch.cuda.synchronize()

    cos = cosine(out, ref)
    diff = (out.float() - ref.float()).abs()
    relmax = (diff.max() / max(ref.float().abs().max().item(), 1e-6)).item()
    on = out.float().norm().item()
    rn = ref.float().norm().item()
    print(f"[{name}] M={M} N={N} K={K}  cosine={cos:.6f}  rel_max={relmax:.5f}  "
          f"out_norm={on:.3f} ref_norm={rn:.3f}  ratio={on/max(rn,1e-6):.3f}")
    print(f"   out[0,:6]={out[0,:6].float().tolist()}")
    print(f"   ref[0,:6]={ref[0,:6].float().tolist()}")
    return cos


def make_var_scale(nb, kb):
    """~100x span across K blocks, like real DeepSeek weight_scale_inv."""
    var = torch.empty(nb, kb, dtype=torch.float32, device=DEV)
    for bi in range(nb):
        for ki in range(kb):
            var[bi, ki] = 2.0 ** (-17 + (ki % 8))
    return var


def main():
    results = {}

    # ---- small shape: (a) uniform, (b) high-variation ----
    M, N, K = 128, 256, 1024
    nb, kb = N // 128, K // 128
    uni = torch.full((nb, kb), 2.0**-12, dtype=torch.float32, device=DEV)
    results["small-uniform"] = run_case("small-uniform", M, N, K, uni)
    results["small-highvar"] = run_case(
        "small-highvar", M, N, K, make_var_scale(nb, kb))

    # ---- REAL DeepSeek dense-MLP shapes, high-variation scales ----
    # gate_up: N=2*I=36864, K=H=7168.  down: N=H=7168, K=I=18432.
    # Test both M=128 (full M-tile) AND M=1 (real decode shape) — the model
    # runs at M=mbt=1 (single decode token), which exercises the partial-M-tile
    # / OOB-masked-row path.
    H, I = 7168, 18432
    for Mr in (128, 1):
        for nw in (4, 1):
            for name, (Nr, Kr) in {
                "gate_up": (2 * I, H),
                "down":    (H, I),
            }.items():
                nb, kb = Nr // 128, Kr // 128
                results[f"{name}-M{Mr}-nw{nw}"] = run_case(
                    f"{name}-M{Mr}-nw{nw}", Mr, Nr, Kr,
                    make_var_scale(nb, kb), num_workers=nw)

    print()
    for k, v in results.items():
        print(f"  {k:20s} cosine={v:.6f}")
    ok = all(v > 0.999 for v in results.values())
    print("PASSED" if ok else "FAILED (some shape mis-handles scales)")
    import sys
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
