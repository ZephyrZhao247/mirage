"""
Test: DeepSeek V4-Flash mHC prenorm-GEMM via PersistentKernel test_mode.

Validates the decomposed v1 implementation of `mhc_prenorm_gemm_layer`
(see docs/mpk/deepseek_v4/hc.md §1).

Math (v1 contract, single split):
    residual: [N, hc, H] bf16
    fn:       [hc3, hc*H] bf16 (cast from fp32 at convert time)

    residual_flat = residual.reshape(N, hc*H)
    gemm_out_mul[n, j]   = sum_k residual_flat[n, k] * fn[j, k]    # bf16
    gemm_out_sqrsum[n]   = sum_k float(residual_flat[n, k])**2     # fp32

Reference snippet derived from `model.py:677-679` (mHC pre-norm half of
`hc_pre`) — see `docs/mpk/deepseek_v4/hc.md` §1.8.

Run:
    python tests/runtime_python/test_mode/test_mhc_prenorm_gemm_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


def torch_mhc_prenorm_gemm_ref(residual_bf16: torch.Tensor,
                               fn_bf16: torch.Tensor):
    """PyTorch oracle for mhc_prenorm_gemm.

    residual: [N, hc, H] bf16
    fn:       [hc3, hc*H] bf16

    Returns:
      gemm_out_mul    [N, hc3]   bf16 (bf16-bf16-bf16 matmul cast back)
      gemm_out_sqrsum [N]        fp32
    """
    n = residual_bf16.shape[0]
    x_flat = residual_bf16.reshape(n, -1)
    # bf16 in, fp32 accumulate, bf16 out (matches the kernel's accumulator dtype)
    gemm_out_mul = (x_flat.float() @ fn_bf16.float().T).to(torch.bfloat16)
    gemm_out_sqrsum = (x_flat.float() ** 2).sum(dim=-1)
    return gemm_out_mul, gemm_out_sqrsum


def test_mhc_prenorm_gemm_testmode():
    device = "cuda"
    torch.manual_seed(0)

    # Small test dims per docs/mpk/deepseek_v4/hc.md §1.8.
    N = 4
    HC = 4
    H = 128
    HC3 = (2 + HC) * HC  # 24

    print(f"\nTest: mhc_prenorm_gemm (decomposed v1)")
    print(f"  N={N}, hc={HC}, H={H}, hc3={HC3} (hc*H={HC*H})")

    # ---- Tensors ----
    # residual: [N, hc, H] bf16; flatten to 2D for MPK attach.
    residual_3d = torch.randn(N, HC, H, dtype=torch.bfloat16, device=device)
    residual_2d = residual_3d.reshape(N, HC * H).contiguous()
    # fn: cast from fp32 to bf16 at convert-time (v2 will keep fn in fp32).
    fn_fp32 = torch.randn(HC3, HC * H, dtype=torch.float32, device=device) * 0.05
    fn_bf16 = fn_fp32.to(torch.bfloat16).contiguous()

    # Outputs:
    #   gemm_out_mul    [N, hc3] bf16 (v1 dtype; v2 will widen to fp32)
    #   gemm_out_sqrsum [N]      fp32
    gemm_out_mul = torch.zeros(N, HC3, dtype=torch.bfloat16, device=device)
    gemm_out_sqrsum = torch.zeros(N, dtype=torch.float32, device=device)

    # PyTorch reference (matches the kernel's bf16/fp32 accumulator semantics).
    ref_mul, ref_sqrsum = torch_mhc_prenorm_gemm_ref(residual_3d, fn_bf16)

    # ---- Build PersistentKernel ----
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

    residual_dt = pk.attach_input(residual_2d, name="residual")
    fn_dt = pk.attach_input(fn_bf16, name="fn")
    mul_dt = pk.attach_input(gemm_out_mul, name="gemm_out_mul")
    sqrsum_dt = pk.attach_input(gemm_out_sqrsum, name="gemm_out_sqrsum")

    pk.mhc_prenorm_gemm_layer(
        residual=residual_dt,
        fn=fn_dt,
        gemm_out_mul=mul_dt,
        gemm_out_sqrsum=sqrsum_dt,
        n_splits=1,
    )

    # ---- Compile and run ----
    print("Compiling...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running...")
    pk.run_test_mode()
    torch.cuda.synchronize()

    # ---- Compare ----
    print(f"\ngemm_out_mul[0, :8]: {gemm_out_mul[0, :8]}")
    print(f"reference  [0, :8]: {ref_mul[0, :8]}")
    print(f"\ngemm_out_sqrsum: {gemm_out_sqrsum}")
    print(f"reference:        {ref_sqrsum}")

    mul_diff = (gemm_out_mul.float() - ref_mul.float()).abs().max().item()
    sqrsum_diff = (gemm_out_sqrsum - ref_sqrsum).abs().max().item()
    print(f"\nMax abs diff (gemm_out_mul, bf16):    {mul_diff:.6f}")
    print(f"Max abs diff (gemm_out_sqrsum, fp32): {sqrsum_diff:.6f}")

    ok = True
    if mul_diff > 5e-1:
        print(f"FAILED: gemm_out_mul diff {mul_diff} exceeds 5e-1")
        ok = False
    if sqrsum_diff > 5e-1:
        # Allow somewhat looser tolerance — bf16-cast inputs squared then
        # summed introduce noticeable accumulation error vs the fp32 oracle.
        print(f"FAILED: gemm_out_sqrsum diff {sqrsum_diff} exceeds 5e-1")
        ok = False

    if ok:
        print("\nPASSED: mhc_prenorm_gemm (decomposed v1) matches reference")
    else:
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_mhc_prenorm_gemm_testmode()
