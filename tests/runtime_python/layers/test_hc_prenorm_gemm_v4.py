"""V4-Flash ``hc_prenorm_gemm`` (NEW kernel) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/hc_prenorm_gemm_tilelang.md``.

Computes per-token GEMM `x @ fn.T` and sum-of-squares `sum(x^2)` over
the K = HC_MULT * HIDDEN axis. Outputs are fp32 with shapes
`(1, T, HC_MULT3)` and `(1, T)` -- matching the n_splits=1 contract
expected by the downstream `mhc_pre_big_fuse_*` consumer.

Multi-batch from day 1 (``max_num_batched_requests = 4``).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.hc import V4HcPrenormGemm


def test_hc_prenorm_gemm_v4_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # Small shape to keep the naive kernel fast in test mode.
    # The vLLM spec is HC_MULT=4, HIDDEN=4096 -> K=16384, HC_MULT3=24.
    # We shrink HIDDEN for test cycle speed; the kernel is generic.
    num_tokens = 4
    hidden_size = 256
    hc_mult = 4
    hc_mult3 = hc_mult * (2 + hc_mult)  # 24
    K = hc_mult * hidden_size            # 1024

    # ------------------------------------------------------------------
    # Build module + tensors
    # ------------------------------------------------------------------
    m = V4HcPrenormGemm(
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        prefix="v4_hc_gemm_",
    )
    # fp32 fn weights (the reference stores hc_attn_fn / hc_ffn_fn as fp32).
    m.fn.data = m.fn.data.to(device=device, dtype=torch.float32)
    m.fn.data.copy_(
        torch.randn(hc_mult3, K, dtype=torch.float32, device=device) * 0.02
    )

    x = torch.randn(num_tokens, K, dtype=dtype, device=device)

    gemm_out_buf = torch.zeros(
        1, num_tokens, hc_mult3, dtype=torch.float32, device=device
    )
    sqrsum_buf = torch.zeros(
        1, num_tokens, dtype=torch.float32, device=device
    )

    # PyTorch reference (fp32 reduction; outputs fp32).
    ref_gemm, ref_sqr = m.forward(x)

    # ------------------------------------------------------------------
    # Build PersistentKernel in test mode
    # ------------------------------------------------------------------
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

    x_dt = pk.attach_input(x, name="x")

    with pk.compile_scope():
        _ = m.compile(
            x_dt,
            gemm_out=gemm_out_buf,
            sqrsum=sqrsum_buf,
        )

    # ------------------------------------------------------------------
    # Compile and run once
    # ------------------------------------------------------------------
    print("Compiling V4HcPrenormGemm test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4HcPrenormGemm test kernel...")
    pk()
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    print("=== gemm_out (fp32) ===")
    print(f"gemm_out_buf[0, 0, :8]: {gemm_out_buf[0, 0, :8]}")
    print(f"ref_gemm    [0, 0, :8]: {ref_gemm[0, 0, :8]}")
    max_g = (gemm_out_buf - ref_gemm).abs().max().item()
    print(f"gemm_out max-abs diff: {max_g}")

    print("=== sqrsum (fp32) ===")
    print(f"sqrsum_buf [0, :]: {sqrsum_buf[0, :]}")
    print(f"ref_sqr    [0, :]: {ref_sqr[0, :]}")
    max_s = (sqrsum_buf - ref_sqr).abs().max().item()
    print(f"sqrsum   max-abs diff: {max_s}")

    try:
        # bf16 input -> fp32 accumulator: allow ~1% relative drift.
        # The reference also uses fp32 accumulation from bf16 input,
        # but reduction order differs slightly between the kernel
        # (cross-thread tree reduce) and PyTorch (sequential).
        torch.testing.assert_close(gemm_out_buf, ref_gemm, atol=0.05, rtol=0.02)
        torch.testing.assert_close(sqrsum_buf, ref_sqr, atol=0.5, rtol=0.02)
        print("PASSED: V4HcPrenormGemm compile() matches forward()")
    except AssertionError as e:
        print(
            "FAILED: V4HcPrenormGemm compile() disagrees with forward()\n"
            f"{e}"
        )
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_hc_prenorm_gemm_v4_testmode()
