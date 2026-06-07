"""V4-Flash ``dsv3_router_gemm`` (NEW kernel) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/dsv3_router_gemm.md``.

Tier-3 / Tier-4 plain matmul: hidden_states @ weight.T -> fp32 logits.
bf16 in, bf16 weight, fp32 out.

Multi-batch from day 1 (``max_num_batched_requests = 4``).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.moe import V4Dsv3RouterGemm


def test_dsv3_router_gemm_v4_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # Small shape for test speed. V4-Flash uses H=4096, E=256.
    num_tokens = 4
    hidden_size = 128
    num_experts = 16

    m = V4Dsv3RouterGemm(
        hidden_size=hidden_size,
        num_experts=num_experts,
        prefix="v4_router_gemm_",
    )
    m.weight.data = m.weight.data.to(device=device, dtype=dtype)
    m.weight.data.copy_(
        torch.randn(num_experts, hidden_size, dtype=dtype, device=device) * 0.02
    )

    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=dtype, device=device
    )
    out_buf = torch.zeros(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )

    ref_out = m.forward(hidden_states)

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

    hidden_dt = pk.attach_input(hidden_states, name="hidden_states")

    with pk.compile_scope():
        _ = m.compile(hidden_dt, output=out_buf)

    print("Compiling V4Dsv3RouterGemm test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4Dsv3RouterGemm test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"out_buf[0, :8] = {out_buf[0, :8]}")
    print(f"ref_out[0, :8] = {ref_out[0, :8]}")
    max_diff = (out_buf - ref_out).abs().max().item()
    print(f"max-abs diff: {max_diff}")

    try:
        # bf16 input -> fp32 reduce: ~1% relative drift is typical
        # (reduction order differs between thread-strided + warp/cross-warp
        # reduce and PyTorch's sequential reduce).
        torch.testing.assert_close(out_buf, ref_out, atol=0.05, rtol=0.02)
        print("PASSED: V4Dsv3RouterGemm compile() matches forward()")
    except AssertionError as e:
        print(
            "FAILED: V4Dsv3RouterGemm compile() disagrees with forward()\n"
            f"{e}"
        )
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_dsv3_router_gemm_v4_testmode()
