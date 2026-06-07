"""V4-Flash ``silu_and_mul_with_clamp`` (NEW kernel) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/silu_and_mul_with_clamp.md``.

Computes per-token fused SwiGLU + asymmetric clamp:
  out = silu(clamp(gate, max=L)) * clamp(up, -L, L)
bf16 in / bf16 out, fp32 internal math.

Multi-batch from day 1 (``max_num_batched_requests = 4``).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.moe import V4SiluAndMulWithClamp


def test_silu_and_mul_with_clamp_v4_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    num_tokens = 4
    intermediate_size = 256  # smaller than V4-Flash 2048 for test speed
    swiglu_limit = 10.0

    m = V4SiluAndMulWithClamp(
        intermediate_size=intermediate_size,
        swiglu_limit=swiglu_limit,
        prefix="v4_silu_clamp_",
    )

    # Use a wide-amplitude input so the clamp actually kicks in for some
    # elements; mix in a few extreme values to exercise the limit.
    gateup = torch.randn(num_tokens, 2 * intermediate_size, dtype=dtype, device=device) * 5.0
    gateup[0, 0] = torch.tensor(20.0, dtype=dtype)  # gate > limit
    gateup[0, intermediate_size] = torch.tensor(-20.0, dtype=dtype)  # up < -limit

    out_buf = torch.zeros(num_tokens, intermediate_size, dtype=dtype, device=device)

    # PyTorch reference.
    ref_out = m.forward(gateup)

    # Build PersistentKernel in test mode.
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

    gateup_dt = pk.attach_input(gateup, name="gateup")

    with pk.compile_scope():
        _ = m.compile(gateup_dt, output=out_buf)

    print("Compiling V4SiluAndMulWithClamp test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4SiluAndMulWithClamp test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"out_buf[0, :8] = {out_buf[0, :8]}")
    print(f"ref_out[0, :8] = {ref_out[0, :8]}")
    max_diff = (out_buf.float() - ref_out.float()).abs().max().item()
    print(f"max-abs diff: {max_diff}")

    try:
        # bf16 round-trip: ~0.02 atol is the typical bf16 elementwise envelope.
        torch.testing.assert_close(out_buf, ref_out, atol=0.05, rtol=0.05)
        print("PASSED: V4SiluAndMulWithClamp compile() matches forward()")
    except AssertionError as e:
        print(
            "FAILED: V4SiluAndMulWithClamp compile() disagrees with forward()\n"
            f"{e}"
        )
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_silu_and_mul_with_clamp_v4_testmode()
