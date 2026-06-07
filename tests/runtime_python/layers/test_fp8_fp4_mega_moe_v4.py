"""V4-Flash ``fp8_fp4_mega_moe`` test (multi-batch).

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fp8_fp4_mega_moe.md``.
Catalog: ``mirage.mpk.layers.deepseek_v4.moe.fp8_fp4_mega_moe.V4Fp8Fp4MegaMoe``.

Naive port -- uses bf16 weights instead of FP4/FP8 to keep the
correctness reference tractable.  The I/O contract (y [T, H] bf16)
is preserved.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.layers.deepseek_v4.moe.fp8_fp4_mega_moe import V4Fp8Fp4MegaMoe
from mirage.mpk.persistent_kernel import PersistentKernel


def test_v4_fp8_fp4_mega_moe():
    device = "cuda"
    torch.manual_seed(0)

    num_tokens = 4
    top_k = 2
    hidden_size = 32
    intermediate_size = 16
    num_experts = 4

    module = V4Fp8Fp4MegaMoe(
        num_tokens=num_tokens,
        top_k=top_k,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        activation_clamp=None,
    )

    a = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16, device=device) * 0.1
    w13 = torch.randn(num_experts, 2 * intermediate_size, hidden_size,
                       dtype=torch.bfloat16, device=device) * 0.1
    w2 = torch.randn(num_experts, hidden_size, intermediate_size,
                       dtype=torch.bfloat16, device=device) * 0.1
    # Each token's top_k experts are deterministic and within range.
    topk_idx = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0]],
        dtype=torch.int64,
        device=device,
    )
    topk_w = torch.tensor(
        [[0.6, 0.4], [0.5, 0.5], [0.7, 0.3], [0.4, 0.6]],
        dtype=torch.float32,
        device=device,
    )

    ref = module.forward(a, w13, w2, topk_idx, topk_w)

    y_buf = torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = max(num_tokens, 8)
    params["max_num_batched_requests"] = max(num_tokens, 8)
    pk = PersistentKernel(**params)

    a_dt = pk.attach_input(a, name="mm_a")
    w13_dt = pk.attach_input(w13, name="mm_w13")
    w2_dt = pk.attach_input(w2, name="mm_w2")
    ti_dt = pk.attach_input(topk_idx, name="mm_topkidx")
    tw_dt = pk.attach_input(topk_w, name="mm_topkw")
    y_dt = pk.attach_input(y_buf, name="mm_y")

    with pk.compile_scope():
        _ = module.compile(a_dt, w13_dt, w2_dt, ti_dt, tw_dt, y_dt)

    print("Compiling V4Fp8Fp4MegaMoe...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4Fp8Fp4MegaMoe...")
    pk()
    torch.cuda.synchronize()

    print(f"y_buf[0,:8]: {y_buf[0, :8]}")
    print(f"ref[0,:8]:   {ref[0, :8]}")
    max_diff = (y_buf.float() - ref.float()).abs().max().item()
    print(f"Max absolute difference: {max_diff}")

    try:
        torch.testing.assert_close(y_buf, ref, atol=0.05, rtol=0.05)
        print("PASSED: V4Fp8Fp4MegaMoe matches reference.")
    except AssertionError as e:
        print(f"FAILED: V4Fp8Fp4MegaMoe disagrees\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_v4_fp8_fp4_mega_moe()
