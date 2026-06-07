"""V4-Flash ``prepare_megamoe_inputs`` (NEW kernel) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/prepare_megamoe_inputs.md``.

Per-token bf16 -> FP8 E4M3 quantization with UE8M0 packed-int32 group
scales (BLOCK_K=128, GROUP_K=32) + topk repack (int32 -> int64, fp32
byte-copy).

Multi-batch from day 1 (``max_num_batched_requests = 4``).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.moe import V4PrepareMegaMoEInputs


def test_prepare_megamoe_inputs_v4_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # V4-Flash production: H=7168, K=8. Use H=256, K=4 here for test speed
    # (still exercises 2 BLOCK_K=128 chunks and the topk repack).
    num_tokens = 4
    hidden_size = 256
    topk = 4

    m = V4PrepareMegaMoEInputs(
        hidden_size=hidden_size, topk=topk, prefix="v4_megamoe_prep_"
    )

    hidden_states = torch.randn(
        num_tokens, hidden_size, dtype=dtype, device=device
    ) * 2.0
    topk_ids = torch.randint(
        0, 64, (num_tokens, topk), dtype=torch.int32, device=device
    )
    topk_weights = torch.randn(
        num_tokens, topk, dtype=torch.float32, device=device
    )

    num_blocks = hidden_size // m.BLOCK_K
    x_fp8_buf = torch.zeros(
        num_tokens, hidden_size, dtype=torch.float8_e4m3fn, device=device
    )
    x_sf_buf = torch.zeros(
        num_tokens, num_blocks, dtype=torch.int32, device=device
    )
    topk_idx_out_buf = torch.zeros(
        num_tokens, topk, dtype=torch.int64, device=device
    )
    topk_weights_out_buf = torch.zeros(
        num_tokens, topk, dtype=torch.float32, device=device
    )

    ref_x_fp8, ref_x_sf, ref_topk_idx, ref_topk_w = m.forward(
        hidden_states, topk_ids, topk_weights
    )

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
    topk_ids_dt = pk.attach_input(topk_ids, name="topk_ids")
    topk_weights_dt = pk.attach_input(topk_weights, name="topk_weights")

    with pk.compile_scope():
        _ = m.compile(
            hidden_dt,
            topk_ids_dt,
            topk_weights_dt,
            x_fp8=x_fp8_buf,
            x_sf=x_sf_buf,
            topk_idx_out=topk_idx_out_buf,
            topk_weights_out=topk_weights_out_buf,
        )

    print("Compiling V4PrepareMegaMoEInputs test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4PrepareMegaMoEInputs test kernel...")
    pk()
    torch.cuda.synchronize()

    # Compare x_sf (int32 packed UE8M0): must match bit-for-bit.
    sf_match = (x_sf_buf == ref_x_sf).all().item()
    print(f"x_sf bit-match: {sf_match}")

    # Compare topk_idx_out (int64 cast).
    idx_match = (topk_idx_out_buf == ref_topk_idx).all().item()
    print(f"topk_idx_out bit-match: {idx_match}")

    # Compare topk_weights_out (fp32 byte-copy).
    w_match = (topk_weights_out_buf == ref_topk_w).all().item()
    print(f"topk_weights_out bit-match: {w_match}")

    # Compare x_fp8 via dequant: cast both to fp32 (fp8 -> fp32 is exact).
    x_fp8_f = x_fp8_buf.to(torch.float32)
    ref_x_fp8_f = ref_x_fp8.to(torch.float32)
    fp8_diff = (x_fp8_f - ref_x_fp8_f).abs().max().item()
    print(f"x_fp8 max-abs diff (fp32-cast): {fp8_diff}")

    failed = False
    if not sf_match:
        print(f"x_sf MISMATCH: buf={x_sf_buf[0]}, ref={ref_x_sf[0]}")
        failed = True
    if not idx_match:
        print(f"topk_idx MISMATCH: buf={topk_idx_out_buf[0]}, ref={ref_topk_idx[0]}")
        failed = True
    if not w_match:
        print(f"topk_weights MISMATCH: buf={topk_weights_out_buf[0]}, ref={ref_topk_w[0]}")
        failed = True
    if fp8_diff > 0.0:
        # fp8 quant: even bit-identical scales can quantize bf16 inputs to
        # one ULP apart between Triton and CUDA (Triton uses round-to-nearest-
        # even via tl.cast; CUDA __nv_fp8_e4m3(float) does likewise but the
        # round-up rule on ties may differ on edge cases). Tolerate <= 0.5
        # absolute (one fp8 LSB at amax ~ 1.0).
        if fp8_diff > 0.5:
            print(
                f"x_fp8 mismatch too large: diff={fp8_diff}\n"
                f"  buf [0, :8]: {x_fp8_f[0, :8]}\n"
                f"  ref [0, :8]: {ref_x_fp8_f[0, :8]}"
            )
            failed = True
        else:
            print(
                "x_fp8 within fp8 ULP tolerance "
                f"(diff={fp8_diff:.4f} <= 0.5)"
            )

    if failed:
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("PASSED: V4PrepareMegaMoEInputs compile() matches forward()")
    print("Test completed successfully!")


if __name__ == "__main__":
    test_prepare_megamoe_inputs_v4_testmode()
