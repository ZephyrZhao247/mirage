"""V4-Flash ``fused_indexer_q_rope_quant`` (NEW FP8 Q-side kernel) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fused_indexer_q_rope_quant.md``.

Multi-batch from day 1 (``max_num_batched_requests = 4`` via
``num_rows >= 2``).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.indexer import V4FusedIndexerQRopeQuant


def test_fused_indexer_q_rope_quant_v4_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # V4-Flash: head_dim=128, half_rot_dim=32 (rot_dim=64, nope_dim=64).
    head_dim = 128
    half_rot_dim = 32
    rot_dim = 2 * half_rot_dim
    num_tokens = 4
    n_heads = 2
    num_rows = num_tokens * n_heads
    softmax_scale = head_dim ** -0.5    # 1/sqrt(128)
    head_scale = 0.125                  # 1/sqrt(64)

    # Inputs.
    q_in = torch.randn(num_rows, head_dim, dtype=dtype, device=device) * 0.5
    angles = torch.rand(num_rows, half_rot_dim, dtype=torch.float32, device=device) \
             * (2.0 * torch.pi)
    cos = angles.cos()
    sin = angles.sin()
    cos_sin = torch.cat([cos, sin], dim=-1)  # fp32 [num_rows, 2*half_rot_dim]
    weights_in = (
        torch.randn(num_rows, 1, dtype=dtype, device=device) * 0.5
    )

    # Output buffers.
    q_out_buf = torch.zeros(num_rows, head_dim, dtype=torch.float8_e4m3fn, device=device)
    w_out_buf = torch.zeros(num_rows, 1, dtype=torch.float32, device=device)

    # Module + reference.
    module = V4FusedIndexerQRopeQuant(
        head_dim=head_dim,
        half_rot_dim=half_rot_dim,
        softmax_scale=softmax_scale,
        head_scale=head_scale,
        prefix="v4_idxqq_",
    )
    ref_q, ref_w = module.forward(q_in, cos_sin, weights_in)

    # Build PK in test mode.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = num_rows
    params["max_num_batched_requests"] = max(num_rows, 4)
    pk = PersistentKernel(**params)

    q_dt = pk.attach_input(q_in, name="q_in")
    cs_dt = pk.attach_input(cos_sin, name="cos_sin")
    w_in_dt = pk.attach_input(weights_in, name="weights_in")

    with pk.compile_scope():
        _ = module.compile(
            q_dt, cs_dt, w_in_dt,
            q_out=q_out_buf,
            weights_out=w_out_buf,
        )

    print("Compiling V4FusedIndexerQRopeQuant test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4FusedIndexerQRopeQuant test kernel...")
    pk()
    torch.cuda.synchronize()

    # Compare via fp32 conversion.
    q_out_fp32 = q_out_buf.to(torch.float32)
    ref_q_fp32 = ref_q.to(torch.float32)
    print(f"q_out[0, :8] = {q_out_fp32[0, :8]}")
    print(f"ref_q[0, :8] = {ref_q_fp32[0, :8]}")
    print(f"w_out = {w_out_buf.flatten()}")
    print(f"ref_w = {ref_w.flatten()}")
    max_q = (q_out_fp32 - ref_q_fp32).abs().max().item()
    max_w = (w_out_buf - ref_w).abs().max().item()
    print(f"max q diff (fp8 cast): {max_q}")
    print(f"max w diff:            {max_w}")

    try:
        # FP8 e4m3 has ~3-bit mantissa; allow generous absolute tolerance
        # on the dequantized comparison.
        torch.testing.assert_close(q_out_fp32, ref_q_fp32, atol=0.5, rtol=0.1)
        torch.testing.assert_close(w_out_buf, ref_w, atol=1e-3, rtol=1e-3)
        print("PASSED: V4FusedIndexerQRopeQuant compile() matches forward()")
    except AssertionError as e:
        print(
            "FAILED: V4FusedIndexerQRopeQuant compile() disagrees with forward()\n"
            f"{e}"
        )
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_fused_indexer_q_rope_quant_v4_testmode()
