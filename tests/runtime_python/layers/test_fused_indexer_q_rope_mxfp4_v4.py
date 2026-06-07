"""V4-Flash ``fused_indexer_q_rope_mxfp4`` (NEW MXFP4 Q-side kernel) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fused_indexer_q_rope_mxfp4.md``.

Multi-batch from day 1 (``num_rows >= 2``).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.indexer import V4FusedIndexerQRopeMxfp4


def _e2m1_nibble_to_fp32(b: torch.Tensor) -> torch.Tensor:
    """Decode a uint8 E2M1 nibble (lower 4 bits) into fp32."""
    mags = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32, device=b.device,
    )
    sign = ((b & 0x8) != 0).to(torch.float32)
    mag = (b & 0x7).to(torch.int64)
    return torch.where(sign > 0, -mags[mag], mags[mag])


def test_fused_indexer_q_rope_mxfp4_v4_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    head_dim = 128
    half_rot_dim = 32
    mxfp4_block = 32
    num_blocks = head_dim // mxfp4_block
    num_tokens = 4
    n_heads = 2
    num_rows = num_tokens * n_heads
    softmax_scale = head_dim ** -0.5
    head_scale = 0.125

    q_in = torch.randn(num_rows, head_dim, dtype=dtype, device=device) * 0.5
    angles = torch.rand(num_rows, half_rot_dim, dtype=torch.float32, device=device) \
             * (2.0 * torch.pi)
    cos_sin = torch.cat([angles.cos(), angles.sin()], dim=-1)
    weights_in = torch.randn(num_rows, 1, dtype=dtype, device=device) * 0.5

    q_packed_buf = torch.zeros(num_rows, head_dim // 2, dtype=torch.uint8, device=device)
    q_scale_buf = torch.zeros(num_rows, num_blocks, dtype=torch.uint8, device=device)
    w_out_buf = torch.zeros(num_rows, 1, dtype=torch.float32, device=device)

    module = V4FusedIndexerQRopeMxfp4(
        head_dim=head_dim,
        half_rot_dim=half_rot_dim,
        mxfp4_block=mxfp4_block,
        softmax_scale=softmax_scale,
        head_scale=head_scale,
        prefix="v4_idxqm_",
    )
    ref_packed, ref_scale, ref_w = module.forward(q_in, cos_sin, weights_in)

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
            q_packed=q_packed_buf,
            q_scale=q_scale_buf,
            weights_out=w_out_buf,
        )

    print("Compiling V4FusedIndexerQRopeMxfp4 test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4FusedIndexerQRopeMxfp4 test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"q_packed[0, :8]: {q_packed_buf[0, :8].tolist()}")
    print(f"ref_packed[0, :8]: {ref_packed[0, :8].tolist()}")
    print(f"q_scale[0]: {q_scale_buf[0].tolist()}")
    print(f"ref_scale[0]: {ref_scale[0].tolist()}")
    print(f"w_out: {w_out_buf.flatten()}")
    print(f"ref_w: {ref_w.flatten()}")

    # Dequantize both kernel and reference for a softer numeric compare.
    def dequant(packed: torch.Tensor, scale_bytes: torch.Tensor) -> torch.Tensor:
        lo = _e2m1_nibble_to_fp32(packed & 0xF)
        hi = _e2m1_nibble_to_fp32((packed >> 4) & 0xF)
        # Interleave so lane order matches q_full.
        out = torch.zeros(num_rows, head_dim, dtype=torch.float32, device=device)
        out[:, 0::2] = lo
        out[:, 1::2] = hi
        # Apply per-block scale.
        log2_ratio = scale_bytes.to(torch.float32) - 127.0
        block_scale = torch.exp2(log2_ratio)             # [N, num_blocks]
        # broadcast over the block axis
        out_b = out.view(num_rows, num_blocks, mxfp4_block) * block_scale.unsqueeze(-1)
        return out_b.view(num_rows, head_dim)

    kernel_dq = dequant(q_packed_buf, q_scale_buf)
    ref_dq = dequant(ref_packed, ref_scale)
    max_q = (kernel_dq - ref_dq).abs().max().item()
    max_w = (w_out_buf - ref_w).abs().max().item()
    print(f"max dequant q diff: {max_q}")
    print(f"max w diff:          {max_w}")

    try:
        # MXFP4 quant boundary cases may flip a nibble around the bucket
        # boundary; allow generous absolute tolerance on the dequantized
        # comparison.
        torch.testing.assert_close(kernel_dq, ref_dq, atol=0.5, rtol=0.1)
        torch.testing.assert_close(w_out_buf, ref_w, atol=1e-3, rtol=1e-3)
        print("PASSED: V4FusedIndexerQRopeMxfp4 compile() matches forward()")
    except AssertionError as e:
        print(
            "FAILED: V4FusedIndexerQRopeMxfp4 compile() disagrees with forward()\n"
            f"{e}"
        )
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_fused_indexer_q_rope_mxfp4_v4_testmode()
