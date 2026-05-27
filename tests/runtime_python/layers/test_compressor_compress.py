"""Catalog test: ``layers.attention.CompressorCompress`` via PersistentKernel
test_mode.

Validates the V4-Flash Compressor compress kernel (Wave-2 sub-batch C2).
Compares the compiled ``compressor_compress_sm100`` task output against
the PyTorch reference implemented inside :meth:`CompressorCompress.forward`.
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def _slot_view(out_uint8: torch.Tensor, nope_dim: int, rope_dim: int,
               num_scale_blocks: int):
    """Decode the per-slot byte layout back into (fp8, bf16 rope, fp32 scale)."""
    fp8 = out_uint8[:, :nope_dim].contiguous().view(torch.float8_e4m3fn)
    rope_bytes = out_uint8[:, nope_dim:nope_dim + 2 * rope_dim].contiguous()
    rope = rope_bytes.view(torch.bfloat16).reshape(out_uint8.shape[0], rope_dim)
    scale_bytes = out_uint8[
        :,
        nope_dim + 2 * rope_dim: nope_dim + 2 * rope_dim + 4 * num_scale_blocks
    ].contiguous()
    scale = scale_bytes.view(torch.float32).reshape(
        out_uint8.shape[0], num_scale_blocks
    )
    return fp8, rope, scale


def _dequantize(fp8: torch.Tensor, scale: torch.Tensor,
                block_size: int) -> torch.Tensor:
    """Re-multiply FP8 outputs by their per-block fp32 scales."""
    if fp8.dtype != torch.float8_e4m3fn:
        fp8 = fp8.view(torch.float8_e4m3fn)
    T, nope_dim = fp8.shape
    num_blocks = nope_dim // block_size
    blocks = fp8.float().reshape(T, num_blocks, block_size)
    return (blocks * scale.unsqueeze(-1)).reshape(T, nope_dim)


def test_compressor_compress_testmode():
    device = "cuda"
    torch.manual_seed(0)

    # Small shapes per the task prompt.
    head_dim = 64
    rope_dim = 16
    compress_ratio = 4
    overlap = True
    rope_theta = 160000.0
    eps = 1e-6
    max_pos = 64

    window = (2 if overlap else 1) * compress_ratio
    nope_dim = head_dim - rope_dim

    # Boundary-triggering positions: any p with (p+1) % compress_ratio == 0.
    boundary_positions = torch.tensor(
        [3, 7, 11], dtype=torch.int32, device=device
    )
    T = boundary_positions.shape[0]
    B = T  # v1: one state-cache batch per compress-triggering token

    # Pre-fill state_cache with random bf16; layout [B, window, 2*head_dim].
    state_cache = (
        torch.randn(B, window, 2 * head_dim, dtype=torch.bfloat16, device=device)
        * 0.5
    )
    ape = (
        torch.randn(compress_ratio, head_dim, dtype=torch.bfloat16,
                    device=device)
        * 0.25
    )

    # cos_sin_cache built with compress_rope_theta = 160000.
    half_rope = rope_dim // 2
    freqs = (
        torch.arange(max_pos, dtype=torch.float32, device=device)[:, None]
        * (1.0 / (rope_theta ** (
            torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device)
            / rope_dim
        )))
    )
    cos_half = torch.cos(freqs).to(torch.bfloat16)
    sin_half = torch.sin(freqs).to(torch.bfloat16)
    cos_sin_cache = torch.cat([cos_half, sin_half], dim=-1).contiguous()
    assert cos_sin_cache.shape == (max_pos, rope_dim)

    # Catalog module + PyTorch reference output.
    m = layers.CompressorCompress(
        head_dim=head_dim,
        rope_dim=rope_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        rope_theta=rope_theta,
        eps=eps,
        prefix="t_",
    )
    # Randomise the norm weight so the test exercises the RMSNorm-weight path.
    with torch.no_grad():
        m.norm_weight.copy_(
            torch.randn(head_dim, dtype=torch.bfloat16) * 0.1 + 1.0
        )
    m_cuda = m.to(device)
    ref_out = m_cuda.forward(state_cache, ape, cos_sin_cache,
                              boundary_positions)
    assert ref_out.dtype == torch.uint8
    assert ref_out.shape == (T, m_cuda.slot_bytes)

    # Output buffer for the compiled kernel.
    actual_out = torch.zeros(T, m_cuda.slot_bytes, dtype=torch.uint8,
                              device=device)

    # MPK compile + run.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = max(T, 1)
    params["max_num_batched_requests"] = max(T, 1)
    pk = PersistentKernel(**params)

    state_cache_dt = pk.attach_input(state_cache, name="state_cache")

    with pk.compile_scope():
        m_cuda.compile(
            state_cache_dt,
            ape,
            cos_sin_cache,
            boundary_positions,
            actual_out,
        )

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    # Compare the decoded per-slot views.
    fp8_act, rope_act, scale_act = _slot_view(
        actual_out, nope_dim, rope_dim, m_cuda.num_scale_blocks
    )
    fp8_ref, rope_ref, scale_ref = _slot_view(
        ref_out, nope_dim, rope_dim, m_cuda.num_scale_blocks
    )

    deq_act = _dequantize(fp8_act, scale_act, m_cuda.block_size)
    deq_ref = _dequantize(fp8_ref, scale_ref, m_cuda.block_size)

    try:
        # Loose tolerance for FP8 paths.
        torch.testing.assert_close(
            deq_act, deq_ref, atol=0.1, rtol=0.1
        )
        # bf16 RoPE region should be tight (~3-bit mantissa rounding).
        torch.testing.assert_close(
            rope_act.float(), rope_ref.float(), atol=5e-2, rtol=5e-2
        )
        # Scales should be very close (deterministic max+divide).
        torch.testing.assert_close(
            scale_act, scale_ref, atol=1e-3, rtol=1e-3
        )
        print("PASSED: CompressorCompress compile() matches forward()")
    except AssertionError as e:
        print(
            f"FAILED: CompressorCompress compile() disagrees with forward()\n{e}"
        )
        print(f"  scale (actual):\n{scale_act}")
        print(f"  scale (ref):\n{scale_ref}")
        print(f"  rope diff max:    "
              f"{(rope_act.float() - rope_ref.float()).abs().max().item()}")
        print(f"  deq diff max:     "
              f"{(deq_act - deq_ref).abs().max().item()}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_compressor_compress_testmode()
