"""Catalog test: ``layers.attention.InvRopeFP8QuantO`` via PersistentKernel
test_mode.

Validates the V4-Flash MLA post-attention fused inverse-RoPE + per-block
FP8e4m3fn quantization kernel. Compares the compiled
``inv_rope_fp8_quant_o_sm100`` task output against the PyTorch reference
implemented inside :meth:`InvRopeFP8QuantO.forward`.
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def _dequantize(o_fp8: torch.Tensor, o_scale: torch.Tensor,
                block_size: int) -> torch.Tensor:
    """Re-multiply FP8 outputs by their per-block fp32 scales.

    ``o_fp8``   : ``[T, H, D]`` fp8 (or uint8 view of fp8 bits).
    ``o_scale`` : ``[T, H, D / block_size]`` fp32.
    """
    if o_fp8.dtype != torch.float8_e4m3fn:
        o_fp8 = o_fp8.view(torch.float8_e4m3fn)
    T, H, D = o_fp8.shape
    num_blocks = D // block_size
    blocks = o_fp8.float().reshape(T, H, num_blocks, block_size)
    return (blocks * o_scale.unsqueeze(-1)).reshape(T, H, D)


def test_inv_rope_fp8_quant_o_testmode():
    device = "cuda"
    torch.manual_seed(0)

    # Small shapes per the task spec.
    T = 4
    H = 4
    head_dim = 128
    rope_dim = 32
    block_size = 64
    max_pos = 16

    num_blocks = head_dim // block_size
    half_rope = rope_dim // 2

    # Build inputs on the test device.
    o = torch.randn(T, H, head_dim, dtype=torch.bfloat16, device=device) * 0.5
    # cos_sin_cache: pack cos in [:, :half_rope] and sin in [:, half_rope:].
    pos_freqs = torch.arange(max_pos, dtype=torch.float32, device=device)[:, None] * \
        (1.0 / (10000.0 ** (
            torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim
        )))
    cos_half = torch.cos(pos_freqs).to(torch.bfloat16)
    sin_half = torch.sin(pos_freqs).to(torch.bfloat16)
    cos_sin_cache = torch.cat([cos_half, sin_half], dim=-1).contiguous()
    assert cos_sin_cache.shape == (max_pos, rope_dim)
    positions = torch.arange(T, dtype=torch.int32, device=device)

    # Output buffers — pre-allocated so the test can read them back after pk().
    o_fp8 = torch.zeros(T, H, head_dim, dtype=torch.uint8, device=device)
    o_scale = torch.zeros(T, H, num_blocks, dtype=torch.float32, device=device)

    # PyTorch reference.
    m = layers.InvRopeFP8QuantO(
        num_heads=H,
        head_dim=head_dim,
        rope_dim=rope_dim,
        block_size=block_size,
        prefix="t_",
    )
    ref_o_fp8, ref_o_scale = m.forward(o, cos_sin_cache, positions)

    # MPK compile + run.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = T
    params["max_num_batched_requests"] = T
    pk = PersistentKernel(**params)

    o_dt = pk.attach_input(o, name="o")

    with pk.compile_scope():
        m.compile(
            o_dt,
            cos_sin_cache,
            positions,
            o_fp8=o_fp8,
            o_scale=o_scale,
        )

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    # Compare dequantized outputs since direct FP8 byte-equality is
    # too strict (the kernel's internal float math may differ slightly
    # from PyTorch's reference under round-to-nearest).
    deq_actual = _dequantize(o_fp8, o_scale, block_size)
    deq_ref = _dequantize(ref_o_fp8, ref_o_scale, block_size)

    try:
        # Loose tolerance for FP8: per-block-scaled E4M3 has ~2-3 bit
        # mantissa precision, so a ~0.1 absolute tolerance is appropriate
        # for the unit test.
        torch.testing.assert_close(
            deq_actual, deq_ref, atol=0.1, rtol=0.1
        )
        # Scales should be very close (they're plain absmax/448 over the
        # same fp32-promoted block elements).
        torch.testing.assert_close(
            o_scale, ref_o_scale, atol=1e-3, rtol=1e-3
        )
        print("PASSED: InvRopeFP8QuantO compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: InvRopeFP8QuantO compile() disagrees with forward()\n{e}")
        print(f"  o_scale (actual):\n{o_scale}")
        print(f"  o_scale (ref):\n{ref_o_scale}")
        print(f"  deq diff max: "
              f"{(deq_actual - deq_ref).abs().max().item()}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_inv_rope_fp8_quant_o_testmode()
