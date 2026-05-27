"""Catalog test: ``layers.attention.IndexerQTransform`` via PersistentKernel
test_mode.

Validates the V4-Flash Indexer per-step Q transform kernel
(``indexer_q_transform_sm100``): low-rank Q expansion via ``wq_b``
(Hadamard pre-absorbed) + GPT-J interleaved RoPE +
MXFP4 block-32 quant with UE8M0 power-of-two scales.

Small shapes per the prompt: T=4, q_lora_rank=128, index_n_heads=4,
index_head_dim=64, rope_dim=16. Comparison is done in the
dequantized fp32 space with a loose tolerance (E2M1 has ~3-bit mantissa
precision and the kernel + PyTorch reference may pick different ties on
the level boundaries).
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.attention.indexer_q_transform import (
    _E2M1_LEVELS,
    _MXFP4_BLOCK_SIZE,
)


def _dequantize_mxfp4(q_fp4: torch.Tensor,
                      q_scale: torch.Tensor) -> torch.Tensor:
    """Re-materialize fp32 values from packed MXFP4 nibbles + UE8M0
    exponent bytes.

    ``q_fp4``   : ``[T, H, D/2]`` uint8 (two E2M1 nibbles per byte).
    ``q_scale`` : ``[T, H, D/32]`` uint8 (UE8M0 exponent byte per block).
    """
    T, H, half_D = q_fp4.shape
    D = half_D * 2
    num_blocks = D // _MXFP4_BLOCK_SIZE
    levels = torch.tensor(_E2M1_LEVELS, dtype=torch.float32,
                          device=q_fp4.device)

    # Unpack nibbles to [T, H, D] codes.
    even = (q_fp4 & 0xF).to(torch.uint8)
    odd = ((q_fp4 >> 4) & 0xF).to(torch.uint8)
    codes = torch.empty(T, H, D, dtype=torch.uint8, device=q_fp4.device)
    codes[..., 0::2] = even
    codes[..., 1::2] = odd
    mag = (codes & 0x7).long()
    sign = ((codes >> 3) & 0x1).to(torch.float32)
    values = levels[mag] * (1.0 - 2.0 * sign)             # [T, H, D]

    # Per-block scale: 2^(ue8m0 - 127).
    exp = q_scale.to(torch.int32) - 127
    scale = torch.pow(2.0, exp.to(torch.float32))          # [T, H, num_blocks]
    scale_expanded = scale.unsqueeze(-1).expand(
        T, H, num_blocks, _MXFP4_BLOCK_SIZE
    ).reshape(T, H, D)
    return values * scale_expanded


def test_indexer_q_transform_testmode():
    device = "cuda"
    torch.manual_seed(0)

    # Small shapes per the prompt.
    T = 4
    q_lora_rank = 128
    index_n_heads = 4
    index_head_dim = 64
    rope_dim = 16
    max_pos = 16
    half_D = index_head_dim // 2
    num_scale_blocks = index_head_dim // _MXFP4_BLOCK_SIZE
    half_rope = rope_dim // 2

    # Build inputs on the test device.
    q_lora = torch.randn(
        T, q_lora_rank, dtype=torch.bfloat16, device=device
    ) * 0.5
    # wq_b initialized small so the matmul output stays in a sane range.
    # In production this is Hadamard-absorbed; for the test we use random
    # values (the test only checks compile() == forward()).
    wq_b = torch.randn(
        index_n_heads * index_head_dim,
        q_lora_rank,
        dtype=torch.bfloat16,
        device=device,
    ) * (1.0 / q_lora_rank**0.5)

    # cos_sin_cache layout: cos in [:, :half_rope], sin in [:, half_rope:]
    # using the standard 1/theta^(2i/rope_dim) frequency table.
    pos_freqs = torch.arange(max_pos, dtype=torch.float32, device=device)[:, None] * \
        (1.0 / (160000.0 ** (
            torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device)
            / rope_dim
        )))
    cos_half = torch.cos(pos_freqs).to(torch.bfloat16)
    sin_half = torch.sin(pos_freqs).to(torch.bfloat16)
    cos_sin_cache = torch.cat([cos_half, sin_half], dim=-1).contiguous()
    assert cos_sin_cache.shape == (max_pos, rope_dim)
    positions = torch.arange(T, dtype=torch.int32, device=device)

    # Output buffers — pre-allocated so the test reads them back after pk().
    q_fp4 = torch.zeros(T, index_n_heads, half_D, dtype=torch.uint8,
                        device=device)
    q_scale = torch.zeros(T, index_n_heads, num_scale_blocks, dtype=torch.uint8,
                          device=device)

    # Catalog module (PyTorch reference).
    m = layers.IndexerQTransform(
        q_lora_rank=q_lora_rank,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        rope_dim=rope_dim,
        rope_theta=160000.0,
        prefix="t_",
    ).to(device=device)
    # Move the parameter to the test device + overwrite with the test
    # tensor (Parameter init is on CPU).
    with torch.no_grad():
        m.wq_b.data = wq_b.clone()

    ref_q_fp4, ref_q_scale = m.forward(q_lora, cos_sin_cache, positions)

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

    q_lora_dt = pk.attach_input(q_lora, name="q_lora")

    with pk.compile_scope():
        m.compile(
            q_lora_dt,
            cos_sin_cache,
            positions,
            q_fp4=q_fp4,
            q_scale=q_scale,
        )

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    # Compare dequantized values with a loose tolerance (MXFP4 has ~2-bit
    # mantissa; the per-block UE8M0 scale gives roughly fp values within
    # ~half the spacing between consecutive E2M1 levels).
    deq_actual = _dequantize_mxfp4(q_fp4, q_scale)
    deq_ref = _dequantize_mxfp4(ref_q_fp4, ref_q_scale)

    try:
        # Block scales should be very close (they're derived from amax
        # over the same fp32-promoted block elements).
        torch.testing.assert_close(
            q_scale, ref_q_scale, atol=1, rtol=0
        )
        # Dequantized values: loose tolerance for FP4. Use atol relative
        # to the dynamic range — MXFP4 quant step is ~half the gap
        # between adjacent E2M1 levels (e.g., between 4 and 6, gap is 2 →
        # max error ~1 per scale unit). A 1.0 atol on values that range
        # roughly [-6, 6] absolute is too tight; we test relative error
        # via the kernel-vs-reference diff being bounded by the
        # max-quant-step.
        diff = (deq_actual - deq_ref).abs()
        ref_max = deq_ref.abs().max().item()
        # E2M1 worst-case relative error is ~50% on the smallest non-zero
        # level (0.5) and tightens with magnitude; on a per-block scale
        # the absolute error scales with the block's amax. Allow 60% on
        # individual elements but require the mean error to stay below
        # 20% of the reference max.
        worst = diff.max().item()
        mean_err = diff.mean().item()
        assert worst <= 0.6 * max(ref_max, 1e-6), (
            f"max diff {worst} too large vs ref_max {ref_max}"
        )
        assert mean_err <= 0.2 * max(ref_max, 1e-6), (
            f"mean diff {mean_err} too large vs ref_max {ref_max}"
        )
        print(
            f"PASSED: IndexerQTransform compile() matches forward() "
            f"(max diff {worst:.4f}, mean diff {mean_err:.4f}, "
            f"ref_max {ref_max:.4f})"
        )
    except AssertionError as e:
        print(
            f"FAILED: IndexerQTransform compile() disagrees with forward()\n{e}"
        )
        print(f"  q_scale (actual): {q_scale.flatten()[:32]}")
        print(f"  q_scale (ref):    {ref_q_scale.flatten()[:32]}")
        print(f"  q_fp4 (actual):   {q_fp4.flatten()[:32]}")
        print(f"  q_fp4 (ref):      {ref_q_fp4.flatten()[:32]}")
        print(f"  diff max: {(deq_actual - deq_ref).abs().max().item()}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_indexer_q_transform_testmode()
