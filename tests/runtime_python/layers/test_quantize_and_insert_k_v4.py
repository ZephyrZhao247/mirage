"""V4-Flash ``quantize_and_insert_k`` (NEW kernel) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/quantize_and_insert_k_kernel.md``.

Multi-batch from day 1 (``max_num_batched_requests = 4``).
"""
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.attention import V4QuantizeAndInsertK


def test_quantize_and_insert_k_v4_testmode():
    device = "cuda"
    torch.manual_seed(0)

    # Small but spec-compliant shape.
    num_tokens = 4
    block_stride = 64 * 576 + 64 * 8  # 37376 bytes -- the minimum
    num_blocks = 8

    # Build module.
    m = V4QuantizeAndInsertK(block_stride=block_stride, prefix="v4_qik_")

    # K input: T x 512 bf16.
    k = torch.randn(num_tokens, 512, dtype=torch.bfloat16, device=device) * 2.0

    # Slot mapping: place each token into a different block (block 1..4) at pos 0.
    slot_mapping = torch.tensor(
        [1 * 64 + 0, 2 * 64 + 3, -1, 5 * 64 + 12],
        dtype=torch.int64,
        device=device,
    )

    # K cache: num_blocks x block_stride uint8 (zero-initialized).
    k_cache_kernel = torch.zeros(
        num_blocks, block_stride, dtype=torch.uint8, device=device
    )
    k_cache_ref = k_cache_kernel.clone()

    # PyTorch reference (mutates k_cache_ref in place).
    m.forward(k, slot_mapping, k_cache_ref)

    # PersistentKernel test mode.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = num_tokens
    params["max_num_batched_requests"] = max(num_tokens, 2)
    pk = PersistentKernel(**params)

    k_dt = pk.attach_input(k, name="k")
    slot_dt = pk.attach_input(slot_mapping, name="slot_mapping")
    k_cache_dt = pk.attach_input(k_cache_kernel, name="k_cache")

    with pk.compile_scope():
        _ = m.compile(k_dt, slot_dt, k_cache_dt)

    print("Compiling V4QuantizeAndInsertK test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4QuantizeAndInsertK test kernel...")
    pk()
    torch.cuda.synchronize()

    diff = (k_cache_kernel.to(torch.int32) - k_cache_ref.to(torch.int32)).abs()
    max_diff = diff.max().item()
    nnz = (diff > 0).sum().item()
    print(f"k_cache max-abs byte diff: {max_diff}; bytes-differing: {nnz}")

    try:
        # Allow off-by-one rounding on FP8 quant + UE8M0 scale corner cases.
        assert max_diff <= 1, f"max byte diff {max_diff} exceeds tolerance 1"
        print("PASSED: V4QuantizeAndInsertK compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: {e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_quantize_and_insert_k_v4_testmode()
