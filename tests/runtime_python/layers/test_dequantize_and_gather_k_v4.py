"""V4-Flash ``dequantize_and_gather_k`` (Triton variant; NEW kernel) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/dequantize_and_gather_k_kernel.md``.

Producer is :class:`V4QuantizeAndInsertK`. Multi-batch with
``max_num_batched_requests = 4``.
"""
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.attention import (
    V4DequantizeAndGatherK,
    V4QuantizeAndInsertK,
)


def _build_packed_cache(
    num_reqs: int,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_stride: int,
    cache_block_size: int,
    device: str,
):
    """Use V4QuantizeAndInsertK.forward to populate the cache from random K."""
    max_tokens = int(seq_lens.max().item())
    # Place each request's tokens into its first blocks via block_table.
    num_blocks = int(block_table.max().item()) + 1
    k_cache = torch.zeros(num_blocks, block_stride, dtype=torch.uint8, device=device)

    qik = V4QuantizeAndInsertK(block_stride=block_stride)
    # Build the source K and slot_mapping vectors.
    total_tokens = int(seq_lens.sum().item())
    k_src = torch.randn(total_tokens, 512, dtype=torch.bfloat16, device=device)

    slot_list = []
    for b in range(num_reqs):
        slen = int(seq_lens[b].item())
        for pos in range(slen):
            block_in_seq = pos // cache_block_size
            pos_in_block = pos % cache_block_size
            phys = int(block_table[b, block_in_seq].item())
            slot_list.append(phys * cache_block_size + pos_in_block)
    slot = torch.tensor(slot_list, dtype=torch.int64, device=device)
    qik.forward(k_src, slot, k_cache)

    # The "true" dequantized values per token (for comparison).
    return k_cache, k_src


def test_dequantize_and_gather_k_v4_testmode():
    device = "cuda"
    torch.manual_seed(0)

    num_reqs = 4
    cache_block_size = 64
    block_stride = cache_block_size * 576 + cache_block_size * 8  # 37376

    # Per-request seq_lens (in tokens, <= cache_block_size for simplicity).
    seq_lens = torch.tensor([8, 12, 4, 16], dtype=torch.int32, device=device)
    max_blocks_per_seq = 2
    # block_table: each request maps logical block 0 to a distinct physical block.
    block_table = torch.tensor(
        [[1, 0], [2, 0], [3, 0], [4, 0]],
        dtype=torch.int32, device=device,
    )

    k_cache, k_src = _build_packed_cache(
        num_reqs, seq_lens, block_table,
        block_stride, cache_block_size, device,
    )

    # Per-request gather_lens (use smaller window).
    gather_lens = torch.tensor([8, 6, 4, 8], dtype=torch.int32, device=device)
    M = int(seq_lens.max().item()) + 4  # max output rows per req
    out_buf = torch.zeros(num_reqs, M, 576, dtype=torch.bfloat16, device=device)
    offset = 0

    m = V4DequantizeAndGatherK(
        block_stride=block_stride,
        cache_block_size=cache_block_size,
        prefix="v4_deqk_",
    )

    # Reference.
    out_ref = out_buf.clone()
    m.forward(out_ref, k_cache, seq_lens, block_table, gather_lens, offset=offset)

    # PersistentKernel test mode.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = M * num_reqs
    params["max_num_batched_requests"] = num_reqs
    pk = PersistentKernel(**params)

    k_cache_dt = pk.attach_input(k_cache, name="k_cache")
    seq_dt = pk.attach_input(seq_lens, name="seq_lens")
    gl_dt = pk.attach_input(gather_lens, name="gather_lens")
    bt_dt = pk.attach_input(block_table, name="block_table")
    out_dt = pk.attach_input(out_buf, name="out")

    with pk.compile_scope():
        _ = m.compile(
            k_cache_dt, seq_dt, gl_dt, bt_dt, out_dt,
            offset=offset, max_blocks_per_seq=max_blocks_per_seq,
        )

    print("Compiling V4DequantizeAndGatherK test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4DequantizeAndGatherK test kernel...")
    pk()
    torch.cuda.synchronize()

    # Compare only the first 512 lanes (the kernel writes that range).
    out_kernel_512 = out_buf[..., :512].float()
    out_ref_512 = out_ref[..., :512].float()
    max_diff = (out_kernel_512 - out_ref_512).abs().max().item()
    print(f"out[:, :, :512] max-abs diff: {max_diff}")

    try:
        torch.testing.assert_close(
            out_kernel_512, out_ref_512, atol=0.05, rtol=0.05
        )
        print("PASSED: V4DequantizeAndGatherK compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: {e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_dequantize_and_gather_k_v4_testmode()
