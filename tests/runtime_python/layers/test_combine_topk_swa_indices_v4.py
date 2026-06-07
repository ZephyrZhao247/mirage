"""V4-Flash ``combine_topk_swa_indices`` (NEW kernel) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/combine_topk_swa_indices.md``.

Multi-batch from day 1: 4 query tokens across 2 batches.
"""
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.attention import V4CombineTopkSwaIndices
from mirage.mpk.layers.deepseek_v4.attention.combine_topk_swa_indices import (
    combined_topk_for,
)


def test_combine_topk_swa_indices_v4_testmode():
    device = "cuda"
    torch.manual_seed(0)

    num_tokens = 4
    top_k = 16
    compress_ratio = 4
    window_size = 8
    M = 128
    N = 32
    combined_topk = combined_topk_for(top_k, window_size)  # multiple of 128

    # Synthetic per-token state.
    topk_indices = torch.randint(
        -1, 64, (num_tokens, top_k), dtype=torch.int32, device=device
    )
    token_to_batch = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=device)
    positions = torch.tensor([6, 12, 4, 17], dtype=torch.int32, device=device)
    gather_start = torch.tensor([0, 4, 0, 9], dtype=torch.int32, device=device)

    m = V4CombineTopkSwaIndices(
        top_k=top_k,
        compress_ratio=compress_ratio,
        window_size=window_size,
        M=M, N=N,
        prefix="v4_csk_",
    )

    # Reference.
    ref_combined, ref_lens = m.forward(
        topk_indices, token_to_batch, positions, gather_start
    )

    # PersistentKernel.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = num_tokens
    params["max_num_batched_requests"] = max(2, num_tokens)
    pk = PersistentKernel(**params)

    # The kernel writes only the valid prefix; caller pre-fills -1.
    combined = torch.full(
        (num_tokens, combined_topk), -1, dtype=torch.int32, device=device
    )
    combined_lens = torch.zeros(num_tokens, dtype=torch.int32, device=device)

    ti_dt = pk.attach_input(topk_indices, name="topk_indices")
    ttb_dt = pk.attach_input(token_to_batch, name="token_to_batch")
    pos_dt = pk.attach_input(positions, name="positions")
    gs_dt = pk.attach_input(gather_start, name="gather_start")
    ci_dt = pk.attach_input(combined, name="combined_indices")
    cl_dt = pk.attach_input(combined_lens, name="combined_lens")

    with pk.compile_scope():
        _ = m.compile(ti_dt, ttb_dt, pos_dt, gs_dt, ci_dt,
                      combined_lens=cl_dt)

    print("Compiling V4CombineTopkSwaIndices test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4CombineTopkSwaIndices test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"out_lens: {combined_lens}")
    print(f"ref_lens: {ref_lens}")

    try:
        torch.testing.assert_close(combined, ref_combined)
        torch.testing.assert_close(combined_lens, ref_lens)
        print("PASSED: V4CombineTopkSwaIndices compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: {e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_combine_topk_swa_indices_v4_testmode()
