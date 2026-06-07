"""V4-Flash ``compute_global_topk_indices_and_lens`` (NEW kernel) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/compute_global_topk_indices_and_lens.md``.

Multi-batch from day 1 (``max_num_batched_requests = 2``); 4 query tokens
mapping to 2 requests.
"""
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.attention import V4ComputeGlobalTopkIndices


def test_compute_global_topk_indices_v4_testmode():
    device = "cuda"
    torch.manual_seed(0)

    num_tokens = 4
    topk = 32        # small for fast test (real V4-Flash = 2048)
    block_size = 16  # compressed block size / compress_ratio = 64/4
    num_reqs = 2
    max_blocks_per_seq = 16

    # Random local topk indices in [-1, num_blocks*block_size).
    topk_indices = torch.randint(
        -1, max_blocks_per_seq * block_size,
        (num_tokens, topk), dtype=torch.int32, device=device
    )
    # Force some negative sentinels.
    topk_indices[0, 5] = -1
    topk_indices[2, :8] = -1  # trailing sentinels.
    token_to_req = torch.tensor([0, 1, 0, 1], dtype=torch.int32, device=device)
    block_table = torch.randint(
        0, 100, (num_reqs, max_blocks_per_seq),
        dtype=torch.int32, device=device,
    )
    is_valid_token = torch.tensor([1, 1, 0, 1], dtype=torch.uint8, device=device)

    m = V4ComputeGlobalTopkIndices(
        topk=topk, block_size=block_size, prefix="v4_topk_",
    )

    # Reference.
    ref_global, ref_lens = m.forward(
        topk_indices, token_to_req, block_table, is_valid_token,
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
    params["max_num_batched_requests"] = max(num_reqs, 2)
    pk = PersistentKernel(**params)

    ti_dt = pk.attach_input(topk_indices, name="topk_indices")
    ttr_dt = pk.attach_input(token_to_req, name="token_to_req")
    bt_dt = pk.attach_input(block_table, name="block_table")
    iv_dt = pk.attach_input(is_valid_token, name="is_valid_token")

    out_global = torch.zeros(num_tokens, topk, dtype=torch.int32, device=device)
    out_lens = torch.zeros(num_tokens, dtype=torch.int32, device=device)
    og_dt = pk.attach_input(out_global, name="global_topk")
    ol_dt = pk.attach_input(out_lens, name="topk_lens")

    with pk.compile_scope():
        _ = m.compile(
            ti_dt, ttr_dt, bt_dt, iv_dt,
            global_topk_indices=og_dt, topk_lens=ol_dt,
        )

    print("Compiling V4ComputeGlobalTopkIndices test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4ComputeGlobalTopkIndices test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"out_global[0, :8]: {out_global[0, :8]}")
    print(f"ref_global[0, :8]: {ref_global[0, :8]}")
    print(f"out_lens: {out_lens}")
    print(f"ref_lens: {ref_lens}")

    try:
        torch.testing.assert_close(out_global, ref_global)
        torch.testing.assert_close(out_lens, ref_lens)
        print("PASSED: V4ComputeGlobalTopkIndices compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: {e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_compute_global_topk_indices_v4_testmode()
