"""V4-Flash ``save_partial_states`` (NEW kernel) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/save_partial_states.md``.

Per-token write of (kv | score + ape[position % compress_ratio]) into
the state_cache at slot_mapping[t]. Multi-batch from day 1.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.compressor import V4SavePartialStates


def test_save_partial_states_v4_testmode():
    device = "cuda"
    torch.manual_seed(0)

    # Use the indexer-compressor shapes for a compact test: head_dim=128,
    # coff=2 -> head_size=256; compress_ratio=4; block_size=16.
    head_size = 256
    compress_ratio = 4
    block_size = 16
    num_blocks = 4
    num_tokens = 4

    # Per-token inputs.
    kv = torch.randn(num_tokens, head_size, dtype=torch.float32, device=device)
    score = torch.randn(num_tokens, head_size, dtype=torch.float32, device=device)
    ape = torch.randn(compress_ratio, head_size, dtype=torch.float32, device=device)
    positions = torch.tensor([0, 1, 5, 42], dtype=torch.int64, device=device)
    # slot_mapping spans 2 blocks to exercise the block/off split. Token 3
    # uses slot -1 to exercise the early-exit (pad) branch.
    slot_mapping = torch.tensor([0, 1, 18, -1], dtype=torch.int64, device=device)

    state_cache_buf = torch.zeros(
        num_blocks, block_size, 2 * head_size, dtype=torch.float32, device=device
    )
    state_cache_ref = state_cache_buf.clone()

    module = V4SavePartialStates(
        head_size=head_size,
        compress_ratio=compress_ratio,
        block_size=block_size,
        prefix="v4_sps_",
    )
    module.forward(kv, score, ape, positions, slot_mapping, state_cache_ref)

    # Build PK in test mode.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = num_tokens
    params["max_num_batched_requests"] = max(num_tokens, 4)
    pk = PersistentKernel(**params)

    kv_dt = pk.attach_input(kv, name="kv")
    score_dt = pk.attach_input(score, name="score")
    ape_dt = pk.attach_input(ape, name="ape")
    pos_dt = pk.attach_input(positions, name="positions")
    slot_dt = pk.attach_input(slot_mapping, name="slot_mapping")
    cache_dt = pk.attach_input(state_cache_buf, name="state_cache")

    with pk.compile_scope():
        _ = module.compile(
            kv_dt, score_dt, ape_dt, pos_dt, slot_dt, cache_dt
        )

    print("Compiling V4SavePartialStates test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4SavePartialStates test kernel...")
    pk()
    torch.cuda.synchronize()

    # Compare written slots only (token 3 is padded; its slot has no
    # mutation in either path).
    max_diff = (state_cache_buf - state_cache_ref).abs().max().item()
    print(f"max-abs diff (state_cache): {max_diff}")

    try:
        torch.testing.assert_close(state_cache_buf, state_cache_ref,
                                   atol=1e-5, rtol=1e-5)
        print("PASSED: V4SavePartialStates compile() matches forward()")
    except AssertionError as e:
        print("FAILED: V4SavePartialStates compile() disagrees with forward()")
        print(e)
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_save_partial_states_v4_testmode()
