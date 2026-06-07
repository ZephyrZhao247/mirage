"""V4-Flash ``write_zeros_to_output`` test (multi-batch).

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/write_zeros_to_output.md``.
Catalog: ``mirage.mpk.layers.deepseek_v4.moe.write_zeros_to_output.V4WriteZerosToOutput``.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.layers.deepseek_v4.moe.write_zeros_to_output import V4WriteZerosToOutput
from mirage.mpk.persistent_kernel import PersistentKernel


def test_v4_write_zeros_to_output():
    device = "cuda"
    torch.manual_seed(0)

    num_tokens = 4
    top_k = 2
    output_dim = 32  # small N for naive kernel
    block_m = 4
    block_n = 8
    num_experts = 4

    module = V4WriteZerosToOutput(
        num_tokens=num_tokens,
        top_k=top_k,
        output_dim=output_dim,
        block_m=block_m,
        block_n=block_n,
        num_experts=num_experts,
    )
    em = module.em
    num_m_blocks = module.num_m_blocks

    c = torch.full(
        (num_tokens, top_k, output_dim),
        7.0,
        dtype=torch.bfloat16,
        device=device,
    )
    # Build a fake dispatch: half the m-blocks point to expert 0, the
    # other half to expert -1.
    sorted_token_ids = torch.full((em,), num_tokens * top_k, dtype=torch.int32, device=device)
    sorted_token_ids[0:4] = torch.arange(4, dtype=torch.int32, device=device)
    expert_ids = torch.full((num_m_blocks,), -1, dtype=torch.int32, device=device)
    expert_ids[0] = 0       # active expert -> kernel skips
    # Mark m-blocks 1 and 2 explicitly as -1 so they should be zeroed.
    num_tokens_post_pad = torch.tensor([3 * block_m], dtype=torch.int32, device=device)

    # PyTorch reference
    ref = module.forward(c, sorted_token_ids, expert_ids, num_tokens_post_pad)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = max(num_tokens, 8)
    params["max_num_batched_requests"] = max(num_tokens, 8)
    pk = PersistentKernel(**params)

    sorted_dt = pk.attach_input(sorted_token_ids, name="wz_sorted")
    eids_dt = pk.attach_input(expert_ids, name="wz_eids")
    npp_dt = pk.attach_input(num_tokens_post_pad, name="wz_npp")
    c_dt = pk.attach_input(c, name="wz_c")

    with pk.compile_scope():
        _ = module.compile(c_dt, sorted_dt, eids_dt, npp_dt)

    print("Compiling V4WriteZerosToOutput...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4WriteZerosToOutput...")
    pk()
    torch.cuda.synchronize()

    print(f"c[0,0,:8]:   {c[0, 0, :8]}")
    print(f"ref[0,0,:8]: {ref[0, 0, :8]}")
    max_diff = (c.float() - ref.float()).abs().max().item()
    print(f"Max absolute difference: {max_diff}")

    try:
        torch.testing.assert_close(c, ref, atol=0, rtol=0)
        print("PASSED: V4WriteZerosToOutput matches reference.")
    except AssertionError as e:
        print(f"FAILED: V4WriteZerosToOutput disagrees\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_v4_write_zeros_to_output()
