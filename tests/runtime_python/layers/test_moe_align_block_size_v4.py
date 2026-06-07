"""V4-Flash ``moe_align_block_size`` test (multi-batch).

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/moe_align_block_size.md``.
Catalog: ``mirage.mpk.layers.deepseek_v4.moe.moe_align_block_size.V4MoeAlignBlockSize``.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.layers.deepseek_v4.moe.moe_align_block_size import V4MoeAlignBlockSize
from mirage.mpk.persistent_kernel import PersistentKernel


def test_v4_moe_align_block_size():
    device = "cuda"
    torch.manual_seed(0)

    num_tokens = 4
    top_k = 2
    num_experts = 4
    block_size = 4

    module = V4MoeAlignBlockSize(
        num_tokens=num_tokens,
        top_k=top_k,
        num_experts=num_experts,
        block_size=block_size,
    )
    em = module.em
    num_m_blocks = module.num_m_blocks

    # topk_ids: each token has top_k routed expert ids.
    topk_ids = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 0]],
        dtype=torch.int32,
        device=device,
    )

    # Reference
    ref_sorted, ref_eids, ref_npp = module.forward(topk_ids)
    print("ref sorted:", ref_sorted.cpu().tolist())
    print("ref eids:  ", ref_eids.cpu().tolist())
    print("ref npp:   ", ref_npp.cpu().tolist())

    # Build output buffers.
    sorted_buf = torch.zeros(em, dtype=torch.int32, device=device)
    eids_buf = torch.zeros(num_m_blocks, dtype=torch.int32, device=device)
    npp_buf = torch.zeros(1, dtype=torch.int32, device=device)

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

    topk_dt = pk.attach_input(topk_ids, name="ma_topk")
    sorted_dt = pk.attach_input(sorted_buf, name="ma_sorted")
    eids_dt = pk.attach_input(eids_buf, name="ma_eids")
    npp_dt = pk.attach_input(npp_buf, name="ma_npp")

    with pk.compile_scope():
        _ = module.compile(topk_dt, sorted_dt, eids_dt, npp_dt)

    print("Compiling V4MoeAlignBlockSize...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4MoeAlignBlockSize...")
    pk()
    torch.cuda.synchronize()

    print("kernel sorted:", sorted_buf.cpu().tolist())
    print("kernel eids:  ", eids_buf.cpu().tolist())
    print("kernel npp:   ", npp_buf.cpu().tolist())

    try:
        torch.testing.assert_close(npp_buf, ref_npp, atol=0, rtol=0)
        torch.testing.assert_close(eids_buf, ref_eids, atol=0, rtol=0)
        # sorted_token_ids: ordering within an expert bucket may differ
        # between the bucket-sort and the reference (we use atomicAdd),
        # so compare per-expert MULTISET instead of element-by-element.
        bs = block_size
        num_post = int(npp_buf.item())
        for b in range(num_post // bs):
            e = int(eids_buf[b].item())
            kernel_block = set(sorted_buf[b * bs:(b + 1) * bs].cpu().tolist())
            ref_block = set(ref_sorted[b * bs:(b + 1) * bs].cpu().tolist())
            assert kernel_block == ref_block, (
                f"block {b} expert {e}: kernel={kernel_block} vs ref={ref_block}"
            )
        print("PASSED: V4MoeAlignBlockSize matches reference.")
    except AssertionError as e:
        print(f"FAILED: V4MoeAlignBlockSize disagrees\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_v4_moe_align_block_size()
