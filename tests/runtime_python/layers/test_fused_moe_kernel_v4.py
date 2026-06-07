"""V4-Flash ``fused_moe_kernel`` test (multi-batch).

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fused_moe_kernel.md``.
Catalog: ``mirage.mpk.layers.deepseek_v4.moe.fused_moe_kernel.V4FusedMoeKernel``.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.layers.deepseek_v4.moe.fused_moe_kernel import V4FusedMoeKernel
from mirage.mpk.persistent_kernel import PersistentKernel


def test_v4_fused_moe_kernel():
    device = "cuda"
    torch.manual_seed(0)

    num_tokens = 4
    top_k = 2
    k_dim = 32
    n_dim = 16
    num_experts = 4
    block_m = 4
    block_n = 8

    module = V4FusedMoeKernel(
        num_tokens=num_tokens,
        top_k=top_k,
        k_dim=k_dim,
        n_dim=n_dim,
        num_experts=num_experts,
        block_m=block_m,
        block_n=block_n,
        mul_routed_weight=False,
    )
    em = module.em
    num_m_blocks = module.num_m_blocks

    a = torch.randn(num_tokens, k_dim, dtype=torch.bfloat16, device=device) * 0.1
    b = torch.randn(num_experts, n_dim, k_dim, dtype=torch.bfloat16, device=device) * 0.1
    topk_weights = torch.ones(num_tokens * top_k, dtype=torch.float32, device=device)

    # Simple dispatch: token t routes its top_k slots to experts (t, t+1) mod E.
    sorted_token_ids = torch.full((em,), num_tokens * top_k, dtype=torch.int32, device=device)
    expert_ids = torch.full((num_m_blocks,), -1, dtype=torch.int32, device=device)
    cursor = 0
    expert_to_tokens = {e: [] for e in range(num_experts)}
    for t in range(num_tokens):
        for k in range(top_k):
            e = (t + k) % num_experts
            expert_to_tokens[e].append(t * top_k + k)
    for e in range(num_experts):
        if not expert_to_tokens[e]:
            continue
        block_id = cursor // block_m
        for i, ot in enumerate(expert_to_tokens[e]):
            sorted_token_ids[cursor + i] = ot
        expert_ids[block_id] = e
        cursor += block_m
    num_tokens_post_pad = torch.tensor([cursor], dtype=torch.int32, device=device)

    ref = module.forward(
        a, b, sorted_token_ids, expert_ids, num_tokens_post_pad, topk_weights
    )

    c_buf = torch.zeros(num_tokens, top_k, n_dim, dtype=torch.bfloat16, device=device)

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

    a_dt = pk.attach_input(a, name="fm_a")
    b_dt = pk.attach_input(b, name="fm_b")
    sorted_dt = pk.attach_input(sorted_token_ids, name="fm_sorted")
    eids_dt = pk.attach_input(expert_ids, name="fm_eids")
    npp_dt = pk.attach_input(num_tokens_post_pad, name="fm_npp")
    tw_dt = pk.attach_input(topk_weights, name="fm_tw")
    c_dt = pk.attach_input(c_buf, name="fm_c")

    with pk.compile_scope():
        _ = module.compile(a_dt, b_dt, sorted_dt, eids_dt, npp_dt, tw_dt, c_dt)

    print("Compiling V4FusedMoeKernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4FusedMoeKernel...")
    pk()
    torch.cuda.synchronize()

    print(f"c_buf[0,0,:8]: {c_buf[0, 0, :8]}")
    print(f"ref[0,0,:8]:   {ref[0, 0, :8]}")
    max_diff = (c_buf.float() - ref.float()).abs().max().item()
    print(f"Max absolute difference: {max_diff}")

    try:
        torch.testing.assert_close(c_buf, ref, atol=0.05, rtol=0.05)
        print("PASSED: V4FusedMoeKernel matches reference.")
    except AssertionError as e:
        print(f"FAILED: V4FusedMoeKernel disagrees\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_v4_fused_moe_kernel()
