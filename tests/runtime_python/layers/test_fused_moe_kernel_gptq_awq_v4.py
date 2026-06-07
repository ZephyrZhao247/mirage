"""V4-Flash ``fused_moe_kernel_gptq_awq`` test (multi-batch).

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fused_moe_kernel_gptq_awq.md``.
Catalog: ``mirage.mpk.layers.deepseek_v4.moe.fused_moe_kernel_gptq_awq.V4FusedMoeKernelGptqAwq``.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.layers.deepseek_v4.moe.fused_moe_kernel_gptq_awq import (
    V4FusedMoeKernelGptqAwq,
)
from mirage.mpk.persistent_kernel import PersistentKernel


def test_v4_fused_moe_kernel_gptq_awq():
    device = "cuda"
    torch.manual_seed(0)

    num_tokens = 4
    top_k = 2
    k_dim = 32
    n_dim = 16
    num_experts = 4
    block_m = 4
    block_n = 8
    group_size = 8

    module = V4FusedMoeKernelGptqAwq(
        num_tokens=num_tokens,
        top_k=top_k,
        k_dim=k_dim,
        n_dim=n_dim,
        num_experts=num_experts,
        block_m=block_m,
        block_n=block_n,
        group_size=group_size,
        has_zp=False,
        mul_routed_weight=False,
    )
    em = module.em
    num_m_blocks = module.num_m_blocks
    num_k_groups = module.num_k_groups

    a = torch.randn(num_tokens, k_dim, dtype=torch.bfloat16, device=device) * 0.1
    # int8 W in [-127, 127], will be dequantised as (q - 128) * scale.
    b = torch.randint(-32, 32, (num_experts, n_dim, k_dim),
                      dtype=torch.int8, device=device)
    b_scale = (torch.rand(num_experts, n_dim, num_k_groups,
                          dtype=torch.float32, device=device) + 0.5) * 0.01
    b_zp = torch.zeros(num_experts, n_dim, num_k_groups,
                       dtype=torch.int8, device=device)  # dummy
    topk_weights = torch.ones(num_tokens * top_k, dtype=torch.float32, device=device)

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
        a, b, b_scale, b_zp, sorted_token_ids, expert_ids, num_tokens_post_pad,
        topk_weights,
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

    a_dt = pk.attach_input(a, name="fmq_a")
    b_dt = pk.attach_input(b, name="fmq_b")
    bs_dt = pk.attach_input(b_scale, name="fmq_bscale")
    bz_dt = pk.attach_input(b_zp, name="fmq_bzp")
    sorted_dt = pk.attach_input(sorted_token_ids, name="fmq_sorted")
    eids_dt = pk.attach_input(expert_ids, name="fmq_eids")
    npp_dt = pk.attach_input(num_tokens_post_pad, name="fmq_npp")
    tw_dt = pk.attach_input(topk_weights, name="fmq_tw")
    c_dt = pk.attach_input(c_buf, name="fmq_c")

    with pk.compile_scope():
        _ = module.compile(
            a_dt, b_dt, bs_dt, bz_dt, sorted_dt, eids_dt, npp_dt, tw_dt, c_dt,
        )

    print("Compiling V4FusedMoeKernelGptqAwq...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4FusedMoeKernelGptqAwq...")
    pk()
    torch.cuda.synchronize()

    print(f"c_buf[0,0,:8]: {c_buf[0, 0, :8]}")
    print(f"ref[0,0,:8]:   {ref[0, 0, :8]}")
    max_diff = (c_buf.float() - ref.float()).abs().max().item()
    print(f"Max absolute difference: {max_diff}")

    try:
        torch.testing.assert_close(c_buf, ref, atol=0.1, rtol=0.05)
        print("PASSED: V4FusedMoeKernelGptqAwq matches reference.")
    except AssertionError as e:
        print(f"FAILED: V4FusedMoeKernelGptqAwq disagrees\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_v4_fused_moe_kernel_gptq_awq()
