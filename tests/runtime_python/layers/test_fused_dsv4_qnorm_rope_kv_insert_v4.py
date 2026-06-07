"""V4-Flash monolithic 5-op fusion test (multi-batch from day 1).

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/
fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert.md``.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.attention import V4FusedDSV4QNormRopeKVInsert


def test_fused_dsv4_qnorm_rope_kv_insert_v4_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    num_tokens = 2
    num_heads_q = 64
    q_head_padded = 64
    head_dim = 512
    rope_dim = 64
    cache_block_size = 64
    num_blocks = 4

    m = V4FusedDSV4QNormRopeKVInsert(
        num_heads_q=num_heads_q,
        q_head_padded=q_head_padded,
        cache_block_size=cache_block_size,
        num_blocks=num_blocks,
        prefix="v4_dsv4_qnorm_rope_kv_",
    )

    q_in = torch.randn(num_tokens, num_heads_q, head_dim,
                       dtype=dtype, device=device)
    kv_in = torch.randn(num_tokens, head_dim, dtype=dtype, device=device)
    slot_mapping = torch.tensor([0, 1], dtype=torch.int64, device=device)
    positions = torch.tensor([5, 10], dtype=torch.int64, device=device)
    cos_sin_cache = torch.randn(1024, rope_dim, dtype=torch.float32,
                                device=device)
    q_out_buf = torch.zeros(num_tokens, q_head_padded, head_dim,
                            dtype=dtype, device=device)
    k_cache_buf = torch.zeros(num_blocks, m.block_stride_bytes,
                              dtype=torch.uint8, device=device)

    ref_q, _ = m.forward(
        q_in.clone(), kv_in.clone(), slot_mapping.clone(),
        positions.clone(), cos_sin_cache.clone(),
    )

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = num_tokens
    params["max_num_batched_requests"] = num_tokens
    pk = PersistentKernel(**params)

    q_in_dt = pk.attach_input(q_in, name="q_in")
    kv_in_dt = pk.attach_input(kv_in, name="kv_in")
    slot_mapping_dt = pk.attach_input(slot_mapping, name="slot_mapping")
    positions_dt = pk.attach_input(positions, name="positions")
    cos_sin_dt = pk.attach_input(cos_sin_cache, name="cos_sin_cache")
    k_cache_dt = pk.attach_input(k_cache_buf, name="k_cache")

    with pk.compile_scope():
        _ = m.compile(
            q_in_dt, kv_in_dt, slot_mapping_dt, positions_dt,
            cos_sin_dt, k_cache_dt, q_out=q_out_buf,
        )

    pk.compile(output_dir=os.path.dirname(__file__))
    pk()
    torch.cuda.synchronize()

    max_q = (q_out_buf[:, :num_heads_q].float() -
             ref_q[:, :num_heads_q].float()).abs().max().item()
    print(f"q_out (live) max-abs diff: {max_q}")
    try:
        torch.testing.assert_close(
            q_out_buf[:, :num_heads_q].float(),
            ref_q[:, :num_heads_q].float(),
            atol=0.1, rtol=0.1,
        )
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED:\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_fused_dsv4_qnorm_rope_kv_insert_v4_testmode()
