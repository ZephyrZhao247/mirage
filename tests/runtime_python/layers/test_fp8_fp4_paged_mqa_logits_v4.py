"""V4-Flash ``fp8_fp4_paged_mqa_logits`` (NEW paged-MQA kernel, FP8 path) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fp8_fp4_paged_mqa_logits.md``.

Multi-batch from day 1 (Q = B * NEXT_N >= 2).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.indexer import V4Fp8Fp4PagedMqaLogits


def test_fp8_fp4_paged_mqa_logits_v4_testmode():
    device = "cuda"
    torch.manual_seed(0)

    # Small shape for fast cycle. V4-Flash uses N_HEADS=64, HEAD_DIM=128;
    # we shrink N_HEADS for test speed.
    n_heads = 4
    head_dim = 32
    block_size = 4
    kv_head_width = head_dim + 4
    max_model_len = 8
    # Multi-batch: Q >= 2.
    B = 2
    NEXT_N = 1
    Q = B * NEXT_N
    num_blocks = 4
    max_blocks = max_model_len // block_size  # 2

    # Build inputs.
    q = (
        torch.randn(Q, n_heads, head_dim, dtype=torch.float32, device=device) * 0.5
    ).to(torch.float8_e4m3fn)
    # kv_cache as uint8 buffer.
    # Build random fp8 K rows + fp32 scale per (block, slot).
    k_fp8 = (
        torch.randn(num_blocks, block_size, head_dim, dtype=torch.float32, device=device) * 0.5
    ).to(torch.float8_e4m3fn)
    k_scale = torch.rand(num_blocks, block_size, dtype=torch.float32, device=device) * 0.5 + 0.5
    kv_cache = torch.zeros(
        num_blocks, block_size, 1, kv_head_width, dtype=torch.uint8, device=device
    )
    kv_cache[..., : head_dim] = k_fp8.view(num_blocks, block_size, 1, head_dim).view(torch.uint8)
    # Write fp32 scale into bytes [head_dim:head_dim+4].
    kv_cache[..., head_dim : head_dim + 4] = (
        k_scale.view(num_blocks, block_size, 1, 1)
        .expand(num_blocks, block_size, 1, 1)
        .contiguous()
        .view(torch.uint8)  # this is just to allocate; we overwrite below
        if False
        else kv_cache[..., head_dim : head_dim + 4]
    )
    # Pack the fp32 scale bytes properly.
    scale_bytes = k_scale.view(num_blocks, block_size, 1).contiguous().view(torch.uint8).view(
        num_blocks, block_size, 1, 4
    )
    kv_cache[..., head_dim : head_dim + 4] = scale_bytes

    weights = torch.randn(Q, n_heads, dtype=torch.float32, device=device) * 0.5

    # Build block_tables: each batch gets a distinct mapping.
    block_tables = torch.zeros(Q, max_blocks, dtype=torch.int32, device=device)
    for qi in range(Q):
        for b in range(max_blocks):
            block_tables[qi, b] = (qi + b) % num_blocks

    # context_lens: vary per query atom.
    context_lens = torch.tensor(
        [max_model_len, max_model_len - 2][:Q],
        dtype=torch.int32, device=device,
    )

    logits_buf = torch.zeros(Q, max_model_len, dtype=torch.float32, device=device)

    module = V4Fp8Fp4PagedMqaLogits(
        n_heads=n_heads,
        head_dim=head_dim,
        block_size=block_size,
        kv_head_width=kv_head_width,
        max_model_len=max_model_len,
        prefix="v4_paged_",
    )
    ref = module.forward(q, kv_cache, weights, block_tables, context_lens)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = Q
    params["max_num_batched_requests"] = max(Q, 2)
    pk = PersistentKernel(**params)

    q_dt = pk.attach_input(q, name="q")
    kv_dt = pk.attach_input(kv_cache, name="kv_cache")
    w_dt = pk.attach_input(weights, name="weights")
    bt_dt = pk.attach_input(block_tables, name="block_tables")
    cl_dt = pk.attach_input(context_lens, name="context_lens")

    with pk.compile_scope():
        _ = module.compile(
            q_dt, kv_dt, w_dt, bt_dt, cl_dt,
            logits_out=logits_buf,
        )

    print("Compiling V4Fp8Fp4PagedMqaLogits test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4Fp8Fp4PagedMqaLogits test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"logits[0, :8] = {logits_buf[0, :8]}")
    print(f"ref   [0, :8] = {ref[0, :8]}")
    max_l = (logits_buf - ref).abs().max().item()
    print(f"max logit diff: {max_l}")

    try:
        # FP8 quant + fp32 accumulation -> generous tolerance.
        torch.testing.assert_close(logits_buf, ref, atol=1e-2, rtol=5e-2)
        print("PASSED: V4Fp8Fp4PagedMqaLogits compile() matches forward()")
    except AssertionError as e:
        print(
            "FAILED: V4Fp8Fp4PagedMqaLogits compile() disagrees with forward()\n"
            f"{e}"
        )
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_fp8_fp4_paged_mqa_logits_v4_testmode()
