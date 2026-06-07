"""V4-Flash ``vocab_parallel_embedding`` test (multi-batch).

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/vocab_parallel_embedding.md``.
Catalog: ``mirage.mpk.layers.deepseek_v4.std.V4VocabParallelEmbedding``
(REUSE alias of the existing :class:`mirage.mpk.layers.Embed` for the
``tp_size=1`` production path).

Run on a free GPU:
    CUDA_VISIBLE_DEVICES=0 python tests/runtime_python/layers/test_vocab_parallel_embedding_v4.py
"""

import os
import sys

import torch
import torch.nn.functional as F

import mirage
from mirage.mpk.layers.deepseek_v4.std import V4VocabParallelEmbedding
from mirage.mpk.persistent_kernel import PersistentKernel


def test_v4_vocab_parallel_embedding_multibatch():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # Tiny vocab + small hidden dim — keeps compile time fast. Multi-batch
    # from day 1: batch_size >= 2.
    num_embeddings = 128
    embedding_dim = 256
    batch_size = 4

    weight = torch.randn(num_embeddings, embedding_dim, dtype=dtype, device=device)
    input_tokens = torch.randint(
        low=0,
        high=num_embeddings,
        size=(batch_size,),
        dtype=torch.int64,
        device=device,
    )
    out_buf = torch.zeros(batch_size, embedding_dim, dtype=dtype, device=device)

    module = V4VocabParallelEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        prefix="v4emb_",
    )
    module = module.to(device=device, dtype=dtype)
    with torch.no_grad():
        module.weight.data.copy_(weight)

    ref = module.forward(input_tokens)
    ref_plain = F.embedding(input_tokens, weight)
    assert torch.equal(ref, ref_plain), (
        "V4VocabParallelEmbedding.forward disagrees with F.embedding."
    )

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    pk = PersistentKernel(**params)

    input_dt = pk.attach_input(input_tokens, name="v4emb_input_tokens")
    with pk.compile_scope():
        # input_source=1 ⇒ kernel reads token IDs from this DTensor (not
        # the runtime's rolling token buffer), matching the standalone-test
        # contract.
        _ = module.compile(input_dt, input_source=1, output=out_buf)

    print("Compiling V4 vocab_parallel_embedding test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4 vocab_parallel_embedding test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"out_buf[:2, :8]:\n{out_buf[:2, :8]}")
    print(f"ref[:2, :8]:\n{ref[:2, :8]}")

    try:
        # Embedding is a byte-copy of bf16 rows — exact match expected.
        torch.testing.assert_close(out_buf, ref, atol=0.0, rtol=0.0)
        print(
            "PASSED: V4VocabParallelEmbedding produces exact lookup "
            "(multi-batch)."
        )
    except AssertionError as e:
        max_diff = (out_buf.float() - ref.float()).abs().max().item()
        print(f"FAILED: V4VocabParallelEmbedding disagrees, max diff = {max_diff}")
        print(str(e))
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_v4_vocab_parallel_embedding_multibatch()
