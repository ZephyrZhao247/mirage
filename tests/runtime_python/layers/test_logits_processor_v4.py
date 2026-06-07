"""V4-Flash ``logits_processor`` test (multi-batch).

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/logits_processor.md``.
Catalog: ``mirage.mpk.layers.deepseek_v4.std.V4LogitsProcessor`` (REUSE
alias of the existing :class:`mirage.mpk.layers.Linear` for the
``tp_size=1, soft_cap=None, scale=1.0`` production path).

For V4-Flash defaults the whole LogitsProcessor collapses to a single
bf16 GEMM ``F.linear(hidden_states, lm_head.weight)``; the alias
forwards directly to ``Linear``.

Run on a free GPU:
    CUDA_VISIBLE_DEVICES=0 python tests/runtime_python/layers/test_logits_processor_v4.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.layers.deepseek_v4.std import V4LogitsProcessor
from mirage.mpk.persistent_kernel import PersistentKernel


def test_v4_logits_processor_multibatch():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(42)

    # Multi-batch from day 1: batch_size >= 2. We pick a tiny vocab that
    # satisfies the Linear catalog's auto-grid (vocab % 64 == 0 or
    # vocab % 96 == 0). 4096 % 64 == 0.
    batch_size = 8
    hidden_size = 4096
    vocab_size = 4096

    hidden_states = torch.randn(
        batch_size, hidden_size, dtype=dtype, device=device
    )
    # Scale the weight to keep the bf16 GEMM in numerical range; matches
    # tests/runtime_python/layers/test_linear.py convention.
    weight = (
        torch.randn(vocab_size, hidden_size, dtype=dtype, device=device) * 0.01
    )
    out_buf = torch.zeros(batch_size, vocab_size, dtype=dtype, device=device)

    # PyTorch reference via fp32 accumulate (matches the kernel's fp32
    # accumulator) then cast to bf16.
    ref = (hidden_states.float() @ weight.float().T).to(dtype)

    module = V4LogitsProcessor(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        prefix="v4lp_",
    )
    module = module.to(device=device, dtype=dtype)
    module.weight.data.copy_(weight)

    # Sanity check: forward() agrees with the manual reference to within
    # a bf16 ULP. (Linear.forward uses F.linear, which accumulates in
    # fp32 and casts; matches the kernel.)
    ref_forward = module.forward(hidden_states)
    forward_max_diff = (ref_forward.float() - ref.float()).abs().max().item()
    assert forward_max_diff < 0.05, (
        f"V4LogitsProcessor.forward disagrees with manual reference: "
        f"max_diff={forward_max_diff}"
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

    x_dt = pk.attach_input(hidden_states, name="v4lp_hidden_states")
    with pk.compile_scope():
        _ = module.compile(x_dt, output=out_buf)

    print("Compiling V4 logits_processor test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4 logits_processor test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"out_buf[0, :8]: {out_buf[0, :8]}")
    print(f"ref[0, :8]:     {ref[0, :8]}")

    max_diff = (out_buf.float() - ref.float()).abs().max().item()
    print(f"Max absolute difference: {max_diff:.6f}")

    try:
        # bf16 GEMM tolerance (matches test_linear.py convention).
        torch.testing.assert_close(out_buf, ref, atol=0.5, rtol=0.5)
        print("PASSED: V4LogitsProcessor matches reference (multi-batch).")
    except AssertionError as e:
        print(f"FAILED: V4LogitsProcessor disagrees with reference\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_v4_logits_processor_multibatch()
