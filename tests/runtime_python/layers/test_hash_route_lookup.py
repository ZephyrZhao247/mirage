"""Catalog test: ``layers.moe.HashRouteLookup`` via PersistentKernel test_mode.

Validates the hash-based MoE expert routing used by DeepSeek V4-Flash for
``layer_idx < num_hash_layers`` (3 in V4-Flash). Compares the compiled
``hash_route_lookup_sm100`` kernel against the PyTorch reference (a
plain gather + uniform weights).
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_hash_route_lookup_testmode():
    device = "cuda"
    torch.manual_seed(0)

    # Toy shapes per the task spec.
    vocab_size = 16
    k = 2
    n = 8

    # Hand-crafted ``tid2eid`` so the gather is easy to eyeball when
    # debugging. Experts are int32 in ``[0, num_experts)``; for the
    # lookup-table test we only care that the kernel returns whatever
    # value sits in ``tid2eid[input_id, k]`` — no global expert count
    # is enforced inside the kernel.
    tid2eid = torch.randint(
        0, 64, (vocab_size, k), dtype=torch.int32, device=device
    )
    input_ids = torch.randint(
        0, vocab_size, (n,), dtype=torch.int32, device=device
    )

    # Output buffers (so the test can compare against the reference).
    expert_ids = torch.zeros(n, k, dtype=torch.int32, device=device)
    topk_weights = torch.zeros(n, k, dtype=torch.float32, device=device)

    # Build the catalog module and load the tid2eid table into it.
    m = layers.HashRouteLookup(
        vocab_size=vocab_size,
        num_experts_per_tok=k,
        prefix="t_",
    )
    # Push the table into the nn.Parameter (mimics load_state_dict). We
    # need .data.copy_ because nn.Parameter only stores floating-point
    # tensors by default; the constructor created an int32 zero tensor.
    m.tid2eid.data = tid2eid.cpu().clone()

    # PyTorch reference (executed on CPU here because m.tid2eid lives on
    # CPU; the device-side reference would differ only by the gather's
    # device). Move both to the test device for the comparison.
    ref_expert_ids, ref_topk_weights = m.forward(input_ids.cpu())
    ref_expert_ids = ref_expert_ids.to(device)
    ref_topk_weights = ref_topk_weights.to(device)

    # Put the parameter on the test device so attach_input sees a CUDA
    # tensor (attach_input does not move tensors itself).
    m.tid2eid.data = m.tid2eid.data.to(device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = n
    params["max_num_batched_requests"] = n
    pk = PersistentKernel(**params)

    input_ids_dt = pk.attach_input(input_ids, name="input_ids")

    with pk.compile_scope():
        m.compile(
            input_ids_dt,
            expert_ids=expert_ids,
            topk_weights=topk_weights,
        )

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    try:
        torch.testing.assert_close(expert_ids, ref_expert_ids)
        torch.testing.assert_close(
            topk_weights, ref_topk_weights, atol=1e-7, rtol=0
        )
        print("PASSED: HashRouteLookup compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: HashRouteLookup compile() disagrees with forward()\n{e}")
        print(f"  expert_ids:\n{expert_ids}")
        print(f"  ref_expert_ids:\n{ref_expert_ids}")
        print(f"  topk_weights:\n{topk_weights}")
        print(f"  ref_topk_weights:\n{ref_topk_weights}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_hash_route_lookup_testmode()
