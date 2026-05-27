"""Catalog test: ``layers.attention.CompressorStateUpdate`` via
PersistentKernel test_mode.

Validates the V4-Flash Compressor state-update kernel (the C2 sub-batch
of Wave-2).  Compares the compiled ``compressor_state_update_sm100``
task output against the PyTorch reference implemented inside
:meth:`CompressorStateUpdate.forward`.
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_compressor_state_update_testmode():
    device = "cuda"
    torch.manual_seed(0)

    # Small shapes per the task spec (head_dim=64, ratio=4, overlap=True, T=8).
    T = 8
    head_dim = 64
    compress_ratio = 4
    overlap = True
    # Flat ring: enough slots that positions=[0..7] don't collide.
    num_slots = 16

    # Random bf16 inputs.
    kv = torch.randn(T, head_dim, dtype=torch.bfloat16, device=device) * 0.5
    score = torch.randn(T, head_dim, dtype=torch.bfloat16, device=device) * 0.5
    positions = torch.arange(T, dtype=torch.int32, device=device)
    # slot_mapping: position mod num_slots (a plain ring index).
    slot_mapping = (positions.to(torch.long) % num_slots).to(torch.int32)
    # Pre-init state_cache to zeros.
    state_cache = torch.zeros(
        num_slots, 2 * head_dim, dtype=torch.bfloat16, device=device
    )

    # Build the module and seed its learned `ape` parameter with
    # something non-trivial so the score+ape branch actually moves the
    # numbers.
    m = layers.CompressorStateUpdate(
        head_dim=head_dim,
        compress_ratio=compress_ratio,
        overlap=overlap,
        prefix="t_",
    )
    with torch.no_grad():
        m.ape.copy_(
            torch.randn(
                compress_ratio, head_dim, dtype=torch.bfloat16, device=device
            ) * 0.25
        )
    m.to(device)

    # PyTorch reference on a *copy* of state_cache so the compile path
    # below sees a pristine zero buffer.
    ref_state = state_cache.clone()
    ref_state = m.forward(kv, score, positions, slot_mapping, ref_state)

    # ----------------------------------------------------------------
    # MPK compile + run path
    # ----------------------------------------------------------------
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = T
    params["max_num_batched_requests"] = T
    pk = PersistentKernel(**params)

    kv_dt = pk.attach_input(kv, name="kv")

    with pk.compile_scope():
        m.compile(
            kv_dt,
            score,
            positions,
            slot_mapping,
            state_cache,
        )

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    # Compare.  bf16 + a single add is well within rtol/atol = 1e-3.
    try:
        torch.testing.assert_close(
            state_cache, ref_state, atol=1e-3, rtol=1e-3
        )
        print("PASSED: CompressorStateUpdate compile() matches forward()")
    except AssertionError as e:
        print(
            "FAILED: CompressorStateUpdate compile() disagrees with "
            f"forward()\n{e}"
        )
        diff = (state_cache.float() - ref_state.float()).abs()
        print(f"  max abs diff: {diff.max().item()}")
        print(f"  actual[0, :8]: {state_cache[0, :8]}")
        print(f"  ref[0, :8]:    {ref_state[0, :8]}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_compressor_state_update_testmode()
