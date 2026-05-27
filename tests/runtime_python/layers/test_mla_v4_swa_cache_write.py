"""Catalog test: ``layers.attention.MLAv4SWACacheWrite`` via PersistentKernel
test_mode. Validates the per-token SWA cache write kernel against a plain
PyTorch reference copy.

Shapes (small):
  T = 4, head_dim = 64, window_size = 16, B = 1.
  positions = [0, 5, 10, 15] (no aliasing in this run).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.layers.attention.mla_v4_swa_cache_write import MLAv4SWACacheWrite
from mirage.mpk.persistent_kernel import PersistentKernel


def test_mla_v4_swa_cache_write_testmode():
    device = "cuda"
    torch.manual_seed(0)

    T = 4
    head_dim = 64
    window_size = 16

    kv_in = torch.randn(T, head_dim, dtype=torch.bfloat16, device=device)
    positions = torch.tensor([0, 5, 10, 15], dtype=torch.int32, device=device)
    swa_cache = torch.zeros(window_size, head_dim, dtype=torch.bfloat16, device=device)

    # PyTorch reference (operates on a clone so we can compare).
    ref_cache = swa_cache.clone()
    m = MLAv4SWACacheWrite(head_dim=head_dim, window_size=window_size, prefix="t_")
    m.forward(kv_in, positions, ref_cache)

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

    kv_in_dt = pk.attach_input(kv_in, name="kv_in")

    with pk.compile_scope():
        m.compile(kv_in_dt, positions, swa_cache)

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    try:
        # Exact copy — atol=0, rtol=0.
        torch.testing.assert_close(swa_cache, ref_cache, atol=0, rtol=0)
        print("PASSED: MLAv4SWACacheWrite compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: MLAv4SWACacheWrite compile() disagrees with forward()\n{e}")
        print(f"  swa_cache:\n{swa_cache}")
        print(f"  ref_cache:\n{ref_cache}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_mla_v4_swa_cache_write_testmode()
