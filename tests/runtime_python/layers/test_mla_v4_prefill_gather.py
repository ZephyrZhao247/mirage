"""Catalog test: ``layers.attention.MLAv4PrefillGather`` via PersistentKernel.

Validates the V4-Flash MLA prefill paged-to-contiguous KV gather kernel.
Compares the compiled ``mla_v4_prefill_gather_sm100`` task output against
the PyTorch reference (``index_select`` over a hand-crafted page table).
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_mla_v4_prefill_gather_testmode():
    device = "cuda"
    torch.manual_seed(0)

    # Small shapes per the task spec.
    head_dim = 64
    page_size = 8
    num_pages = 4
    num_kv_tokens = 16  # = 2 active pages worth of rows

    # Hand-crafted page table: block_id -> physical page id.
    # 2 active pages = ceil(num_kv_tokens / page_size).
    num_active_pages = (num_kv_tokens + page_size - 1) // page_size  # = 2
    # Map block 0 -> physical page 2, block 1 -> physical page 0
    # (deliberately permuted to confirm the gather honors the page table).
    page_table_list = [2, 0]
    assert len(page_table_list) == num_active_pages

    page_table = torch.tensor(page_table_list, dtype=torch.int32, device=device)

    # Random SWA cache.
    swa_cache = torch.randn(
        num_pages, page_size, head_dim, dtype=torch.bfloat16, device=device
    )

    # Reference output via PyTorch.
    m = layers.MLAv4PrefillGather(
        head_dim=head_dim, page_size=page_size, prefix="t_"
    )
    ref = m.forward(swa_cache, page_table, num_kv_tokens)
    assert ref.shape == (num_kv_tokens, head_dim)

    # Pre-allocate the output buffer for the kernel.
    gathered_kv = torch.zeros(
        num_kv_tokens, head_dim, dtype=torch.bfloat16, device=device
    )

    # MPK build-up.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = num_kv_tokens
    params["max_num_batched_requests"] = 1
    params["max_seq_length"] = num_kv_tokens
    params["max_num_pages"] = num_pages
    params["page_size"] = page_size

    # Wire the page-table meta-tensors. v1 only consumes
    # paged_kv_indices_buffer; paged_kv_indptr / last_page_len are accepted
    # by the task signature for forward compatibility but unused.
    n_req = 1
    paged_kv_indptr_buffer = torch.tensor(
        [0, num_active_pages], dtype=torch.int32, device=device
    )
    paged_kv_indices_buffer = page_table.clone()
    paged_kv_last_page_len_buffer = torch.tensor(
        [num_kv_tokens - (num_active_pages - 1) * page_size],
        dtype=torch.int32,
        device=device,
    )

    params["meta_tensors"] = {
        "paged_kv_indptr_buffer": paged_kv_indptr_buffer,
        "paged_kv_indices_buffer": paged_kv_indices_buffer,
        "paged_kv_last_page_len_buffer": paged_kv_last_page_len_buffer,
    }
    _ = n_req  # silence linter; documented above

    pk = PersistentKernel(**params)
    swa_dt = pk.attach_input(swa_cache, name="swa_cache")
    gkv_dt = pk.attach_input(gathered_kv, name="gathered_kv")

    with pk.compile_scope():
        m.compile(swa_dt, gathered_kv=gkv_dt)

    print("Compiling MLAv4PrefillGather test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    try:
        # The gather is a pure copy — fp math is bit-exact.
        torch.testing.assert_close(gathered_kv, ref, atol=0.0, rtol=0.0)
        print("PASSED: MLAv4PrefillGather compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: MLAv4PrefillGather compile() disagrees with forward()\n{e}")
        diff = (gathered_kv.float() - ref.float()).abs()
        print(f"  diff max: {diff.max().item()}")
        print(f"  diff sum: {diff.sum().item()}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_mla_v4_prefill_gather_testmode()
