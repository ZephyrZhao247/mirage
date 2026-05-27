"""Catalog test: ``layers.attention.IndexerScoreTopK`` via PersistentKernel
test_mode.

Validates the V4-Flash Indexer score + Top-K kernel (v1: bf16 q/kv,
contiguous KV cache). Compares the compiled ``indexer_score_topk_sm100``
kernel against the PyTorch reference implemented inside
:meth:`IndexerScoreTopK.forward`.

Compares set-wise (sorted indices), since exact order may differ when
scores tie — the kernel uses an SMEM heap that emits the unsorted
top-K candidates while the PyTorch reference uses ``torch.topk`` (also
unsorted under ties).
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_indexer_score_topk_testmode():
    device = "cuda"
    torch.manual_seed(0)

    # Small shapes per the task spec.
    T = 4
    n_heads = 4
    head_dim = 32
    s_max = 16
    topk = 8
    compress_ratio = 4

    # Per-token Q (bf16). Random.
    q = torch.randn(
        T, n_heads, head_dim, dtype=torch.bfloat16, device=device
    ) * 0.5
    # Compressed KV cache (bf16, contiguous).
    kv_cache = torch.randn(
        s_max, head_dim, dtype=torch.bfloat16, device=device
    ) * 0.5
    # Per-head weights (fp32). Random — sign matters for the relu
    # interaction (negative weights flip the sign of the per-head
    # contribution; both reference and kernel must agree on ordering).
    weights_proj = torch.randn(
        T, n_heads, dtype=torch.float32, device=device
    ) * 0.5
    # Positions chosen so the causal mask leaves a healthy number of
    # valid candidates (>= topk) for at least some tokens.
    # positions // compress_ratio + 1 = valid_len:
    #   pos=12 -> valid_len=4 (< topk -> trailing -1 sentinels)
    #   pos=28 -> valid_len=8 (== topk)
    #   pos=44 -> valid_len=12
    #   pos=60 -> valid_len=16 (== s_max)
    positions = torch.tensor(
        [12, 28, 44, 60], dtype=torch.int32, device=device
    )

    # Output buffer (pre-allocated).
    topk_indices = torch.zeros(T, topk, dtype=torch.int32, device=device)

    m = layers.IndexerScoreTopK(
        index_n_heads=n_heads,
        index_head_dim=head_dim,
        topk=topk,
        compress_ratio=compress_ratio,
        prefix="t_",
    )

    # PyTorch reference (use the same tensors on device).
    ref_topk = m.forward(q, kv_cache, weights_proj, positions)

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

    q_dt = pk.attach_input(q, name="q")

    with pk.compile_scope():
        m.compile(
            q_dt,
            kv_cache,
            weights_proj,
            positions,
            topk_indices=topk_indices,
        )

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    # Set-wise comparison: per token, compare the sorted index lists.
    # Equivalently we could compare sorted score_summed[topk] values, but
    # the indices give a stronger guarantee — the kernel must select the
    # *exact same* candidate set (up to ties).
    def _normalise(t: torch.Tensor) -> torch.Tensor:
        # Sort each row's top-K indices ascending. -1 sentinels sort to
        # the front; their count must agree with the reference's, so the
        # mismatch is exposed if the kernel writes an extra/missing
        # valid index.
        sorted_t, _ = torch.sort(t.to(torch.int64), dim=-1)
        return sorted_t

    out_sorted = _normalise(topk_indices)
    ref_sorted = _normalise(ref_topk)

    try:
        torch.testing.assert_close(
            out_sorted, ref_sorted, atol=0, rtol=0
        )
        print(
            "PASSED: IndexerScoreTopK compile() matches forward() "
            "(set-wise)"
        )
    except AssertionError as e:
        print(
            "FAILED: IndexerScoreTopK compile() disagrees with "
            f"forward()\n{e}"
        )
        print(f"  out sorted [0]:  {out_sorted[0]}")
        print(f"  ref sorted [0]:  {ref_sorted[0]}")
        # Diagnostic: if the indices differ, also compare the *score
        # values* at each set of indices. They should be equal even when
        # tie-breaking picks different indices.
        for t in range(T):
            out_idx = out_sorted[t].tolist()
            ref_idx = ref_sorted[t].tolist()
            print(f"  t={t}: out={out_idx} ref={ref_idx}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_indexer_score_topk_testmode()
