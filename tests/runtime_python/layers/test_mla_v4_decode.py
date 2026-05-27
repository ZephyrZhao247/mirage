"""Catalog test: ``layers.attention.MLAv4Decode`` via PersistentKernel
test_mode.

Validates the V4-Flash MLA decode (v1: SWA-only, compress_ratio=0).
Compares the compiled ``mla_v4_decode_sm100`` kernel against the
PyTorch reference implemented inside :meth:`MLAv4Decode.forward`.
"""

import math
import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_mla_v4_decode_testmode():
    device = "cuda"
    torch.manual_seed(0)

    # Small shapes per the task spec.
    T = 4
    num_heads = 4
    head_dim = 64
    rope_dim = 16
    softmax_scale = 1.0 / math.sqrt(head_dim)

    # SWA cache: contiguous [swa_total, head_dim] bf16. With T=4 and
    # positions = [1, 2, 3, 4] each token attends to a strictly-prefix
    # window of the cache. swa_total covers the maximum position.
    swa_total = 16
    swa_cache = torch.randn(
        swa_total, head_dim, dtype=torch.bfloat16, device=device
    ) * 0.5

    # Per-token Q and absolute positions. Avoid position 0 (no KV rows)
    # so the comparison has signal on every token.
    q = torch.randn(
        T, num_heads, head_dim, dtype=torch.bfloat16, device=device
    ) * 0.5
    positions = torch.tensor(
        [1, 2, 3, 4], dtype=torch.int32, device=device
    )

    # Output buffer (pre-allocated so the test can read it after pk()).
    o = torch.zeros(
        T, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )

    # Build the catalog module with random attn_sink so the comparison
    # is non-trivial.
    m = layers.MLAv4Decode(
        num_heads=num_heads,
        head_dim=head_dim,
        qk_rope_head_dim=rope_dim,
        softmax_scale=softmax_scale,
        prefix="t_",
    )
    m.attn_sink.data = torch.randn(
        num_heads, dtype=torch.float32
    )

    # PyTorch reference (CPU for the attn_sink, then move).
    ref_o = m.forward(
        q.cpu(), swa_cache.cpu(), positions.cpu()
    ).to(device)

    # Move weights onto the test device so attach_input sees CUDA tensors.
    m.attn_sink.data = m.attn_sink.data.to(device)

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
            swa_cache,
            positions,
            o=o,
        )

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    try:
        torch.testing.assert_close(o, ref_o, atol=2e-2, rtol=2e-2)
        print("PASSED: MLAv4Decode compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: MLAv4Decode compile() disagrees with forward()\n{e}")
        print(f"  o[0, 0, :8]:     {o[0, 0, :8]}")
        print(f"  ref_o[0, 0, :8]: {ref_o[0, 0, :8]}")
        print(f"  diff max: {(o - ref_o).abs().max().item()}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_mla_v4_decode_testmode()
