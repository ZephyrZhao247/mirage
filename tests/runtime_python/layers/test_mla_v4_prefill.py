"""Catalog test: ``layers.attention.MLAv4Prefill`` via PersistentKernel
test_mode.

Validates the V4-Flash MLA prefill kernel (FlashAttn-style tile loop over
a gathered KV workspace, FP32 accumulators, causal mask, attn_sink) by
comparing the compiled ``mla_v4_prefill_sm100`` task output against the
PyTorch reference implemented inside :meth:`MLAv4Prefill.forward`.
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_mla_v4_prefill_testmode():
    device = "cuda"
    torch.manual_seed(0)

    # Small shapes per the task prompt.
    T_q = 8
    T_kv = 16          # ratio=0 SWA-only: the gather staged T_kv rows
    num_heads = 4
    head_dim = 64
    qk_rope_head_dim = 16   # metadata only in v1
    softmax_scale = 1.0 / (head_dim ** 0.5)

    # Inputs.
    q = torch.randn(T_q, num_heads, head_dim,
                    dtype=torch.bfloat16, device=device) * 0.25
    gathered_kv = torch.randn(T_kv, head_dim,
                              dtype=torch.bfloat16, device=device) * 0.25

    # Pre-allocated output so we can read it back after pk().
    o = torch.zeros(T_q, num_heads, head_dim,
                    dtype=torch.bfloat16, device=device)

    # Module + PyTorch reference.
    m = layers.MLAv4Prefill(
        num_heads=num_heads,
        head_dim=head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        softmax_scale=softmax_scale,
        prefix="t_",
    ).to(device)

    # Set a non-trivial attn_sink so the test exercises the sink-slot path.
    with torch.no_grad():
        m.attn_sink.copy_(torch.randn(num_heads, dtype=torch.float32,
                                      device=device) * 0.1)

    ref_o = m.forward(q, gathered_kv)

    # MPK compile + run.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = max(T_q, T_kv)
    params["max_num_batched_requests"] = T_q
    pk = PersistentKernel(**params)

    q_dt = pk.attach_input(q, name="q")

    with pk.compile_scope():
        m.compile(
            q_dt,
            gathered_kv,
            o=o,
        )

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    try:
        torch.testing.assert_close(o, ref_o, rtol=2e-2, atol=2e-2)
        print("PASSED: MLAv4Prefill compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: MLAv4Prefill compile() disagrees with forward()\n{e}")
        diff = (o.float() - ref_o.float()).abs()
        print(f"  diff max: {diff.max().item():.4f}")
        print(f"  diff mean: {diff.mean().item():.4f}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_mla_v4_prefill_testmode()
