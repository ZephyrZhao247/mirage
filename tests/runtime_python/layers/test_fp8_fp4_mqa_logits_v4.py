"""V4-Flash ``fp8_fp4_mqa_logits`` (NEW non-paged MQA kernel, FP8 path) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fp8_fp4_mqa_logits.md``.

Multi-batch from day 1 (T_chunk >= 2).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.indexer import V4Fp8Fp4MqaLogits


def test_fp8_fp4_mqa_logits_v4_testmode():
    device = "cuda"
    torch.manual_seed(0)

    n_heads = 4
    head_dim = 32
    T_chunk = 4
    N = 8

    q = (
        torch.randn(T_chunk, n_heads, head_dim, dtype=torch.float32, device=device) * 0.5
    ).to(torch.float8_e4m3fn)
    k_packed = (
        torch.randn(N, head_dim, dtype=torch.float32, device=device) * 0.5
    ).to(torch.float8_e4m3fn)
    k_scales = torch.rand(N, dtype=torch.float32, device=device) * 0.5 + 0.5
    weights = torch.randn(T_chunk, n_heads, dtype=torch.float32, device=device) * 0.5

    # Per-Q-row causal window.
    cu_seqlen_ks = torch.zeros(T_chunk, dtype=torch.int32, device=device)
    cu_seqlen_ke = torch.tensor(
        [N, N - 1, N - 2, N - 3][:T_chunk],
        dtype=torch.int32, device=device,
    )

    logits_buf = torch.zeros(T_chunk, N, dtype=torch.float32, device=device)

    module = V4Fp8Fp4MqaLogits(
        n_heads=n_heads,
        head_dim=head_dim,
        n_kv=N,
        prefix="v4_mqa_",
    )
    ref = module.forward(q, k_packed, k_scales, weights, cu_seqlen_ks, cu_seqlen_ke)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = T_chunk
    params["max_num_batched_requests"] = max(T_chunk, 2)
    pk = PersistentKernel(**params)

    q_dt = pk.attach_input(q, name="q")
    kp_dt = pk.attach_input(k_packed, name="k_packed")
    ks_dt = pk.attach_input(k_scales, name="k_scales")
    w_dt = pk.attach_input(weights, name="weights")
    cks_dt = pk.attach_input(cu_seqlen_ks, name="cu_seqlen_ks")
    cke_dt = pk.attach_input(cu_seqlen_ke, name="cu_seqlen_ke")

    with pk.compile_scope():
        _ = module.compile(
            q_dt, kp_dt, ks_dt, w_dt, cks_dt, cke_dt,
            logits_out=logits_buf,
        )

    print("Compiling V4Fp8Fp4MqaLogits test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4Fp8Fp4MqaLogits test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"logits[0, :8] = {logits_buf[0, :8]}")
    print(f"ref   [0, :8] = {ref[0, :8]}")
    max_l = (logits_buf - ref).abs().max().item()
    print(f"max logit diff: {max_l}")

    try:
        torch.testing.assert_close(logits_buf, ref, atol=1e-2, rtol=5e-2)
        print("PASSED: V4Fp8Fp4MqaLogits compile() matches forward()")
    except AssertionError as e:
        print(
            "FAILED: V4Fp8Fp4MqaLogits compile() disagrees with forward()\n"
            f"{e}"
        )
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_fp8_fp4_mqa_logits_v4_testmode()
