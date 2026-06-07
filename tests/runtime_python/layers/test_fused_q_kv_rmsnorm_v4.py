"""V4-Flash ``fused_q_kv_rmsnorm`` (NEW kernel) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fused_q_kv_rmsnorm.md``.

Multi-batch from day 1 (``max_num_batched_requests = 2``).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.attention import V4FusedQKVRMSNorm


def test_fused_q_kv_rmsnorm_v4_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    num_tokens = 2
    q_size = 1024
    kv_size = 512
    eps = 1e-6

    m = V4FusedQKVRMSNorm(
        q_size=q_size, kv_size=kv_size, eps=eps, prefix="v4_qkv_rms_"
    )
    m.q_weight.data = m.q_weight.data.to(device=device, dtype=dtype)
    m.q_weight.data.copy_(torch.randn(q_size, dtype=dtype, device=device))
    m.kv_weight.data = m.kv_weight.data.to(device=device, dtype=dtype)
    m.kv_weight.data.copy_(torch.randn(kv_size, dtype=dtype, device=device))

    qr = torch.randn(num_tokens, q_size, dtype=dtype, device=device)
    kv = torch.randn(num_tokens, kv_size, dtype=dtype, device=device)
    qr_out_buf = torch.zeros(num_tokens, q_size, dtype=dtype, device=device)
    kv_out_buf = torch.zeros(num_tokens, kv_size, dtype=dtype, device=device)

    ref_qr, ref_kv = m.forward(qr, kv)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = num_tokens
    params["max_num_batched_requests"] = num_tokens
    pk = PersistentKernel(**params)

    qr_dt = pk.attach_input(qr, name="qr")
    kv_dt = pk.attach_input(kv, name="kv")

    with pk.compile_scope():
        _ = m.compile(qr_dt, kv_dt, qr_out=qr_out_buf, kv_out=kv_out_buf)

    print("Compiling V4FusedQKVRMSNorm test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))
    print("Running V4FusedQKVRMSNorm test kernel...")
    pk()
    torch.cuda.synchronize()

    max_q = (qr_out_buf.float() - ref_qr.float()).abs().max().item()
    max_k = (kv_out_buf.float() - ref_kv.float()).abs().max().item()
    print(f"qr max-abs diff: {max_q}; kv max-abs diff: {max_k}")

    try:
        torch.testing.assert_close(qr_out_buf, ref_qr, atol=0.05, rtol=0.05)
        torch.testing.assert_close(kv_out_buf, ref_kv, atol=0.05, rtol=0.05)
        print("PASSED")
    except AssertionError as e:
        print(f"FAILED:\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_fused_q_kv_rmsnorm_v4_testmode()
