"""Catalog test: ``layers.attention.MLAv4QKVRMSNorm`` via PersistentKernel test_mode.

Validates the V4-Flash pre-attention joint Q-lora + KV-lora RMSNorm.
Compares the compiled ``mla_v4_q_kv_rmsnorm_sm100`` kernel against the
PyTorch reference (two independent RMSNorm calls — exactly what vLLM's
``fused_q_kv_rmsnorm`` Triton kernel fuses for throughput).
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_mla_v4_q_kv_rmsnorm_testmode():
    device = "cuda"
    torch.manual_seed(0)

    # Small shapes per the task spec. Both ranks chosen as multiples of
    # 128 (the kernel's NUM_THREADS default). q_lora_rank=256 and
    # kv_lora_rank=128 still exercise the two-stream code path while
    # keeping compile + run time low.
    T = 4
    q_lora_rank = 256
    kv_lora_rank = 128

    q_lora = torch.randn(T, q_lora_rank, dtype=torch.bfloat16, device=device)
    kv_lora = torch.randn(T, kv_lora_rank, dtype=torch.bfloat16, device=device)

    # Output buffers (so the test can compare against the reference).
    q_norm = torch.zeros(T, q_lora_rank, dtype=torch.bfloat16, device=device)
    kv_norm = torch.zeros(T, kv_lora_rank, dtype=torch.bfloat16, device=device)

    # Build the catalog module with random per-channel weights to make
    # the comparison non-trivial (the default ones() would let a buggy
    # kernel pass by accident).
    m = layers.MLAv4QKVRMSNorm(
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        eps=1e-6,
        prefix="t_",
    )
    m.q_norm_weight.data = torch.randn(
        q_lora_rank, dtype=torch.bfloat16
    )
    m.kv_norm_weight.data = torch.randn(
        kv_lora_rank, dtype=torch.bfloat16
    )

    # PyTorch reference on CPU; move to device for the comparison. We
    # use CPU weights here because m.*_weight live on CPU until we
    # attach them; the reference math is device-independent.
    ref_q_norm, ref_kv_norm = m.forward(q_lora.cpu(), kv_lora.cpu())
    ref_q_norm = ref_q_norm.to(device)
    ref_kv_norm = ref_kv_norm.to(device)

    # Move weights onto the test device so attach_input sees CUDA tensors.
    m.q_norm_weight.data = m.q_norm_weight.data.to(device)
    m.kv_norm_weight.data = m.kv_norm_weight.data.to(device)

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

    q_lora_dt = pk.attach_input(q_lora, name="q_lora")
    kv_lora_dt = pk.attach_input(kv_lora, name="kv_lora")

    with pk.compile_scope():
        m.compile(
            q_lora_dt,
            kv_lora_dt,
            q_norm=q_norm,
            kv_norm=kv_norm,
        )

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    try:
        torch.testing.assert_close(
            q_norm, ref_q_norm, atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            kv_norm, ref_kv_norm, atol=1e-2, rtol=1e-2
        )
        print("PASSED: MLAv4QKVRMSNorm compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: MLAv4QKVRMSNorm compile() disagrees with forward()\n{e}")
        print(f"  q_norm[0, :8]:     {q_norm[0, :8]}")
        print(f"  ref_q_norm[0, :8]: {ref_q_norm[0, :8]}")
        print(f"  kv_norm[0, :8]:     {kv_norm[0, :8]}")
        print(f"  ref_kv_norm[0, :8]: {ref_kv_norm[0, :8]}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_mla_v4_q_kv_rmsnorm_testmode()
