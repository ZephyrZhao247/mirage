"""V4-Flash ``apply_rotary_emb`` test (multi-batch).

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/apply_rotary_emb.md``.
Catalog: ``mirage.mpk.layers.deepseek_v4.std.V4ApplyRotaryEmb`` (NEW
naive GPT-J / interleaved RoPE; task ``apply_rotary_emb_v4_sm100``).

This test runs the V4 RoPE catalog through the full MPK compile +
execute pipeline in ``test_mode=True`` on a multi-batch input
(num_tokens * num_heads >= 2) and compares the kernel output against the
catalog's PyTorch ``forward()`` reference. The catalog flattens the
leading ``(num_tokens, num_heads)`` axes into a single ``num_rows`` axis
before invoking the kernel.

Run on a free GPU:
    CUDA_VISIBLE_DEVICES=0 python tests/runtime_python/layers/test_apply_rotary_emb_v4.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.layers.deepseek_v4.std import V4ApplyRotaryEmb
from mirage.mpk.persistent_kernel import PersistentKernel


def test_v4_apply_rotary_emb_multibatch():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # V4-Flash uses head_dim = rotary_dim = 64 on the qk_rope path
    # (qk_rope_head_dim=64). Multi-batch from day 1: num_tokens >= 2 and
    # num_heads >= 2.
    num_tokens = 4
    num_heads = 2
    head_dim = 64
    rotary_dim = 64  # V4-Flash typically sets rotary_dim == head_dim.
    num_rows = num_tokens * num_heads

    # Random input tensor. The catalog accepts (num_rows, head_dim) 2-D
    # input; we pass it directly in flattened form (multi-batch is just
    # multi-row).
    x = torch.randn(num_rows, head_dim, dtype=dtype, device=device)

    # Per-row cos/sin tables. In production, these are gathered from a
    # length-max_pos cos/sin table at position p indexing per token row;
    # the per-(token,head) row is broadcast across heads. For the test
    # we use freely-randomized cos/sin in the legal range — the kernel
    # does not care that they actually form a unit vector.
    angles = torch.rand(num_rows, rotary_dim // 2, dtype=torch.float32, device=device) \
             * (2.0 * torch.pi)
    cos = angles.cos().to(dtype)
    sin = angles.sin().to(dtype)

    out_buf = torch.zeros(num_rows, head_dim, dtype=dtype, device=device)

    module = V4ApplyRotaryEmb(
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        prefix="v4rope_",
    )

    # PyTorch reference via the catalog's faithful forward().
    ref = module.forward(x, cos, sin)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    # max_num_batched_requests must be >= num_rows so the runtime
    # accepts our request grid; multi-batch from day 1.
    params["max_num_batched_tokens"] = num_rows
    params["max_num_batched_requests"] = num_rows
    pk = PersistentKernel(**params)

    x_dt = pk.attach_input(x, name="v4rope_x")
    cos_dt = pk.attach_input(cos, name="v4rope_cos")
    sin_dt = pk.attach_input(sin, name="v4rope_sin")

    with pk.compile_scope():
        _ = module.compile(x_dt, cos_dt, sin_dt, output=out_buf)

    print("Compiling V4 apply_rotary_emb test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4 apply_rotary_emb test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"out_buf[0, :8]: {out_buf[0, :8]}")
    print(f"ref[0, :8]:     {ref[0, :8]}")
    print(f"out_buf[-1, -8:]: {out_buf[-1, -8:]}")
    print(f"ref[-1, -8:]:     {ref[-1, -8:]}")

    max_diff = (out_buf.float() - ref.float()).abs().max().item()
    print(f"Max absolute difference: {max_diff}")

    try:
        # RoPE is fp32 mul-add then bf16 store; bf16 ULP tolerance.
        torch.testing.assert_close(out_buf, ref, atol=0.05, rtol=0.05)
        print(
            "PASSED: V4ApplyRotaryEmb compile() matches forward() "
            "(multi-batch interleaved RoPE)."
        )
    except AssertionError as e:
        print(f"FAILED: V4ApplyRotaryEmb disagrees with reference\n{e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_v4_apply_rotary_emb_multibatch()
