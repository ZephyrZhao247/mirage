"""V4-Flash ``deepseek_v4_fp8_einsum`` (wo_a o-projection, NEW) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/deepseek_v4_fp8_einsum.md``.

Consumes the layout produced by wave-5A's
``fused_inv_rope_fp8_quant``: ``o_fp8`` is `[T, n_groups, d_in]`
float8_e4m3fn, ``o_scale`` is `[T, n_groups, scale_inner]` int32
UE8M0-packed (4 exponent bytes per int32).

Multi-batch from day 1 (``max_num_batched_requests = 2``, 4 tokens).
"""
import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.attention import V4DeepseekFP8Einsum


def _quantize_block_fp8_ue8m0(x: torch.Tensor, block_size: int = 128):
    """Block-FP8-quantize an (..., D) fp32 tensor with UE8M0 scales.

    Returns (fp8_e4m3fn bytes, packed-int32 scales).
    """
    leading = x.shape[:-1]
    D = x.shape[-1]
    assert D % block_size == 0
    n_blocks = D // block_size
    chunks = x.view(*leading, n_blocks, block_size)
    absmax = chunks.abs().amax(dim=-1).clamp(min=1e-10)
    raw = absmax / 448.0
    # UE8M0 exponent = ceil(log2(raw)).
    exponent = torch.ceil(torch.log2(raw)).to(torch.int32)
    enc = (exponent + 127).clamp(0, 255).to(torch.int32)
    scale = torch.pow(torch.tensor(2.0), exponent.to(torch.float32))
    fp8 = (chunks / scale.unsqueeze(-1)).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    fp8_out = fp8.view(*leading, D)

    # Pack 4 UE8M0 bytes into one int32 along the n_blocks axis.
    pad = (4 - n_blocks % 4) % 4
    if pad > 0:
        enc = torch.nn.functional.pad(enc, (0, pad))
    packed_count = enc.shape[-1] // 4
    enc4 = enc.view(*leading, packed_count, 4)
    weights = torch.tensor([1, 256, 65536, 16777216], dtype=torch.int32, device=x.device)
    packed = (enc4 * weights).sum(dim=-1).to(torch.int32)
    # Return packed as uint32 view for parity with the producer's int32.
    return fp8_out, packed.view(torch.uint32)


def test_deepseek_v4_fp8_einsum_v4_testmode():
    device = "cuda"
    torch.manual_seed(0)

    T = 4
    n_groups = 2
    d_in = 256        # = heads_per_group * head_dim (small for test)
    d_out = 64        # = o_lora_rank (small for test)
    quant_block = 128

    # Synthetic fp32 references.
    o_fp32 = torch.randn(T, n_groups, d_in, dtype=torch.float32, device=device) * 0.1
    w_fp32 = torch.randn(n_groups, d_out, d_in, dtype=torch.float32, device=device) * 0.05

    # Block-quantize both into FP8 + UE8M0 packed scales.
    o_fp8, o_scale = _quantize_block_fp8_ue8m0(o_fp32, block_size=quant_block)
    w_fp8, w_scale = _quantize_block_fp8_ue8m0(w_fp32, block_size=quant_block)

    # Build module and seed its weights.
    m = V4DeepseekFP8Einsum(
        n_groups=n_groups, d_in=d_in, d_out=d_out,
        quant_block_size=quant_block, prefix="v4_eins_",
    )
    m.wo_a_fp8.data = w_fp8.view(torch.uint8).contiguous().to(device)
    m.wo_a_scale.data = w_scale.contiguous().to(device)

    # Eager reference.
    ref = m.forward(o_fp8.view(torch.uint8), o_scale)

    # PersistentKernel.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = T
    params["max_num_batched_requests"] = max(2, T)
    pk = PersistentKernel(**params)

    # Attach o_fp8 (uint8 viewable) and o_scale (uint32).
    o_fp8_dt = pk.attach_input(
        o_fp8.view(torch.uint8).contiguous(), name="o_fp8"
    )
    o_scale_dt = pk.attach_input(o_scale.contiguous(), name="o_scale")

    out_buf = torch.zeros(T, n_groups, d_out, dtype=torch.bfloat16, device=device)
    out_dt = pk.attach_input(out_buf, name="einsum_out")

    with pk.compile_scope():
        _ = m.compile(o_fp8_dt, o_scale_dt, output=out_dt)

    print("Compiling V4DeepseekFP8Einsum test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4DeepseekFP8Einsum test kernel...")
    pk()
    torch.cuda.synchronize()

    diff = (out_buf.float() - ref.float()).abs()
    print(f"out[0,0,:4]    : {out_buf[0,0,:4]}")
    print(f"ref[0,0,:4]    : {ref[0,0,:4]}")
    print(f"max-abs diff   : {diff.max().item():.4f}")

    try:
        # FP8 + block-scale arithmetic; bf16 round-trip; allow 5% rel.
        torch.testing.assert_close(out_buf.float(), ref.float(),
                                   atol=0.2, rtol=0.05)
        print("PASSED: V4DeepseekFP8Einsum compile() matches forward()")
    except AssertionError as e:
        print(f"FAILED: {e}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_deepseek_v4_fp8_einsum_v4_testmode()
