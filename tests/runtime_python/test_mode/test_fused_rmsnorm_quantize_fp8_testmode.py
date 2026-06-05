"""Test: FusedRMSNormQuantizeFP8 catalog layer via PersistentKernel test_mode.

Drives the catalog module's ``compile()`` (registers
``fused_rmsnorm_quantize_fp8_sm100``) and compares against the module's own
``forward()`` reference — the bottom-up correctness contract used by the
DeepSeek V3 new-API modeling.

Two modes (matching the two DeepSeek V3 call sites):
  1. full-width input-layernorm → qkv_a  (process_dim=None, emit_bf16=True)
  2. inner q_a-layernorm slice → q_b      (process_dim=q_lora_rank, emit_bf16=False)

Run:
    CUDA_VISIBLE_DEVICES=3 python tests/runtime_python/test_mode/test_fused_rmsnorm_quantize_fp8_testmode.py
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers import FusedRMSNormQuantizeFP8


def _make_pk(batch):
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True,
        num_workers=num_workers,
        num_local_schedulers=num_schedulers,
        mpi_rank=0,
        world_size=1,
        max_num_batched_tokens=batch,
        max_num_batched_requests=batch,
    )
    return PersistentKernel(**params)


def _dequant_f32_scale(fp8_u8: torch.Tensor, scale_f32: torch.Tensor) -> torch.Tensor:
    """Dequantize (rows, K) FP8 bytes with (rows, K//128) f32 block scales."""
    rows, K = fp8_u8.shape
    ng = K // 128
    vals = fp8_u8.view(torch.float8_e4m3fn).float().reshape(rows, ng, 128)
    return (vals * scale_f32.reshape(rows, ng, 1)).reshape(rows, K)


def _run_mode(name, batch, in_width, process_dim, emit_bf16):
    device = "cuda"
    torch.manual_seed(0)
    width = in_width if process_dim is None else process_dim

    layer = FusedRMSNormQuantizeFP8(width, scale_ue8m0=False).to(device, torch.bfloat16)
    layer.weight.data.normal_(mean=1.0, std=0.02)

    x = (torch.randn(batch, in_width, device=device, dtype=torch.bfloat16) * 0.5)

    # Reference (fp32 normed value over the active slice).
    sliced = x[:, :width].float()
    var = sliced.pow(2).mean(dim=-1, keepdim=True)
    ref_normed = (sliced * torch.rsqrt(var + layer.eps)) * layer.weight[:width].float()

    out_bf16 = torch.zeros(batch, in_width, dtype=torch.bfloat16, device=device)
    out_fp8 = torch.zeros(batch, width, dtype=torch.float8_e4m3fn, device=device)
    out_scale = torch.zeros(batch, width // 128, dtype=torch.float32, device=device)

    pk = _make_pk(batch)
    x_dt = pk.attach_input(x, name="x")
    bf16_dt = pk.attach_input(out_bf16, name="out_bf16")
    fp8_dt = pk.attach_input(out_fp8, name="out_fp8")
    scale_dt = pk.attach_input(out_scale, name="out_scale")

    with pk.compile_scope():
        layer.compile(
            x_dt,
            process_dim=None if process_dim == in_width else process_dim,
            output_bf16=bf16_dt,
            output_fp8=fp8_dt,
            output_scale=scale_dt,
            emit_bf16=emit_bf16,
            scale_ue8m0=False,
        )

    print(f"[{name}] compiling...")
    pk.compile(output_dir=os.path.dirname(__file__))
    pk()
    torch.cuda.synchronize()

    deq = _dequant_f32_scale(out_fp8, out_scale)[:, :width]
    fp8_abs = (deq.float() - ref_normed).abs().max().item()
    fp8_rel = fp8_abs / max(ref_normed.abs().max().item(), 1e-6)
    print(f"[{name}] fp8 dequant max abs {fp8_abs:.5f} rel {fp8_rel:.5f}")
    ok = fp8_rel < 0.06

    if emit_bf16:
        bf16_abs = (out_bf16[:, :width].float() - ref_normed).abs().max().item()
        bf16_rel = bf16_abs / max(ref_normed.abs().max().item(), 1e-6)
        print(f"[{name}] bf16 out max abs {bf16_abs:.5f} rel {bf16_rel:.5f}")
        ok = ok and bf16_rel < 0.02

    print(f"[{name}] {'PASSED' if ok else 'FAILED'}")
    pk.finalize()
    return ok


def test_full_width_qkv_a():
    # DeepSeek V3 hidden = 7168 (qkv_a input). 7168/128=56, 56%4==0.
    return _run_mode("qkv_a", batch=8, in_width=7168, process_dim=7168, emit_bf16=True)


def test_inner_q_a_slice():
    # Inner q_a layernorm: input buffer wider than the q_lora_rank=1536 slice.
    return _run_mode("q_a_slice", batch=8, in_width=2176, process_dim=1536, emit_bf16=False)


if __name__ == "__main__":
    ok = True
    ok &= test_full_width_qkv_a()
    ok &= test_inner_q_a_slice()
    if not ok:
        sys.exit(1)
    print("\nAll FusedRMSNormQuantizeFP8 test-mode checks PASSED!")
