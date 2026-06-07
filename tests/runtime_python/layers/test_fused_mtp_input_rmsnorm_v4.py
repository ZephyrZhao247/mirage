"""V4-Flash ``fused_mtp_input_rmsnorm`` (NEW kernel) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/fused_mtp_input_rmsnorm.md``.

Joint per-token kernel that runs (HC_MULT + 1) RMS norms back-to-back:

* slot 0          : enorm(inputs_embeds[t, :]) with pos==0 zero-mask
* slot 1..HC_MULT : hnorm(prev_hidden[t, slot-1, :]), unconditional

Multi-batch from day 1 (``max_num_batched_requests = 4``). We pick the
positions vector to include a ``pos == 0`` token (token 0) AND a
non-zero token (tokens 1..3), so both branches of the pos-mask are
exercised.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.mtp import V4FusedMTPInputRMSNorm


def test_fused_mtp_input_rmsnorm_v4_testmode():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)

    # Shape per spec (V4-Flash): HIDDEN = 4096, HC_MULT = 8.
    # Multi-batch: 4 tokens. Tokens 0..3, with positions = [0, 1, 17, 42]
    # to cover BOTH branches of the pos mask (token 0 hits the
    # force_zero path; tokens 1..3 hit the regular RMSNorm path).
    num_tokens = 4
    hidden_size = 4096
    hc_mult = 8
    eps = 1e-6  # matches the kernel's hard-coded 1e-6f

    # ------------------------------------------------------------------
    # Build module + tensors
    # ------------------------------------------------------------------
    m = V4FusedMTPInputRMSNorm(
        hidden_size=hidden_size,
        hc_mult=hc_mult,
        eps=eps,
        prefix="v4_mtp_in_",
    )
    # Random bf16 weights (default init is all-ones; randn exercises the
    # scale path so we'd catch any silent identity-collapse).
    m.enorm_weight.data = m.enorm_weight.data.to(device=device, dtype=dtype)
    m.enorm_weight.data.copy_(
        torch.randn(hidden_size, dtype=dtype, device=device)
    )
    m.hnorm_weight.data = m.hnorm_weight.data.to(device=device, dtype=dtype)
    m.hnorm_weight.data.copy_(
        torch.randn(hidden_size, dtype=dtype, device=device)
    )

    inputs_embeds = torch.randn(
        num_tokens, hidden_size, dtype=dtype, device=device
    )
    positions = torch.tensor(
        [0, 1, 17, 42], dtype=torch.int64, device=device
    )
    previous_hidden_states = torch.randn(
        num_tokens, hc_mult, hidden_size, dtype=dtype, device=device
    )

    enorm_out_buf = torch.zeros(
        num_tokens, hidden_size, dtype=dtype, device=device
    )
    hnorm_out_buf = torch.zeros(
        num_tokens, hc_mult, hidden_size, dtype=dtype, device=device
    )

    # PyTorch reference (fp32 reduction, bf16 output, with pos==0 mask
    # on the enorm stream).
    ref_enorm, ref_hnorm = m.forward(
        inputs_embeds, positions, previous_hidden_states
    )

    # ------------------------------------------------------------------
    # Build PersistentKernel in test mode
    # ------------------------------------------------------------------
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

    inputs_embeds_dt = pk.attach_input(inputs_embeds, name="inputs_embeds")
    positions_dt = pk.attach_input(positions, name="positions")
    prev_hidden_dt = pk.attach_input(
        previous_hidden_states, name="previous_hidden_states"
    )

    with pk.compile_scope():
        _ = m.compile(
            inputs_embeds_dt,
            positions_dt,
            prev_hidden_dt,
            enorm_out=enorm_out_buf,
            hnorm_out=hnorm_out_buf,
        )

    # ------------------------------------------------------------------
    # Compile and run once
    # ------------------------------------------------------------------
    print("Compiling V4FusedMTPInputRMSNorm test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4FusedMTPInputRMSNorm test kernel...")
    pk()
    torch.cuda.synchronize()

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    print("=== enorm ===")
    print(f"enorm_out[:2, :8]: {enorm_out_buf[:2, :8]}")
    print(f"ref_enorm[:2, :8]: {ref_enorm[:2, :8]}")
    max_e = (enorm_out_buf.float() - ref_enorm.float()).abs().max().item()
    print(f"enorm max-abs diff: {max_e}")

    # Token 0 (pos == 0) MUST be exactly zero (vLLM pos-mask semantics).
    tok0_max = enorm_out_buf[0].abs().max().item()
    print(f"enorm token 0 (pos==0) max-abs: {tok0_max}")
    assert tok0_max == 0.0, (
        "pos==0 mask broken: token 0's enorm_out is not all-zero "
        f"(got max-abs = {tok0_max})"
    )

    print("=== hnorm ===")
    print(f"hnorm_out[0, 0, :8]: {hnorm_out_buf[0, 0, :8]}")
    print(f"ref_hnorm[0, 0, :8]: {ref_hnorm[0, 0, :8]}")
    max_h = (hnorm_out_buf.float() - ref_hnorm.float()).abs().max().item()
    print(f"hnorm max-abs diff: {max_h}")

    try:
        torch.testing.assert_close(enorm_out_buf, ref_enorm, atol=0.05, rtol=0.05)
        torch.testing.assert_close(hnorm_out_buf, ref_hnorm, atol=0.05, rtol=0.05)
        print("PASSED: V4FusedMTPInputRMSNorm compile() matches forward()")
    except AssertionError as e:
        print(
            "FAILED: V4FusedMTPInputRMSNorm compile() disagrees with forward()\n"
            f"{e}"
        )
        pk.finalize()
        sys.exit(1)

    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_fused_mtp_input_rmsnorm_v4_testmode()
