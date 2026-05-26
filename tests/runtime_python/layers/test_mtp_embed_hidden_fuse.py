"""Catalog test: ``layers.mtp.MTPEmbedHiddenFuse`` via PersistentKernel test_mode.

V4-Flash MTPBlock input fuse — embed projection plus per-HC hidden
projection, decomposed over existing ``Linear`` + ``elementwise_add``
catalog primitives.

PyTorch oracle:

    fused[t, j, d] = (e @ e_proj.weight.T)[t, d]
                   + (h[t, j, :] @ h_proj.weight.T)[d]
"""

import os
import sys

import torch

import mirage
from mirage.mpk import layers
from mirage.mpk.persistent_kernel import PersistentKernel


def test_mtp_embed_hidden_fuse_testmode():
    device = "cuda"
    torch.manual_seed(0)

    T = 4
    HC = 4
    D = 128

    # Random inputs and weights.
    e_in = torch.randn(T, D, dtype=torch.bfloat16, device=device)
    h_in = torch.randn(T, HC, D, dtype=torch.bfloat16, device=device)

    module = layers.MTPEmbedHiddenFuse(
        hidden_size=D, hc_mult=HC, prefix="t_"
    )
    module = module.to(device=device, dtype=torch.bfloat16)
    # Random init for the projection weights.
    with torch.no_grad():
        module.e_proj.weight.copy_(
            torch.randn(D, D, dtype=torch.bfloat16, device=device) * 0.05
        )
        module.h_proj.weight.copy_(
            torch.randn(D, D, dtype=torch.bfloat16, device=device) * 0.05
        )

    # ---- PyTorch reference -------------------------------------------
    ref_fused = module.forward(e_in, h_in)   # [T, HC, D] bf16

    # ---- Output buffers for host readback ----------------------------
    # One contiguous [T, D] buffer per HC copy (the catalog API expects
    # per-HC inputs/outputs because pk.attach_input requires contiguous
    # row-major; a strided 3-D view does not satisfy that).
    fused_per_hc = [
        torch.zeros(T, D, dtype=torch.bfloat16, device=device)
        for _ in range(HC)
    ]
    # Per-HC contiguous slices of h.
    h_per_hc_torch = [h_in[:, j, :].contiguous() for j in range(HC)]

    # ---- Build the PK in test mode -----------------------------------
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

    e_dt = pk.attach_input(e_in, name="e_in")
    h_per_hc_dts = [
        pk.attach_input(h_per_hc_torch[j], name=f"h_in_hc{j}")
        for j in range(HC)
    ]

    with pk.compile_scope():
        module.compile(
            e_dt,
            h_per_hc_dts,
            fused_per_hc=fused_per_hc,
            block_dim=(128, 1, 1),
        )

    print("Compiling test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))

    print("Running test kernel...")
    pk()
    torch.cuda.synchronize()

    # ---- Stitch per-HC outputs back into [T, HC, D] for comparison ---
    fused_out = torch.stack(fused_per_hc, dim=1)  # [T, HC, D]

    diff = (fused_out.float() - ref_fused.float()).abs().max().item()
    print(f"fused max abs diff: {diff:.6e}")

    try:
        torch.testing.assert_close(
            fused_out.float(), ref_fused.float(), rtol=1e-2, atol=1e-2
        )
        print("PASSED: MTPEmbedHiddenFuse compile() matches forward()")
    except AssertionError as exc:
        print(f"FAILED: MTPEmbedHiddenFuse compile() disagrees with forward()\n{exc}")
        pk.finalize()
        sys.exit(1)

    pk.finalize()


if __name__ == "__main__":
    test_mtp_embed_hidden_fuse_testmode()
