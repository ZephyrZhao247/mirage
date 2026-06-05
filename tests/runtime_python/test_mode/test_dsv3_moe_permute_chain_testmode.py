"""Isolation test for the DeepSeek V3 MoE permute-path chain (FP8).

Exercises the grouped-GEMM permute pipeline with FIXED routing (no router
GEMM / routing kernel), comparing the MoEUnpermute output against a manual
PyTorch reference of the same routed-expert computation:

  quantize(x) -> MoEPermute -> FP8GroupGEMM(w13) -> MoESiluMul ->
  quantize -> FP8GroupGEMM(w2) -> MoEUnpermute(combine + residual)

This de-risks the permute path (which has no existing test-mode coverage)
before composing it into DeepseekV3MoE.

Run:
  CUDA_VISIBLE_DEVICES=<free> python tests/runtime_python/test_mode/test_dsv3_moe_permute_chain_testmode.py
"""

import os
import sys

import numpy as np
import torch

# Verified UE8M0 helpers from the blackwell common dir.
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "blackwell", "common"))

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers import (
    MoEPermute, MoEUnpermute, FP8GroupGEMMSmallM, MoESiluMul,
)
from sm100_fp8_scale_layout import (  # noqa: E402
    encode_ue8m0, quantize_to_fp8_packed_ue8m0, dequant_from_packed_ue8m0,
)

FP8_MAX = 448.0


def _pack_group_weight(w_bf16, block=128):
    """Quantize expert weights (E, N, K) with 128x128 blocks → FP8 (uint8) +
    K-outer UE8M0 packed scale (num_sf_k, E*N) — the FP8GroupGEMMSmallM layout."""
    E, N, K = w_bf16.shape
    nb, kb = N // block, K // block
    num_sf_k = (kb + 3) // 4
    w = w_bf16.float()
    fp8 = torch.empty(E, N, K, dtype=torch.float8_e4m3fn, device=w.device)
    scale2d = torch.empty(E, nb, kb, dtype=torch.float32, device=w.device)
    for e in range(E):
        for bi in range(nb):
            for ki in range(kb):
                blk = w[e, bi*block:(bi+1)*block, ki*block:(ki+1)*block]
                amax = blk.abs().max().item()
                s = amax / FP8_MAX if amax > 0 else 1.0
                # snap to UE8M0 power-of-2 so dequant matches the kernel.
                enc = encode_ue8m0(s)
                snapped = 2.0 ** (enc - 127)
                scale2d[e, bi, ki] = snapped
                fp8[e, bi*block:(bi+1)*block, ki*block:(ki+1)*block] = (
                    (blk / snapped).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn))
    # Pack on CPU via numpy (torch lacks CUDA uint32 bitwise-or).
    packed_np = np.zeros((num_sf_k, E * N), dtype=np.uint32)
    s2d_cpu = scale2d.cpu()
    for e in range(E):
        for n in range(N):
            for k in range(kb):
                enc = encode_ue8m0(float(s2d_cpu[e, n // block, k]))
                sk = k // 4
                packed_np[sk, e * N + n] |= np.uint32((enc & 0xFF) << ((k % 4) * 8))
    packed = torch.from_numpy(packed_np).to(w.device)
    return fp8.view(torch.uint8), packed, scale2d


def _dequant_group(fp8_u8, scale2d, block=128):
    """Dequant (E,N,K) fp8 with 128x128 scale (E, N/128, K/128) → fp32."""
    E, N, K = fp8_u8.shape
    w = fp8_u8.view(torch.float8_e4m3fn).float()
    nb, kb = N // block, K // block
    s = scale2d.reshape(E, nb, 1, kb, 1)
    return (w.reshape(E, nb, block, kb, block) * s).reshape(E, N, K)


def test_moe_permute_chain():
    device = "cuda"
    torch.manual_seed(0)
    E = 4              # local experts
    bm = 128
    m_total = E * bm
    H = 256            # hidden  (256/128=2 → K_PACKED=1)
    I = 256            # moe intermediate
    mbt = 4
    topk = 2

    # ---- Fixed routing: token t → experts [t%E, (t+1)%E], weights [0.6,0.4]
    weights = torch.tensor([0.6, 0.4], dtype=torch.float32)
    topk_idx = torch.zeros(mbt, topk, dtype=torch.long)
    routing_indices = torch.zeros(E, mbt, dtype=torch.int32, device=device)
    topk_weights = torch.zeros(mbt, topk, dtype=torch.float32, device=device)
    for t in range(mbt):
        for s in range(topk):
            e = (t + s) % E
            topk_idx[t, s] = e
            routing_indices[e, t] = s + 1
            topk_weights[t, s] = weights[s]

    # ---- Weights + input
    w13 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=device) * 0.05
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device=device) * 0.05
    x = torch.randn(mbt, H, dtype=torch.bfloat16, device=device) * 0.5
    residual = torch.randn(mbt, H, dtype=torch.bfloat16, device=device) * 0.1

    w13_fp8, w13_scale, w13_s2d = _pack_group_weight(w13)
    w2_fp8, w2_scale, w2_s2d = _pack_group_weight(w2)

    # ---- Reference (dequantized FP8, matching what the kernel computes)
    xq, xs = quantize_to_fp8_packed_ue8m0(x)
    x_dq = dequant_from_packed_ue8m0(xq, xs)          # (mbt, H)
    w13_dq = _dequant_group(w13_fp8, w13_s2d)          # (E, 2I, H)
    w2_dq = _dequant_group(w2_fp8, w2_s2d)             # (E, H, I)
    routed = torch.zeros(mbt, H, dtype=torch.float32, device=device)
    for t in range(mbt):
        for s in range(topk):
            e = int(topk_idx[t, s])
            gu = x_dq[t] @ w13_dq[e].t()               # (2I,)
            act = torch.nn.functional.silu(gu[:I]) * gu[I:]
            # quantize act to fp8 (the kernel re-quantizes silu output)
            aq, asc = quantize_to_fp8_packed_ue8m0(act.bfloat16().unsqueeze(0))
            a_dq = dequant_from_packed_ue8m0(aq, asc)[0]
            routed[t] += weights[s].item() * (a_dq @ w2_dq[e].t())
    ref = (residual.float() + routed).to(torch.bfloat16)

    # ---- Build the MPK permute chain
    K_PACKED = (H // 128 + 3) // 4
    K_PACKED_I = (I // 128 + 3) // 4
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=num_workers,
                  num_local_schedulers=num_schedulers, mpi_rank=0, world_size=1,
                  max_num_batched_tokens=mbt, max_num_batched_requests=mbt)
    pk = PersistentKernel(**params)

    from mirage.core import bfloat16 as _bf16, float8_e4m3 as _fp8, uint32 as _u32, int32 as _i32

    x_dt = pk.attach_input(x, name="x")
    res_dt = pk.attach_input(residual, name="residual")
    ri_dt = pk.attach_input(routing_indices, name="routing_indices")
    tw_dt = pk.attach_input(topk_weights, name="topk_weights")
    m_indices = (torch.arange(m_total, dtype=torch.int32, device=device) // bm)
    mi_dt = pk.attach_input(m_indices, name="m_indices")

    w13_l = FP8GroupGEMMSmallM(E, in_features=H, out_features=2 * I, prefix="w13_").to(device)
    w2_l = FP8GroupGEMMSmallM(E, in_features=I, out_features=H, prefix="w2_").to(device)
    with torch.no_grad():
        w13_l.weight.copy_(w13_fp8); w13_l.weight_scale.copy_(w13_scale)
        w2_l.weight.copy_(w2_fp8); w2_l.weight_scale.copy_(w2_scale)
    permute = MoEPermute(E, H, topk, bm_padding=bm, prefix="perm_")
    unpermute = MoEUnpermute(H, prefix="unperm_")
    silu = MoESiluMul(I, prefix="silu_")

    out = torch.zeros(mbt, H, dtype=torch.bfloat16, device=device)
    out_dt = pk.attach_input(out, name="out")

    with pk.compile_scope():
        in_fp8 = pk.new_tensor(dims=(mbt, H), dtype=_fp8, name="in_fp8")
        in_scale = pk.new_tensor(dims=(mbt, K_PACKED), dtype=_u32, name="in_scale")
        pk.quantize_fp8_layer(input=x_dt, output_fp8=in_fp8, output_scale=in_scale,
                              grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1), scale_ue8m0=True)
        meta = pk.new_tensor(dims=(2, m_total + mbt * topk), dtype=_i32, name="meta")
        pk.tensor_init_layer(target=meta, dummy=in_fp8, grid_dim=(1, 1, 1),
                             block_dim=(128, 1, 1), dummy_input_map=(-1, -1, -1),
                             target_input_map=(-1, -1, -1))
        perm_fp8 = pk.new_tensor(dims=(m_total, H), dtype=_fp8, name="perm_fp8")
        perm_scale = pk.new_tensor(dims=(K_PACKED, m_total), dtype=_u32, name="perm_scale")
        permute.compile(in_fp8, in_scale, tw_dt, ri_dt, perm_fp8, perm_scale, meta)
        w13_out = pk.new_tensor(dims=(m_total, 2 * I), dtype=_bf16, name="w13_out")
        w13_l.compile(perm_fp8, perm_scale, mi_dt, w13_out, num_workers=num_workers)
        silu_out = pk.new_tensor(dims=(m_total, I), dtype=_bf16, name="silu_out")
        silu.compile(w13_out, output=silu_out)
        silu_fp8 = pk.new_tensor(dims=(m_total, I), dtype=_fp8, name="silu_fp8")
        silu_scale = pk.new_tensor(dims=(K_PACKED_I, m_total), dtype=_u32, name="silu_scale")
        pk.quantize_fp8_layer(input=silu_out, output_fp8=silu_fp8, output_scale=silu_scale,
                              grid_dim=(m_total, 1, 1), block_dim=(128, 1, 1),
                              scale_ue8m0=True, process_all_rows=True)
        w2_out = pk.new_tensor(dims=(m_total, H), dtype=_bf16, name="w2_out")
        w2_l.compile(silu_fp8, silu_scale, mi_dt, w2_out, num_workers=num_workers)
        unpermute.compile(permuted_output=w2_out, meta=meta, residual=res_dt, output=out_dt)

    print("Compiling MoE permute chain...")
    pk.compile(output_dir=os.path.dirname(__file__))
    pk()
    torch.cuda.synchronize()

    if out.isnan().any() or out.isinf().any():
        print(f"FAILED: NaN/Inf (out[0,:6]={out[0,:6]})")
        pk.finalize(); sys.exit(1)
    max_abs = (out.float() - ref.float()).abs().max().item()
    max_rel = max_abs / max(ref.float().abs().max().item(), 1e-6)
    print(f"out[0,:6]: {out[0,:6]}")
    print(f"ref[0,:6]: {ref[0,:6]}")
    print(f"max abs {max_abs:.5f} rel {max_rel:.5f}")
    ok = max_rel < 0.12
    print("PASSED" if ok else "FAILED")
    pk.finalize()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    test_moe_permute_chain()
