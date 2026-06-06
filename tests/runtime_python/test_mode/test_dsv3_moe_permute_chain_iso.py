"""Isolation harness for the DSv3 MoE permute-chain corruption at H>=2048.

Parameterized hidden size + bypass mode to pinpoint whether the corruption
lives in quantize_fp8_sm100 (input scale write layout) or moe_permute_sm100
(input scale read layout).

Env knobs:
  H_SIZE    hidden size (default 2048; 16 K-blocks, K_PACKED=4)
  ISO_MODE  one of:
      full          kernel-quantize  + kernel-permute   (the real chain)
      pyq           python-quantize  + kernel-permute   (row-major in_scale)
      pyperm        kernel-quantize  + python-permute    (col-major read)

Run on gpu0:
  CUDA_VISIBLE_DEVICES=0 H_SIZE=2048 ISO_MODE=full   python <this>
"""

import os
import sys

import numpy as np
import torch

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
from test_dsv3_moe_permute_chain_testmode import _pack_group_weight, _dequant_group

FP8_MAX = 448.0


def test_iso():
    H = int(os.environ.get("H_SIZE", "2048"))
    mode = os.environ.get("ISO_MODE", "full")
    device = "cuda"
    torch.manual_seed(0)
    E = 4
    bm = 128
    m_total = E * bm
    I = 256
    mbt = 4
    topk = 2
    assert H % 128 == 0

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

    w13 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=device) * 0.05
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device=device) * 0.05
    x = torch.randn(mbt, H, dtype=torch.bfloat16, device=device) * 0.5
    residual = torch.randn(mbt, H, dtype=torch.bfloat16, device=device) * 0.1

    w13_fp8, w13_scale, w13_s2d = _pack_group_weight(w13)
    w2_fp8, w2_scale, w2_s2d = _pack_group_weight(w2)

    # Reference
    xq, xs = quantize_to_fp8_packed_ue8m0(x)       # row-major (mbt, K_PACKED)
    x_dq = dequant_from_packed_ue8m0(xq, xs)
    w13_dq = _dequant_group(w13_fp8, w13_s2d)
    w2_dq = _dequant_group(w2_fp8, w2_s2d)
    routed = torch.zeros(mbt, H, dtype=torch.float32, device=device)
    for t in range(mbt):
        for s in range(topk):
            e = int(topk_idx[t, s])
            gu = x_dq[t] @ w13_dq[e].t()
            act = torch.nn.functional.silu(gu[:I]) * gu[I:]
            aq, asc = quantize_to_fp8_packed_ue8m0(act.bfloat16().unsqueeze(0))
            a_dq = dequant_from_packed_ue8m0(aq, asc)[0]
            routed[t] += weights[s].item() * (a_dq @ w2_dq[e].t())
    ref = (residual.float() + routed).to(torch.bfloat16)

    K_PACKED = (H // 128 + 3) // 4
    K_PACKED_I = (I // 128 + 3) // 4
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=num_workers,
                  num_local_schedulers=num_schedulers, mpi_rank=0, world_size=1,
                  max_num_batched_tokens=mbt, max_num_batched_requests=mbt)
    pk = PersistentKernel(**params)

    from mirage.core import (bfloat16 as _bf16, float8_e4m3 as _fp8,
                             uint32 as _u32, int32 as _i32)

    # ---- Precompute python buffers for bypass variants ----
    in_fp8_py = xq.view(torch.uint8)                       # (mbt, H)
    in_scale_rm = xs.to(torch.uint32)                      # (mbt, K_PACKED) row-major
    # The kernel quantize_fp8_sm100 writes the input scale COLUMN-MAJOR
    # [K_PACKED, MBT_ALIGNED] (out[sf*MBT_ALIGNED + t]). To mimic the kernel
    # producer for the pyq variant, pack the row-major python scale into a
    # column-major (mbt, K_PACKED) buffer with stride MBT_ALIGNED.
    mbt_aligned = ((mbt + 3) // 4) * 4
    in_scale_cm = torch.zeros(mbt, K_PACKED, dtype=torch.uint32, device=device)
    in_scale_cm_flat = in_scale_cm.view(-1)
    for t in range(mbt):
        for sf in range(K_PACKED):
            in_scale_cm_flat[sf * mbt_aligned + t] = in_scale_rm[t, sf]

    # python permute (matches kernel semantics): build (m_total, H) fp8 +
    # (K_PACKED, m_total) transposed packed scale + meta.
    def py_permute(in_fp8_u8, in_scale_rowmajor):
        perm_fp8 = torch.zeros(m_total, H, dtype=torch.uint8, device=device)
        perm_scale = torch.zeros(K_PACKED, m_total, dtype=torch.uint32, device=device)
        meta = torch.zeros(2, m_total + mbt * topk, dtype=torch.int32, device=device)
        out_w = meta[0, :m_total]
        tok2perm = meta[0, m_total:]
        for e in range(E):
            slot = 0
            for t in range(mbt):
                rv = int(routing_indices[e, t].item())
                if rv > 0:
                    row = e * bm + slot
                    perm_fp8[row] = in_fp8_u8[t]
                    for sf in range(K_PACKED):
                        perm_scale[sf, row] = in_scale_rowmajor[t, sf]
                    k_slot = rv - 1
                    out_w[row] = topk_weights[t, k_slot].view(torch.int32)
                    tok2perm[t * topk + k_slot] = row + 1
                    slot += 1
        return perm_fp8, perm_scale, meta

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

    # tensors that may be python-supplied
    if mode in ("pyq",):
        in_fp8_attached = pk.attach_input(in_fp8_py, name="in_fp8_py")
        # column-major scale, matching the kernel-quantize producer the
        # (fixed) permute reads.
        in_scale_attached = pk.attach_input(in_scale_cm, name="in_scale_py")
    if mode in ("pyperm",):
        pf, ps, mt = py_permute(in_fp8_py, in_scale_rm)
        perm_fp8_attached = pk.attach_input(pf, name="perm_fp8_py")
        perm_scale_attached = pk.attach_input(ps, name="perm_scale_py")
        meta_attached = pk.attach_input(mt, name="meta_py")

    if mode == "probe_quant":
        # Run ONLY the kernel quantize, write its scale into an attached torch
        # tensor so we can inspect the layout post-run. Output scale buffer is
        # over-allocated (K_PACKED, aligned_batch) flattened so both layouts fit.
        aligned_batch = ((mbt + 3) // 4) * 4
        probe_fp8 = torch.zeros(mbt, H, dtype=torch.float8_e4m3fn, device=device)
        probe_scale = torch.zeros(mbt, K_PACKED, dtype=torch.uint32, device=device)
        probe_fp8_dt = pk.attach_input(probe_fp8, name="probe_fp8")
        probe_scale_dt = pk.attach_input(probe_scale, name="probe_scale")
        with pk.compile_scope():
            pk.quantize_fp8_layer(input=x_dt, output_fp8=probe_fp8_dt,
                                  output_scale=probe_scale_dt,
                                  grid_dim=(mbt, 1, 1), block_dim=(128, 1, 1),
                                  scale_ue8m0=True)
        print(f"[ISO] H={H} K_PACKED={K_PACKED} mode=probe_quant compiling...")
        pk.compile(output_dir=os.path.dirname(__file__))
        pk()
        torch.cuda.synchronize()
        ker = probe_scale.cpu().numpy()                 # interpreted (mbt, K_PACKED)
        py_rm = in_scale_rm.cpu().numpy()               # row-major (mbt, K_PACKED)
        # column-major: kernel wrote out[sf*aligned_batch + t]; flat layout in a
        # (mbt, K_PACKED) buffer means flat index r*K_PACKED + c. Reconstruct what
        # row-major value lands where if the kernel used col-major addressing.
        flat = ker.reshape(-1)
        col_as_rm = np.zeros((mbt, K_PACKED), dtype=np.uint32)
        for t in range(mbt):
            for sf in range(K_PACKED):
                idx = sf * aligned_batch + t
                if idx < flat.size:
                    col_as_rm[t, sf] = flat[idx]
        print("kernel scale (as (mbt,K_PACKED)):\n", ker)
        print("python row-major scale:\n", py_rm)
        match_rm = np.array_equal(ker, py_rm)
        match_col = np.array_equal(col_as_rm, py_rm)
        print(f"[PROBE] kernel==python(row-major)? {match_rm}")
        print(f"[PROBE] kernel(col-major-decoded)==python(row-major)? {match_col}")
        pk.finalize()
        return

    with pk.compile_scope():
        if mode == "full":
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
        elif mode == "pyq":
            meta = pk.new_tensor(dims=(2, m_total + mbt * topk), dtype=_i32, name="meta")
            pk.tensor_init_layer(target=meta, dummy=in_fp8_attached, grid_dim=(1, 1, 1),
                                 block_dim=(128, 1, 1), dummy_input_map=(-1, -1, -1),
                                 target_input_map=(-1, -1, -1))
            perm_fp8 = pk.new_tensor(dims=(m_total, H), dtype=_fp8, name="perm_fp8")
            perm_scale = pk.new_tensor(dims=(K_PACKED, m_total), dtype=_u32, name="perm_scale")
            permute.compile(in_fp8_attached, in_scale_attached, tw_dt, ri_dt,
                            perm_fp8, perm_scale, meta)
        elif mode == "pyperm":
            perm_fp8 = perm_fp8_attached
            perm_scale = perm_scale_attached
            meta = meta_attached
        else:
            raise ValueError(mode)

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

    print(f"[ISO] H={H} K_PACKED={K_PACKED} mode={mode} compiling...")
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
    print(f"[ISO] H={H} mode={mode} max abs {max_abs:.5f} rel {max_rel:.5f}")
    ok = max_rel < 0.15
    print("PASSED" if ok else "FAILED")
    pk.finalize()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    test_iso()
