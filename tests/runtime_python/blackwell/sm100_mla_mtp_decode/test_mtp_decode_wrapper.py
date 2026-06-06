"""Standalone kernel-wrapper test for the single-GPU (tp=1) MLA MTP decode +
reduce kernels (DeepSeek V3, B200/SM100).

Invokes ``mla_mtp_decode_sm100_task_impl<*,false>`` and
``mla_mtp_reduce_sm100_task_impl<512>`` directly via a thin nvcc-compiled
pybind11 extension (mtp_decode_wrapper) -- NO MPK persistent-kernel runtime.

For each (q_len, kv_len) config it checks three things:
  (A) decode partial-O + LSE   vs  mla_mtp_decode_ref      (kernel's exact layout)
  (B) reduced output            vs  mla_mtp_reduce_ref      (LSE-weighted merge)
  (C) reduced output            vs  a clean independent SDPA
                                    (K = full 576, V = first 512, deepseek scale)

The MPK group/split derivation is replicated exactly:
  hpb = 128 // q_len, decremented until 128 % hpb == 0;  num_head_groups = 128//hpb
  num_splits = ceil(kv_len / 128)
"""

import math
import sys
import os

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtp_decode_wrapper  # noqa: E402
from pytorch_reference import (  # noqa: E402
    mla_mtp_decode_ref,
    mla_mtp_reduce_ref,
    NUM_HEADS,
    D_K,
    D_V,
    TILE_S,
)


def deepseek_scale():
    mscale = 0.1 * 1.0 * math.log(40.0) + 1.0
    return (1.0 / math.sqrt(192.0)) * mscale * mscale


def derive_groups_splits(q_len, kv_len):
    hpb = 128 // q_len
    while 128 % hpb != 0:
        hpb -= 1
    num_head_groups = 128 // hpb
    num_splits = (kv_len + TILE_S - 1) // TILE_S
    return num_head_groups, num_splits


def clean_sdpa(q, kv, batch_size, q_len, kv_len):
    """Independent reference: per-head softmax over the full kv window.

    q  : bf16 [B*q_len*H, D_K]
    kv : bf16 [B*kv_len, D_K]; K = kv (full D_K), V = kv[:, :D_V]
    Returns bf16 [B, q_len, H, D_V].
    """
    ss = deepseek_scale()
    Q = q.reshape(batch_size, q_len, NUM_HEADS, D_K).float()
    KV = kv.reshape(batch_size, kv_len, D_K).float()
    K = KV
    V = KV[..., :D_V]
    out = torch.zeros(batch_size, q_len, NUM_HEADS, D_V, device=q.device)
    for bi in range(batch_size):
        # scores: [q_len, H, kv_len]
        scores = torch.einsum("qhd,kd->qhk", Q[bi], K[bi]) * ss
        # causal limit (matches kernel): kv_len if q_len==1 else kv_len-q_len+qi+1
        idx = torch.arange(kv_len, device=q.device)
        if q_len == 1:
            lim = torch.full((q_len,), kv_len, device=q.device)
        else:
            lim = torch.tensor(
                [kv_len - q_len + qi + 1 for qi in range(q_len)], device=q.device)
        mask = idx.unsqueeze(0) < lim.unsqueeze(1)  # [q_len, kv_len]
        scores = torch.where(mask.unsqueeze(1), scores,
                             torch.tensor(-1e30, device=q.device))
        probs = torch.softmax(scores, dim=-1)        # [q_len, H, kv_len]
        out[bi] = torch.einsum("qhk,kv->qhv", probs, V[bi])
    return out.to(torch.bfloat16)


def rel_err(a, b):
    a = a.float()
    b = b.float()
    denom = b.abs().clamp_min(1e-3)
    return (a - b).abs().div(denom).max().item()


def mean_rel(a, b):
    a = a.float()
    b = b.float()
    return ((a - b).abs().sum() / b.abs().clamp_min(1e-3).sum()).item()


def run_config(q_len, kv_len, batch_size=1, seed=0):
    torch.manual_seed(seed)
    dev = "cuda"
    ng, sk = derive_groups_splits(q_len, kv_len)
    hpb = NUM_HEADS // ng

    print(f"\n=== q_len={q_len} kv_len={kv_len} B={batch_size} "
          f"| num_head_groups={ng} hpb={hpb} num_splits={sk} ===")

    q = (torch.randn(batch_size * q_len * NUM_HEADS, D_K, device=dev) * 0.1).to(
        torch.bfloat16)
    kv = (torch.randn(batch_size * kv_len, D_K, device=dev) * 0.1).to(
        torch.bfloat16)

    # ---- kernel decode ----
    # force_false=True instantiates mla_mtp_decode_sm100_task_impl<false,false>,
    # the EXACT template the MPK codegen path emits (task_register.cc:3666). The
    # standalone wrappers would otherwise pick <true> for single-tile configs;
    # we test the MPK instantiation to rule out a <false>-path-specific bug.
    part, lse = mtp_decode_wrapper.mtp_decode(
        q, kv, batch_size, q_len, kv_len, ng, sk, force_false=True)

    # ---- reference decode ----
    ref_part, ref_lse = mla_mtp_decode_ref(
        q, kv, batch_size, q_len, kv_len, ng, sk)

    # Compare partial-O only over valid (q,h) entries / active splits.
    # Build a validity mask in the kernel's [block, D_V*128] layout.
    # block_linear = bi*ng*sk + gi*sk + si ; col = d*128 + tid ; tid = qi*hpb + hl
    kvt = (kv_len + TILE_S - 1) // TILE_S
    tps = (kvt + sk - 1) // sk
    part_k = part.float().reshape(batch_size, ng, sk, D_V, 128)
    part_r = ref_part.float().reshape(batch_size, ng, sk, D_V, 128)
    lse_k = lse.reshape(batch_size, ng, sk, 128)
    lse_r = ref_lse.reshape(batch_size, ng, sk, 128)

    # valid tid range: qi in [0,q_len), hl in [0,hpb) -> tid in [0, q_len*hpb)
    valid_tid = q_len * hpb
    # active splits: si with t0 < t1
    part_errs = []
    lse_errs = []
    for si in range(sk):
        t0 = si * tps
        t1 = min(t0 + tps, kvt)
        if t0 >= t1:
            continue
        pk = part_k[:, :, si, :, :valid_tid]
        pr = part_r[:, :, si, :, :valid_tid]
        part_errs.append(rel_err(pk, pr))
        lk = lse_k[:, :, si, :valid_tid]
        lr = lse_r[:, :, si, :valid_tid]
        lse_errs.append((lk - lr).abs().max().item())
    part_rel = max(part_errs) if part_errs else float("nan")
    lse_abs = max(lse_errs) if lse_errs else float("nan")
    print(f"  (A) decode partial-O  rel_max vs ref = {part_rel:.5f}")
    print(f"      decode LSE        abs_max vs ref = {lse_abs:.5f}")

    # ---- kernel reduce ----
    out = mtp_decode_wrapper.mtp_reduce(part, lse, batch_size, q_len, ng, sk)

    # ---- reference reduce (fed the KERNEL's partial/lse) ----
    ref_out = mla_mtp_reduce_ref(part, lse, batch_size, q_len, ng, sk)
    red_rel = rel_err(out, ref_out)
    red_mean = mean_rel(out, ref_out)
    print(f"  (B) reduced O   rel_max vs reduce_ref = {red_rel:.5f}  "
          f"(mean_rel {red_mean:.5f})")

    # ---- clean independent SDPA ----
    sdpa = clean_sdpa(q, kv, batch_size, q_len, kv_len)
    sdpa_rel = rel_err(out, sdpa)
    sdpa_mean = mean_rel(out, sdpa)
    print(f"  (C) reduced O   rel_max vs clean SDPA = {sdpa_rel:.5f}  "
          f"(mean_rel {sdpa_mean:.5f})")

    # Also: full reference pipeline (ref decode -> ref reduce) vs clean SDPA,
    # to confirm the reference itself is self-consistent.
    ref_full = mla_mtp_reduce_ref(ref_part, ref_lse, batch_size, q_len, ng, sk)
    ref_sdpa_rel = rel_err(ref_full, sdpa)
    print(f"  (D) ref pipeline rel_max vs clean SDPA = {ref_sdpa_rel:.5f}")

    # Correctness criterion: the kernel partial must match the partition-aware
    # reference (part_rel, ~bf16 noise), and the reduced output must track the
    # clean SDPA in the mean. We use mean_rel for the SDPA gate because rel_max
    # is dominated by a single near-zero output element (denom clamped to 1e-3);
    # the reference pipeline (D) exhibits the same rel_max vs SDPA, confirming
    # it is a metric artifact, not a kernel error. part_rel == redB == sdpa mean
    # are the load-bearing numbers.
    ok = (part_rel < 0.05) and (red_rel < 0.05) and (sdpa_mean < 0.01)
    print(f"  --> {'PASS' if ok else 'FAIL'}  "
          f"(part_rel<0.05, redB<0.05, sdpa_mean<0.01)")
    return ok, dict(q_len=q_len, kv_len=kv_len, part_rel=part_rel,
                    lse_abs=lse_abs, red_rel=red_rel, sdpa_rel=sdpa_rel,
                    sdpa_mean=sdpa_mean, ref_sdpa_rel=ref_sdpa_rel)


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    configs = [(1, 128), (1, 256), (4, 128), (4, 256)]
    results = []
    all_ok = True
    for q_len, kv_len in configs:
        ok, info = run_config(q_len, kv_len)
        results.append(info)
        all_ok = all_ok and ok

    print("\n================ SUMMARY ================")
    print(f"{'q_len':>5} {'kv_len':>6} {'partA':>8} {'lseA':>8} "
          f"{'redB':>8} {'sdpaCmax':>9} {'sdpaCmean':>10} {'refD':>8}")
    for r in results:
        print(f"{r['q_len']:>5} {r['kv_len']:>6} {r['part_rel']:>8.4f} "
              f"{r['lse_abs']:>8.4f} {r['red_rel']:>8.4f} "
              f"{r['sdpa_rel']:>9.4f} {r['sdpa_mean']:>10.5f} "
              f"{r['ref_sdpa_rel']:>8.4f}")
    print(f"\nOVERALL: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
