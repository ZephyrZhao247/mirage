"""Test: DeepseekV3MoE (FP8 permute path) compile() vs HF-faithful forward().

Full MoE block: router GEMM (bf16) -> MoETopkRouting(sigmoid) -> permute
chain (validated by test_dsv3_moe_permute_chain_testmode.py) -> shared expert.
Compares the MPK compile() output against the module's forward() (HF gate +
per-expert routed loop + shared(x) + residual).

Run:
  CUDA_VISIBLE_DEVICES=<free> python tests/runtime_python/test_mode/test_dsv3_moe_testmode.py
"""

import os
import sys
from types import SimpleNamespace

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.models.deepseek_v3.modeling import DeepseekV3MoE
from test_dsv3_moe_permute_chain_testmode import _pack_group_weight
from test_dsv3_mlp_fp8_testmode import _qfp8  # 2D 128x128 block quantizer


def test_dsv3_moe():
    device = "cuda"
    torch.manual_seed(0)
    mbt = 4
    # H/I overridable via env to exercise the production hidden size (7168),
    # which triggers K_PACKED>=4 in the permute input-scale read path.
    H = int(os.environ.get("MOE_H", "256"))
    I = int(os.environ.get("MOE_I", "256"))  # moe_intermediate_size
    E = 16                 # n_routed_experts (EXPERTS_PER_GROUP = E/n_group must be a multiple of VPT=8)
    cfg = SimpleNamespace(
        hidden_size=H, moe_intermediate_size=I, n_routed_experts=E,
        num_experts_per_tok=2, n_shared_experts=1, n_group=2, topk_group=1,
        routed_scaling_factor=2.5, norm_topk_prob=True,
        intermediate_size=I,  # unused here (dense path)
    )

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(test_mode=True, num_workers=num_workers,
                  num_local_schedulers=num_schedulers, mpi_rank=0, world_size=1,
                  max_num_batched_tokens=mbt, max_num_batched_requests=mbt)
    pk = PersistentKernel(**params)

    x = torch.randn(mbt, H, dtype=torch.bfloat16, device=device) * 0.5
    residual = torch.randn(mbt, H, dtype=torch.bfloat16, device=device) * 0.1
    out = torch.zeros(mbt, H, dtype=torch.bfloat16, device=device)

    # Routed expert weights (separated logits via a structured gate to keep
    # bf16 vs fp32 top-k selection stable).
    w13 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=device) * 0.05
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device=device) * 0.05
    w13_fp8, w13_scale, _ = _pack_group_weight(w13)
    w2_fp8, w2_scale, _ = _pack_group_weight(w2)
    # Shared expert (gate_up concat + down), 128x128-block quantized.
    sg = torch.randn(I, H, dtype=torch.bfloat16, device=device) * 0.05
    su = torch.randn(I, H, dtype=torch.bfloat16, device=device) * 0.05
    sd = torch.randn(H, I, dtype=torch.bfloat16, device=device) * 0.05
    sgu_fp8, sgu_scale = _qfp8(torch.cat([sg, su], dim=0))
    sd_fp8, sd_scale = _qfp8(sd)
    gate_w = torch.randn(E, H, dtype=torch.bfloat16, device=device) * 0.2

    with pk.compile_scope():
        m = DeepseekV3MoE(cfg, prefix="moe_").to(device)
        with torch.no_grad():
            m.gate_weight.copy_(gate_w)
            m.routing.bias.zero_()
            m.experts_w13.weight.copy_(w13_fp8)
            m.experts_w13.weight_scale.copy_(w13_scale)
            m.experts_w2.weight.copy_(w2_fp8)
            m.experts_w2.weight_scale.copy_(w2_scale)
            m.shared_experts.gate_up_weight.copy_(sgu_fp8)
            m.shared_experts.gate_up_scale.copy_(sgu_scale)
            m.shared_experts.down_weight.copy_(sd_fp8)
            m.shared_experts.down_scale.copy_(sd_scale)

        ref = m.forward(x, residual)

        x_dt = pk.attach_input(x, name="x")
        res_dt = pk.attach_input(residual, name="residual")
        out_dt = pk.attach_input(out, name="out")
        m.compile(x_dt, res_dt, output=out_dt)

    print("Compiling DeepseekV3MoE (FP8 permute path)...")
    pk.compile(output_dir=_HERE)
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
    ok = max_rel < 0.15
    print("PASSED" if ok else "FAILED")
    pk.finalize()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    test_dsv3_moe()
