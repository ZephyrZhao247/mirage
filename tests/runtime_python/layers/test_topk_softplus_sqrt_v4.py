"""V4-Flash ``topk_softplus_sqrt`` (NEW kernel) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/topk_softplus_sqrt.md``.

Both USE_HASH branches are exercised:
* scored branch (USE_HASH=false): top-k argmax over biased sqrt(softplus)
  scores; bias is subtracted for the WEIGHT output.
* hash branch  (USE_HASH=true):  expert ids come from a precomputed
  tid2eid table indexed by input_ids; weights are
  sqrt(softplus(scores[expert_ids])).

Multi-batch from day 1 (``max_num_batched_requests = 4``).
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.moe import V4TopkSoftplusSqrt


def _run_branch(use_hash: bool):
    device = "cuda"
    torch.manual_seed(0)

    num_tokens = 4
    num_experts = 16
    topk = 4
    routed_scaling_factor = 1.5

    m = V4TopkSoftplusSqrt(
        num_experts=num_experts,
        topk=topk,
        use_hash=use_hash,
        renormalize=True,
        routed_scaling_factor=routed_scaling_factor,
        vocab_size=128 if use_hash else None,
        prefix=f"v4_topk_{'hash' if use_hash else 'scored'}_",
    )

    gating_output = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device
    ) * 0.5

    input_ids = None
    tid2eid = None
    if use_hash:
        input_ids = torch.randint(
            0, 128, (num_tokens,), dtype=torch.int32, device=device
        )
        tid2eid = torch.randint(
            0, num_experts, (128, topk), dtype=torch.int32, device=device
        )
    else:
        m.correction_bias.data = m.correction_bias.data.to(device)
        m.correction_bias.data.copy_(
            torch.randn(num_experts, dtype=torch.float32, device=device) * 0.05
        )

    topk_weights_buf = torch.zeros(
        num_tokens, topk, dtype=torch.float32, device=device
    )
    topk_indices_buf = torch.zeros(
        num_tokens, topk, dtype=torch.int32, device=device
    )
    token_expert_indices_buf = None
    if not use_hash:
        token_expert_indices_buf = torch.zeros(
            num_tokens, topk, dtype=torch.int32, device=device
        )

    ref = m.forward(gating_output, input_ids=input_ids, tid2eid=tid2eid)
    ref_w, ref_i, ref_tei = ref

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

    gating_dt = pk.attach_input(gating_output, name=f"gating_{use_hash}")
    if use_hash:
        input_ids_dt = pk.attach_input(input_ids, name=f"input_ids_{use_hash}")

    with pk.compile_scope():
        if use_hash:
            _ = m.compile(
                gating_dt,
                input_ids=input_ids_dt,
                tid2eid=tid2eid,
                topk_weights=topk_weights_buf,
                topk_indices=topk_indices_buf,
            )
        else:
            _ = m.compile(
                gating_dt,
                topk_weights=topk_weights_buf,
                topk_indices=topk_indices_buf,
                token_expert_indices=token_expert_indices_buf,
            )

    print(f"Compiling V4TopkSoftplusSqrt test kernel ({'hash' if use_hash else 'scored'})...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print(f"Running V4TopkSoftplusSqrt test kernel ({'hash' if use_hash else 'scored'})...")
    pk()
    torch.cuda.synchronize()

    print(f"topk_weights_buf[0] = {topk_weights_buf[0]}")
    print(f"ref_w           [0] = {ref_w[0]}")
    print(f"topk_indices_buf[0] = {topk_indices_buf[0]}")
    print(f"ref_i           [0] = {ref_i[0]}")
    w_diff = (topk_weights_buf - ref_w).abs().max().item()
    i_diff = (topk_indices_buf.to(torch.int64) - ref_i.to(torch.int64)).abs().max().item()
    print(f"weights max-abs diff: {w_diff}")
    print(f"indices max-abs diff: {i_diff}")

    try:
        torch.testing.assert_close(topk_weights_buf, ref_w, atol=1e-3, rtol=1e-3)
        # Allow set-equality on indices: different argmax orderings can swap
        # equal-weight winners. We sort by index for a deterministic compare
        # (the kernel writes in selection order; PyTorch also writes in order;
        # both use the same tie-break so this is typically exact).
        assert (topk_indices_buf.to(torch.int64) == ref_i.to(torch.int64)).all(), (
            "indices mismatch"
        )
        print(f"PASSED: V4TopkSoftplusSqrt ({'hash' if use_hash else 'scored'})")
    except AssertionError as e:
        print(
            f"FAILED: V4TopkSoftplusSqrt ({'hash' if use_hash else 'scored'})\n{e}"
        )
        pk.finalize()
        sys.exit(1)

    pk.finalize()


def test_topk_softplus_sqrt_v4_testmode():
    _run_branch(use_hash=False)
    _run_branch(use_hash=True)
    print("Test completed successfully!")


if __name__ == "__main__":
    test_topk_softplus_sqrt_v4_testmode()
