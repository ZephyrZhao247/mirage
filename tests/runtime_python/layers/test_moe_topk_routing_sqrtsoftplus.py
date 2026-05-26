"""Numerical tests for ``MoETopkSigmoidRouting.score_func`` variants.

Three scoring functions are exercised through the full MPK compilation
pipeline (test_mode):

* ``"sigmoid"``    — DeepSeek V3 regression (current behavior; this is
                     equivalent to the existing ``test_moe_routing.py``).
* ``"sqrtsoftplus"`` — DeepSeek V4-Flash:
                     ``score = sqrt(softplus(logit)) = sqrt(log(1+exp(logit)))``.
* ``"softmax"``    — sanity test for the third enum value.

We compare the kernel ``moe_topk_weights`` to the reference produced by
``MoETopkSigmoidRouting.forward(...)`` after sorting each row (the kernel
may pick tied experts in a different order). Routing-index sanity is also
checked (each token has exactly K non-zero slots numbered 1..K).
"""

import os

import torch

import mirage
from mirage.mpk.layers.moe.routing import MoETopkSigmoidRouting
from mirage.mpk.persistent_kernel import PersistentKernel


def _check_routing_sanity(routing_indices: torch.Tensor, batch_size: int, topk: int):
    nz_per_token = (routing_indices != 0).sum(dim=0)
    assert torch.all(nz_per_token == topk), (
        f"each token must be routed to {topk} experts; "
        f"got per-token nz counts {nz_per_token.tolist()}"
    )
    for t in range(batch_size):
        col = routing_indices[:, t]
        slots = col[col != 0].sort().values
        expected = torch.arange(1, topk + 1, dtype=col.dtype, device=col.device)
        assert torch.equal(slots, expected), (
            f"token {t} routing slots are not a permutation of "
            f"1..{topk}: {slots.tolist()}"
        )


def _topk_weights_close(out_w: torch.Tensor, ref_w: torch.Tensor, *, atol=2e-2, rtol=2e-2):
    out_sorted, _ = torch.sort(out_w, dim=-1)
    ref_sorted, _ = torch.sort(ref_w, dim=-1)
    torch.testing.assert_close(out_sorted, ref_sorted, atol=atol, rtol=rtol)


def _run_routing_test(
    *,
    score_func: str,
    num_experts: int,
    topk: int,
    num_groups: int,
    topk_group: int,
    routed_scaling_factor: float,
    batch_size: int,
    seed: int,
    tag: str,
):
    device = "cuda"
    torch.manual_seed(seed)

    print(f"\n=== {tag} ===")
    logits_bf16 = torch.randn(
        batch_size, num_experts, dtype=torch.bfloat16, device=device
    )
    bias = torch.randn(num_experts, dtype=torch.float32, device=device) * 0.1
    topk_weights = torch.zeros(batch_size, topk, dtype=torch.float32, device=device)
    routing_indices = torch.zeros(num_experts, batch_size, dtype=torch.int32, device=device)
    moe_mask = torch.zeros(num_experts + 1, dtype=torch.int32, device=device)

    m = MoETopkSigmoidRouting(
        num_experts=num_experts,
        num_experts_per_tok=topk,
        num_groups=num_groups,
        topk_group=topk_group,
        routed_scaling_factor=routed_scaling_factor,
        score_func=score_func,
    ).to(device=device)
    with torch.no_grad():
        m.bias.copy_(bias)

    ref_weights, _ref_routing, _ = m.forward(logits_bf16.float())

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = batch_size
    params["max_num_batched_requests"] = batch_size
    pk = PersistentKernel(**params)

    logits_dt = pk.attach_input(logits_bf16, name=f"{tag}_logits")
    topk_w_dt = pk.attach_input(topk_weights, name=f"{tag}_topk_w")
    routing_dt = pk.attach_input(routing_indices, name=f"{tag}_indices")
    mask_dt = pk.attach_input(moe_mask, name=f"{tag}_mask")

    with pk.compile_scope():
        _ = m.compile(logits_dt, topk_w_dt, routing_dt, mask_dt)

    print(f"Compiling {tag} routing test kernel...")
    pk.compile(output_dir=os.path.dirname(__file__))
    print(f"Running {tag} test kernel...")
    pk()
    torch.cuda.synchronize()

    print(f"topk_weights[0]: {topk_weights[0]}")
    print(f"ref_weights[0]:  {ref_weights[0]}")

    _check_routing_sanity(routing_indices, batch_size, topk)
    _topk_weights_close(topk_weights, ref_weights)
    print(f"PASSED: MoETopkSigmoidRouting(score_func={score_func!r})")
    pk.finalize()


def test_score_func_sigmoid_regression():
    """Regression: existing V3 group-limited sigmoid path unchanged."""
    _run_routing_test(
        score_func="sigmoid",
        num_experts=128,
        topk=4,
        num_groups=4,
        topk_group=2,
        routed_scaling_factor=2.5,
        batch_size=2,
        seed=1,
        tag="sigmoid",
    )


def test_score_func_sqrtsoftplus():
    """V4-Flash flat-topK with sqrt(softplus(x)) scoring."""
    _run_routing_test(
        score_func="sqrtsoftplus",
        num_experts=128,
        # flat top-K: num_groups=1, topk_group=1
        topk=4,
        num_groups=1,
        topk_group=1,
        routed_scaling_factor=1.5,
        batch_size=2,
        seed=2,
        tag="sqrtsoftplus",
    )


def test_score_func_softmax():
    """Sanity: third enum value (softmax per-row) compiles + runs."""
    _run_routing_test(
        score_func="softmax",
        num_experts=128,
        topk=4,
        num_groups=1,
        topk_group=1,
        routed_scaling_factor=1.0,
        batch_size=2,
        seed=3,
        tag="softmax",
    )


if __name__ == "__main__":
    test_score_func_sigmoid_regression()
    test_score_func_sqrtsoftplus()
    test_score_func_softmax()
    print("All score_func variant tests completed successfully!")
