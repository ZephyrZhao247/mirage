"""
Tests for DAG (fan-out / fan-in) support in MPK persistent kernel.

Tests the register_mugraph DAG dependency analysis with:
- 1 to 4-way fan-out (one layer's output feeds N downstream layers)
- 1 to 4-way fan-in (N independent paths merge via cascading linear_with_residual)
- Complete DAG patterns combining fork and join (diamond, wide diamond, nested)

All tests use test_mode: compile → run once → verify against PyTorch reference.
"""

import torch
import sys
import os

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel


# ---------------------------------------------------------------------------
# Reference helpers
# ---------------------------------------------------------------------------

def torch_rmsnorm(x, weight, eps=1e-5):
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return (x_normed * weight).to(x.dtype)


def torch_linear_with_residual(x, weight, residual):
    # Use bf16 matmul to match the GPU kernel's rounding behavior.
    # Using float32 reference causes exponential error amplification
    # in cascading matmuls that obscures DAG correctness testing.
    return (x.to(torch.bfloat16) @ weight.T.to(torch.bfloat16)).to(
        torch.float32
    ) + residual.to(torch.float32)


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

DEVICE = "cuda"
DTYPE = torch.bfloat16
BATCH = 2
HIDDEN = 4096
TOLERANCE_ELEMENTWISE = 0.1   # bf16 tolerance for elementwise ops (rmsnorm)
TOLERANCE_MATMUL = 2.0        # bf16 tolerance for matmul (residual rounding)


def make_pk():
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    return PersistentKernel(**params)


def get_block_dim(pk):
    return (256, 1, 1) if pk.target_cc >= 90 else (128, 1, 1)


def get_linear_grid_dim(pk, out_dim):
    """Compute grid_dim for linear layers based on target compute capability."""
    if pk.target_cc >= 100:
        tile = 64
    elif pk.target_cc >= 90:
        tile = 64
    else:
        tile = 64
    return (out_dim // tile, 1, 1)


def check(name, actual, expected, tol=TOLERANCE_ELEMENTWISE):
    diff = (actual.float() - expected.float()).abs().max().item()
    if diff < tol:
        print(f"  PASSED {name} (max_diff={diff:.6f})")
        return True
    else:
        print(f"  FAILED {name} (max_diff={diff:.6f} > tol={tol})")
        return False


# ===================================================================
# Fan-out tests: one layer's output feeds N downstream layers
# ===================================================================

def test_1way_chain():
    """Baseline: input -> rmsnorm -> output (no fan-out)."""
    print("\n=== test_1way_chain ===")
    pk = make_pk()
    bd = get_block_dim(pk)

    x = torch.randn(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)
    w = torch.randn(HIDDEN, dtype=DTYPE, device=DEVICE)
    out = torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)

    x_dt = pk.attach_input(x, name="x")
    w_dt = pk.attach_input(w, name="w")
    out_dt = pk.attach_input(out, name="out")

    pk.rmsnorm_layer(
        input=x_dt, weight=w_dt, output=out_dt,
        grid_dim=(BATCH, 1, 1), block_dim=bd,
    )

    pk.compile(output_dir=os.path.dirname(__file__))
    pk.run_test_mode()
    torch.cuda.synchronize()

    ref = torch_rmsnorm(x, w)
    ok = check("rmsnorm", out, ref)
    pk.finalize()
    return ok


def _test_n_way_fanout(n):
    """input -> rmsnorm_root -> {rmsnorm_1, ..., rmsnorm_n}"""
    print(f"\n=== test_{n}way_fanout ===")
    pk = make_pk()
    bd = get_block_dim(pk)

    x = torch.randn(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)
    w_root = torch.randn(HIDDEN, dtype=DTYPE, device=DEVICE)
    root_buf = torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)

    x_dt = pk.attach_input(x, name="x")
    w_root_dt = pk.attach_input(w_root, name="w_root")
    root_dt = pk.attach_input(root_buf, name="root_out")

    # Root layer: produces root_out
    pk.rmsnorm_layer(
        input=x_dt, weight=w_root_dt, output=root_dt,
        grid_dim=(BATCH, 1, 1), block_dim=bd,
    )

    # Fan-out: N downstream layers all read root_out
    weights = []
    outputs = []
    out_bufs = []
    for i in range(n):
        w_i = torch.randn(HIDDEN, dtype=DTYPE, device=DEVICE)
        o_i = torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)
        weights.append(w_i)
        out_bufs.append(o_i)
        w_dt_i = pk.attach_input(w_i, name=f"w_{i}")
        o_dt_i = pk.attach_input(o_i, name=f"out_{i}")
        outputs.append(o_dt_i)

        pk.rmsnorm_layer(
            input=root_dt, weight=w_dt_i, output=o_dt_i,
            grid_dim=(BATCH, 1, 1), block_dim=bd,
        )

    pk.compile(output_dir=os.path.dirname(__file__))
    pk.run_test_mode()
    torch.cuda.synchronize()

    # Reference
    root_ref = torch_rmsnorm(x, w_root)
    all_ok = True
    for i in range(n):
        ref_i = torch_rmsnorm(root_ref, weights[i])
        if not check(f"branch_{i}", out_bufs[i], ref_i):
            all_ok = False

    pk.finalize()
    return all_ok


def test_2way_fanout():
    return _test_n_way_fanout(2)


def test_3way_fanout():
    return _test_n_way_fanout(3)


def test_4way_fanout():
    return _test_n_way_fanout(4)


# ===================================================================
# Fan-in tests: N independent paths merge via cascading
# linear_with_residual layers
# ===================================================================

def _test_n_way_fanin(n):
    """
    N independent rmsnorm paths → cascading merge via linear_with_residual.

    Graph (example n=3):
        x_0 → rmsnorm_0 → a0 ─┐
        x_1 → rmsnorm_1 → a1 ─┤── merge_0(a0, W0, a1) → m0 ─┐
        x_2 → rmsnorm_2 → a2 ──────────────────────────────────┤── merge_1(m0, W1, a2) → output
    """
    print(f"\n=== test_{n}way_fanin ===")
    pk = make_pk()
    bd = get_block_dim(pk)
    lgd = get_linear_grid_dim(pk, HIDDEN)

    # Create N independent inputs, weights, and output buffers
    xs = [torch.randn(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE) for _ in range(n)]
    ws = [torch.randn(HIDDEN, dtype=DTYPE, device=DEVICE) for _ in range(n)]
    path_bufs = [
        torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE) for _ in range(n)
    ]

    x_dts = [pk.attach_input(xs[i], name=f"x_{i}") for i in range(n)]
    w_dts = [pk.attach_input(ws[i], name=f"w_{i}") for i in range(n)]
    path_dts = [pk.attach_input(path_bufs[i], name=f"path_{i}") for i in range(n)]

    # N independent rmsnorm paths
    for i in range(n):
        pk.rmsnorm_layer(
            input=x_dts[i], weight=w_dts[i], output=path_dts[i],
            grid_dim=(BATCH, 1, 1), block_dim=bd,
        )

    if n == 1:
        # No merge needed — single path is the output
        final_buf = path_bufs[0]
    else:
        # Cascading merges: merge path_0 + path_1, then + path_2, etc.
        merge_weights = []
        merge_bufs = []
        prev_dt = path_dts[0]
        for i in range(1, n):
            # Scale weights by 1/sqrt(K) to prevent magnitude explosion
            # across cascading matmuls (Xavier initialization)
            mw = torch.randn(HIDDEN, HIDDEN, dtype=DTYPE, device=DEVICE) * (
                HIDDEN ** -0.5
            )
            mb = torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)
            merge_weights.append(mw)
            merge_bufs.append(mb)
            mw_dt = pk.attach_input(mw, name=f"merge_w_{i}")
            mb_dt = pk.attach_input(mb, name=f"merge_out_{i}")

            pk.linear_with_residual_layer(
                input=prev_dt,
                weight=mw_dt,
                residual=path_dts[i],
                output=mb_dt,
                grid_dim=lgd,
                block_dim=bd,
            )
            prev_dt = mb_dt
        final_buf = merge_bufs[-1]

    pk.compile(output_dir=os.path.dirname(__file__))
    pk.run_test_mode()
    torch.cuda.synchronize()

    # Reference: compute each path, then cascade merge
    path_refs = [torch_rmsnorm(xs[i], ws[i]) for i in range(n)]
    if n == 1:
        ref = path_refs[0]
    else:
        ref = path_refs[0]
        for i in range(1, n):
            ref = torch_linear_with_residual(ref, merge_weights[i - 1], path_refs[i])
        ref = ref.to(DTYPE)

    tol = TOLERANCE_MATMUL * max(n - 1, 1)  # cascading matmuls accumulate error
    ok = check("final_output", final_buf, ref, tol=tol if n > 1 else TOLERANCE_ELEMENTWISE)
    pk.finalize()
    return ok


def test_1way_fanin():
    return _test_n_way_fanin(1)


def test_2way_fanin():
    return _test_n_way_fanin(2)


def test_3way_fanin():
    return _test_n_way_fanin(3)


def test_4way_fanin():
    return _test_n_way_fanin(4)


# ===================================================================
# Complete DAG tests: fan-out + fan-in combined
# ===================================================================

def test_diamond():
    """
    Diamond: 2-way fork + 2-way join.

        input → rmsnorm_root → root_out
                               /       \\
                       rmsnorm_A      rmsnorm_B
                          a_out        b_out
                               \\       /
                        linear_with_residual → output

    Tests: 2-way fan-out at root, 2-way fan-in at merge.
    """
    print("\n=== test_diamond ===")
    pk = make_pk()
    bd = get_block_dim(pk)
    lgd = get_linear_grid_dim(pk, HIDDEN)

    x = torch.randn(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)
    w_root = torch.randn(HIDDEN, dtype=DTYPE, device=DEVICE)
    root_buf = torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)
    w_a = torch.randn(HIDDEN, dtype=DTYPE, device=DEVICE)
    w_b = torch.randn(HIDDEN, dtype=DTYPE, device=DEVICE)
    a_buf = torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)
    b_buf = torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)
    w_merge = torch.randn(HIDDEN, HIDDEN, dtype=DTYPE, device=DEVICE) * (HIDDEN ** -0.5)
    out_buf = torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)

    x_dt = pk.attach_input(x, name="x")
    w_root_dt = pk.attach_input(w_root, name="w_root")
    root_dt = pk.attach_input(root_buf, name="root_out")
    w_a_dt = pk.attach_input(w_a, name="w_a")
    w_b_dt = pk.attach_input(w_b, name="w_b")
    a_dt = pk.attach_input(a_buf, name="a_out")
    b_dt = pk.attach_input(b_buf, name="b_out")
    w_merge_dt = pk.attach_input(w_merge, name="w_merge")
    out_dt = pk.attach_input(out_buf, name="out")

    # Root
    pk.rmsnorm_layer(
        input=x_dt, weight=w_root_dt, output=root_dt,
        grid_dim=(BATCH, 1, 1), block_dim=bd,
    )
    # Fork: two branches read root_out
    pk.rmsnorm_layer(
        input=root_dt, weight=w_a_dt, output=a_dt,
        grid_dim=(BATCH, 1, 1), block_dim=bd,
    )
    pk.rmsnorm_layer(
        input=root_dt, weight=w_b_dt, output=b_dt,
        grid_dim=(BATCH, 1, 1), block_dim=bd,
    )
    # Join: merge a_out and b_out
    pk.linear_with_residual_layer(
        input=a_dt, weight=w_merge_dt, residual=b_dt, output=out_dt,
        grid_dim=lgd, block_dim=bd,
    )

    pk.compile(output_dir=os.path.dirname(__file__))
    pk.run_test_mode()
    torch.cuda.synchronize()

    # Reference
    root_ref = torch_rmsnorm(x, w_root)
    a_ref = torch_rmsnorm(root_ref, w_a)
    b_ref = torch_rmsnorm(root_ref, w_b)
    out_ref = torch_linear_with_residual(a_ref, w_merge, b_ref).to(DTYPE)

    ok = check("diamond_output", out_buf, out_ref, tol=TOLERANCE_MATMUL)
    pk.finalize()
    return ok


def test_wide_diamond():
    """
    4-way fork at root + cascading 2-way joins.

        input → rmsnorm_root → root_out
                           / | | \\
                          A  B  C  D
                           \\ | | /
                        merge cascade → output

    Tests: 4-way fan-out + cascading fan-in.
    """
    print("\n=== test_wide_diamond ===")
    pk = make_pk()
    bd = get_block_dim(pk)
    lgd = get_linear_grid_dim(pk, HIDDEN)
    N_BRANCHES = 4

    x = torch.randn(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)
    w_root = torch.randn(HIDDEN, dtype=DTYPE, device=DEVICE)
    root_buf = torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)

    x_dt = pk.attach_input(x, name="x")
    w_root_dt = pk.attach_input(w_root, name="w_root")
    root_dt = pk.attach_input(root_buf, name="root_out")

    pk.rmsnorm_layer(
        input=x_dt, weight=w_root_dt, output=root_dt,
        grid_dim=(BATCH, 1, 1), block_dim=bd,
    )

    # 4-way fan-out
    branch_weights = []
    branch_bufs = []
    branch_dts = []
    for i in range(N_BRANCHES):
        w_i = torch.randn(HIDDEN, dtype=DTYPE, device=DEVICE)
        b_i = torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)
        branch_weights.append(w_i)
        branch_bufs.append(b_i)
        w_dt_i = pk.attach_input(w_i, name=f"w_br_{i}")
        b_dt_i = pk.attach_input(b_i, name=f"br_{i}")
        branch_dts.append(b_dt_i)

        pk.rmsnorm_layer(
            input=root_dt, weight=w_dt_i, output=b_dt_i,
            grid_dim=(BATCH, 1, 1), block_dim=bd,
        )

    # Cascading merge
    merge_weights = []
    merge_bufs = []
    prev_dt = branch_dts[0]
    for i in range(1, N_BRANCHES):
        mw = torch.randn(HIDDEN, HIDDEN, dtype=DTYPE, device=DEVICE) * (HIDDEN ** -0.5)
        mb = torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)
        merge_weights.append(mw)
        merge_bufs.append(mb)
        mw_dt = pk.attach_input(mw, name=f"mw_{i}")
        mb_dt = pk.attach_input(mb, name=f"m_{i}")

        pk.linear_with_residual_layer(
            input=prev_dt, weight=mw_dt, residual=branch_dts[i], output=mb_dt,
            grid_dim=lgd, block_dim=bd,
        )
        prev_dt = mb_dt

    pk.compile(output_dir=os.path.dirname(__file__))
    pk.run_test_mode()
    torch.cuda.synchronize()

    # Reference
    root_ref = torch_rmsnorm(x, w_root)
    branch_refs = [torch_rmsnorm(root_ref, branch_weights[i]) for i in range(N_BRANCHES)]
    ref = branch_refs[0]
    for i in range(1, N_BRANCHES):
        ref = torch_linear_with_residual(ref, merge_weights[i - 1], branch_refs[i])
    ref = ref.to(DTYPE)

    ok = check("wide_diamond_output", merge_bufs[-1], ref,
               tol=TOLERANCE_MATMUL * (N_BRANCHES - 1))
    pk.finalize()
    return ok


def test_nested_dag():
    """
    Nested fork-join: two diamonds stacked.

        input → rmsnorm_root → root_out
                              /       \\
                      rmsnorm_A      rmsnorm_B
                         a_out        b_out
                              \\       /
                        merge_1 (linear_w_res) → m1_out
                              /       \\
                      rmsnorm_C      rmsnorm_D
                         c_out        d_out
                              \\       /
                        merge_2 (linear_w_res) → output

    Tests: nested fan-out / fan-in at two levels.
    """
    print("\n=== test_nested_dag ===")
    pk = make_pk()
    bd = get_block_dim(pk)
    lgd = get_linear_grid_dim(pk, HIDDEN)

    x = torch.randn(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)
    w_root = torch.randn(HIDDEN, dtype=DTYPE, device=DEVICE)
    root_buf = torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)
    w_a = torch.randn(HIDDEN, dtype=DTYPE, device=DEVICE)
    w_b = torch.randn(HIDDEN, dtype=DTYPE, device=DEVICE)
    a_buf = torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)
    b_buf = torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)
    w_merge1 = torch.randn(HIDDEN, HIDDEN, dtype=DTYPE, device=DEVICE) * (HIDDEN ** -0.5)
    m1_buf = torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)
    w_c = torch.randn(HIDDEN, dtype=DTYPE, device=DEVICE)
    w_d = torch.randn(HIDDEN, dtype=DTYPE, device=DEVICE)
    c_buf = torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)
    d_buf = torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)
    w_merge2 = torch.randn(HIDDEN, HIDDEN, dtype=DTYPE, device=DEVICE) * (HIDDEN ** -0.5)
    out_buf = torch.zeros(BATCH, HIDDEN, dtype=DTYPE, device=DEVICE)

    x_dt = pk.attach_input(x, name="x")
    w_root_dt = pk.attach_input(w_root, name="w_root")
    root_dt = pk.attach_input(root_buf, name="root_out")
    w_a_dt = pk.attach_input(w_a, name="w_a")
    w_b_dt = pk.attach_input(w_b, name="w_b")
    a_dt = pk.attach_input(a_buf, name="a_out")
    b_dt = pk.attach_input(b_buf, name="b_out")
    w_m1_dt = pk.attach_input(w_merge1, name="w_merge1")
    m1_dt = pk.attach_input(m1_buf, name="m1_out")
    w_c_dt = pk.attach_input(w_c, name="w_c")
    w_d_dt = pk.attach_input(w_d, name="w_d")
    c_dt = pk.attach_input(c_buf, name="c_out")
    d_dt = pk.attach_input(d_buf, name="d_out")
    w_m2_dt = pk.attach_input(w_merge2, name="w_merge2")
    out_dt = pk.attach_input(out_buf, name="out")

    # Level 1: root → fork
    pk.rmsnorm_layer(
        input=x_dt, weight=w_root_dt, output=root_dt,
        grid_dim=(BATCH, 1, 1), block_dim=bd,
    )
    pk.rmsnorm_layer(
        input=root_dt, weight=w_a_dt, output=a_dt,
        grid_dim=(BATCH, 1, 1), block_dim=bd,
    )
    pk.rmsnorm_layer(
        input=root_dt, weight=w_b_dt, output=b_dt,
        grid_dim=(BATCH, 1, 1), block_dim=bd,
    )
    # Level 1: join
    pk.linear_with_residual_layer(
        input=a_dt, weight=w_m1_dt, residual=b_dt, output=m1_dt,
        grid_dim=lgd, block_dim=bd,
    )

    # Level 2: fork from merge_1 output
    pk.rmsnorm_layer(
        input=m1_dt, weight=w_c_dt, output=c_dt,
        grid_dim=(BATCH, 1, 1), block_dim=bd,
    )
    pk.rmsnorm_layer(
        input=m1_dt, weight=w_d_dt, output=d_dt,
        grid_dim=(BATCH, 1, 1), block_dim=bd,
    )
    # Level 2: join
    pk.linear_with_residual_layer(
        input=c_dt, weight=w_m2_dt, residual=d_dt, output=out_dt,
        grid_dim=lgd, block_dim=bd,
    )

    pk.compile(output_dir=os.path.dirname(__file__))
    pk.run_test_mode()
    torch.cuda.synchronize()

    # Reference
    root_ref = torch_rmsnorm(x, w_root)
    a_ref = torch_rmsnorm(root_ref, w_a)
    b_ref = torch_rmsnorm(root_ref, w_b)
    m1_ref = torch_linear_with_residual(a_ref, w_merge1, b_ref)
    c_ref = torch_rmsnorm(m1_ref.to(DTYPE), w_c)
    d_ref = torch_rmsnorm(m1_ref.to(DTYPE), w_d)
    out_ref = torch_linear_with_residual(c_ref, w_merge2, d_ref).to(DTYPE)

    ok = check("nested_dag_output", out_buf, out_ref,
               tol=TOLERANCE_MATMUL * 2)  # two cascading matmuls
    pk.finalize()
    return ok


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":
    results = {}

    print("=" * 60)
    print("MPK DAG Fan-Out / Fan-In Tests")
    print("=" * 60)

    # Fan-out tests
    results["1way_chain"] = test_1way_chain()
    results["2way_fanout"] = test_2way_fanout()
    results["3way_fanout"] = test_3way_fanout()
    results["4way_fanout"] = test_4way_fanout()

    # Fan-in tests
    results["1way_fanin"] = test_1way_fanin()
    results["2way_fanin"] = test_2way_fanin()
    results["3way_fanin"] = test_3way_fanin()
    results["4way_fanin"] = test_4way_fanin()

    # Complete DAG tests
    results["diamond"] = test_diamond()
    results["wide_diamond"] = test_wide_diamond()
    results["nested_dag"] = test_nested_dag()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\nAll tests passed!")
    else:
        print("\nSome tests FAILED!")
        sys.exit(1)
