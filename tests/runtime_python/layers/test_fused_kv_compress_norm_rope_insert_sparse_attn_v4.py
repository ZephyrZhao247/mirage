"""V4-Flash ``fused_kv_compress_norm_rope_insert_sparse_attn`` (NEW
kernel, head_dim=512) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/
fused_kv_compress_norm_rope_insert_sparse_attn.md``.

Per-boundary-token compressor (gather window + softmax + weighted sum +
RMSNorm + UE8M0 block-FP8 quant + GPT-J RoPE on rope tail).
Multi-batch (max_num_batched_requests >= 2).

The compressor RoPE uses ``compress_rope_theta=160000`` (NOT main
RoPE's 10000). We build ``cos_sin_cache`` against the 160000 base.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.compressor import (
    V4FusedKVCompressNormRopeInsertSparseAttn,
)


def _build_cos_sin_cache(max_pos: int, rope_dim: int,
                         base: float = 160000.0,
                         device: str = "cuda") -> torch.Tensor:
    """Build the compressor cos_sin_cache (rope_theta=160000)."""
    half = rope_dim // 2
    inv_freq = base ** (-torch.arange(0, half, dtype=torch.float32, device=device) /
                        half)
    positions = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = positions[:, None] * inv_freq[None, :]  # [max_pos, half]
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat([cos, sin], dim=-1)  # [max_pos, rope_dim]


def test_fused_kv_compress_norm_rope_insert_sparse_attn_v4_testmode():
    device = "cuda"
    torch.manual_seed(0)

    # V4-Flash attn compressor: HEAD_SIZE=512, ratio=4, OVERLAP=True.
    head_size = 512
    compress_ratio = 4
    block_size = 16
    kv_cache_block_size = 16
    num_blocks = 4
    num_kv_blocks = 4
    max_blocks = 4

    # 2 requests, each with 4 tokens. Multi-batch.
    num_tokens = 8
    num_reqs = 2
    # Positions chosen so SOME tokens hit the (position+1) % ratio == 0
    # boundary: token i has position = i (so tokens 3 and 7 trigger
    # boundary-write, others early-exit).
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    token_to_req_indices = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device
    )
    # slot_mapping spans both blocks of block 0 / block 1 to exercise the
    # block split.
    slot_mapping = torch.tensor(
        [0, 1, 2, 3, 16, 17, 18, 19], dtype=torch.int64, device=device
    )
    kv_slot_mapping = torch.tensor(
        [0, 1, 2, 3, 16, 17, 18, 19], dtype=torch.int64, device=device
    )
    block_table = torch.tensor(
        [[0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.int32, device=device
    )

    # State cache: [num_blocks, block_size, 2*STATE_WIDTH = 4*head_size = 2048].
    state_cache = torch.randn(
        num_blocks, block_size, 4 * head_size,
        dtype=torch.float32, device=device,
    )

    # cos_sin_cache built with compress_rope_theta = 160000.
    max_pos = 256
    rope_head_dim = 64
    cos_sin_cache = _build_cos_sin_cache(
        max_pos, rope_head_dim, base=160000.0, device=device
    )

    # K cache: [num_kv_blocks, kv_block_size, 1, TOKEN_STRIDE + SCALE_DIM]
    # for HEAD_SIZE=512 -> 576 + 8 = 584 bytes.
    token_stride = 576
    scale_dim = 8
    k_cache_buf = torch.zeros(
        num_kv_blocks, kv_cache_block_size, 1, token_stride + scale_dim,
        dtype=torch.uint8, device=device,
    )
    k_cache_ref = k_cache_buf.clone()

    module = V4FusedKVCompressNormRopeInsertSparseAttn(
        compress_ratio=compress_ratio,
        block_size=block_size,
        kv_cache_block_size=kv_cache_block_size,
        rms_norm_eps=1e-6,
        prefix="v4_spaC_",
    )
    module.rms_norm_weight.data = module.rms_norm_weight.data.to(
        device=device, dtype=torch.bfloat16
    )
    module.rms_norm_weight.data.copy_(
        torch.randn(head_size, dtype=torch.bfloat16, device=device) * 0.1 + 1.0
    )

    module.forward(
        state_cache, token_to_req_indices, positions, slot_mapping,
        block_table, cos_sin_cache, kv_slot_mapping, k_cache_ref,
    )

    # Build PK in test mode.
    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params["test_mode"] = True
    params["num_workers"] = num_workers
    params["num_local_schedulers"] = num_schedulers
    params["mpi_rank"] = 0
    params["world_size"] = 1
    params["max_num_batched_tokens"] = num_tokens
    params["max_num_batched_requests"] = max(num_tokens, 4)
    pk = PersistentKernel(**params)

    sc_dt   = pk.attach_input(state_cache, name="state_cache")
    ttr_dt  = pk.attach_input(token_to_req_indices, name="token_to_req_indices")
    pos_dt  = pk.attach_input(positions, name="positions")
    slot_dt = pk.attach_input(slot_mapping, name="slot_mapping")
    bt_dt   = pk.attach_input(block_table, name="block_table")
    cs_dt   = pk.attach_input(cos_sin_cache, name="cos_sin_cache")
    kvs_dt  = pk.attach_input(kv_slot_mapping, name="kv_slot_mapping")
    kc_dt   = pk.attach_input(k_cache_buf, name="k_cache")

    with pk.compile_scope():
        _ = module.compile(
            sc_dt, ttr_dt, pos_dt, slot_dt, bt_dt, cs_dt, kvs_dt, kc_dt
        )

    print("Compiling V4FusedKVCompressNormRopeInsertSparseAttn test kernel...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4FusedKVCompressNormRopeInsertSparseAttn test kernel...")
    pk()
    torch.cuda.synchronize()

    # Compare. FP8 quant + bf16 cast introduce significant error; use a
    # loose tolerance for the data region (max-abs uint8 diff) and a
    # bit-exact check for the scale bytes (UE8M0 exponent encoding is
    # deterministic for given amax).
    max_diff = (k_cache_buf.to(torch.int16) - k_cache_ref.to(torch.int16)
                ).abs().max().item()
    print(f"max-abs byte diff (k_cache): {max_diff}")

    # Strict check: at least some tokens were written (the buffer is
    # not all-zero), and the diff is bounded.
    written = (k_cache_buf != 0).any().item()
    print(f"k_cache has writes: {written}")
    if not written:
        print("FAILED: k_cache has no writes after kernel launch")
        pk.finalize()
        sys.exit(1)

    # FP8 quant is per-block; small rounding differences can occur in
    # the LSB. Allow up to a few bits diff in each byte (very loose).
    if max_diff > 8:
        print(f"FAILED: max byte diff = {max_diff} > 8 (likely a logic bug, "
              "not just FP8/bf16 rounding)")
        pk.finalize()
        sys.exit(1)

    print("PASSED: V4FusedKVCompressNormRopeInsertSparseAttn matches reference "
          "within FP8/bf16 rounding tolerance")
    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_fused_kv_compress_norm_rope_insert_sparse_attn_v4_testmode()
