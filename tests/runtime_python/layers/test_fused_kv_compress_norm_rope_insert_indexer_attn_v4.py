"""V4-Flash ``fused_kv_compress_norm_rope_insert_indexer_attn`` (FP8
K-side, NEW kernel, head_dim=128) test.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/
fused_kv_compress_norm_rope_insert_indexer_attn.md``.

Class-B sibling under use_fp4_cache=False. Multi-batch.
"""

import os
import sys

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.layers.deepseek_v4.indexer import (
    V4FusedKVCompressNormRopeInsertIndexerAttn,
)


def _build_cos_sin_cache(max_pos: int, rope_dim: int,
                         base: float = 160000.0,
                         device: str = "cuda") -> torch.Tensor:
    half = rope_dim // 2
    inv_freq = base ** (-torch.arange(0, half, dtype=torch.float32, device=device) /
                        half)
    positions = torch.arange(max_pos, dtype=torch.float32, device=device)
    freqs = positions[:, None] * inv_freq[None, :]
    cos = freqs.cos()
    sin = freqs.sin()
    return torch.cat([cos, sin], dim=-1)


def test_fused_kv_compress_norm_rope_insert_indexer_attn_v4_testmode():
    device = "cuda"
    torch.manual_seed(0)

    head_size = 128
    compress_ratio = 4
    block_size = 16
    kv_cache_block_size = 16
    num_blocks = 4
    num_kv_blocks = 4
    max_blocks = 4

    # 2 requests x 4 tokens. Tokens 3 and 7 hit the boundary.
    num_tokens = 8
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    token_to_req_indices = torch.tensor(
        [0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int32, device=device
    )
    slot_mapping = torch.tensor(
        [0, 1, 2, 3, 16, 17, 18, 19], dtype=torch.int64, device=device
    )
    kv_slot_mapping = torch.tensor(
        [0, 1, 2, 3, 16, 17, 18, 19], dtype=torch.int64, device=device
    )
    block_table = torch.tensor(
        [[0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.int32, device=device
    )

    # State cache: [num_blocks, block_size, 4*head_size = 512].
    state_cache = torch.randn(
        num_blocks, block_size, 4 * head_size,
        dtype=torch.float32, device=device,
    )

    max_pos = 256
    rope_head_dim = 64
    cos_sin_cache = _build_cos_sin_cache(max_pos, rope_head_dim,
                                          base=160000.0, device=device)

    # K cache: [num_kv_blocks, kv_block_size, 1, 132] for FP8 path
    # (128 FP8 + 4 fp32-scale).
    token_stride = 128
    scale_dim = 4
    k_cache_buf = torch.zeros(
        num_kv_blocks, kv_cache_block_size, 1, token_stride + scale_dim,
        dtype=torch.uint8, device=device,
    )
    k_cache_ref = k_cache_buf.clone()

    module = V4FusedKVCompressNormRopeInsertIndexerAttn(
        block_size=block_size,
        kv_cache_block_size=kv_cache_block_size,
        rms_norm_eps=1e-6,
        prefix="v4_idxK_",
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

    print("Compiling V4FusedKVCompressNormRopeInsertIndexerAttn (FP8) ...")
    folder_path = os.path.dirname(__file__)
    pk.compile(output_dir=folder_path)

    print("Running V4FusedKVCompressNormRopeInsertIndexerAttn (FP8) ...")
    pk()
    torch.cuda.synchronize()

    max_diff = (k_cache_buf.to(torch.int16) - k_cache_ref.to(torch.int16)
                ).abs().max().item()
    print(f"max-abs byte diff (k_cache): {max_diff}")
    written = (k_cache_buf != 0).any().item()
    print(f"k_cache has writes: {written}")
    if not written:
        print("FAILED: k_cache has no writes after kernel launch")
        pk.finalize()
        sys.exit(1)
    # The fp32 scale bytes are exact (UE8M0-discrete pow2), the FP8 data
    # may differ by a few LSBs due to reduction-order variance. Allow up
    # to 8.
    if max_diff > 8:
        print(f"FAILED: max byte diff = {max_diff} > 8 (likely a logic bug)")
        pk.finalize()
        sys.exit(1)
    print("PASSED: V4FusedKVCompressNormRopeInsertIndexerAttn matches reference")
    pk.finalize()
    print("Test completed successfully!")


if __name__ == "__main__":
    test_fused_kv_compress_norm_rope_insert_indexer_attn_v4_testmode()
