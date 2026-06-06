"""Test: DeepseekV3MLA (absorbed decode, BF16) compile() vs a PyTorch oracle.

`DeepseekV3MLA.forward()` is intentionally stubbed (the absorbed-decode path is
tied to MPK runtime/paged-KV state), so this test builds its OWN correctness
oracle instead of calling `module.forward()`.

Pipeline verified (full module `compile()`):
    q_a_proj -> q_a_layernorm -> q_b_proj (KV-absorbed) -> kv_a_proj ->
    kv_a_layernorm(c_latent slice) -> RoPE(q fused / k) -> mla_kv_gather
    (append + materialise slab) -> mla_decode -> o_proj + residual.

Config: single layer, 1 decode token (mbt=mbr=1), kv_len=128 (one 128-token
tile => num_splits=1, so split-K decode collapses to a single per-head softmax),
full DeepSeek V3 attention dims (H=128, D_K=576, D_V=512). The KV cache is
pre-seeded with 127 prior-context rows; the gather appends this token at
position 127 (kv_start_pos = seq_len - num_new_tokens) and gathers [0,128) into
the contiguous decode slab (see mla_kv_cache_gather_sm100.cuh:64-92).

At num_splits=1 the decode codegen instantiates WRITE_FINAL=true: it writes the
final head-major attn output directly to its output slot and the reduce is
SKIPPED (matching builder.py:2707-2913). The split-K reduce only runs for
num_splits >= 2. (Earlier this module always ran decode->reduce, which fed the
reduce a head-major final output as a col-major partial + garbage LSE and
corrupted the result — fixed.)

The decode/reduce *math* is already proven numerically by
tests/runtime_python/blackwell/sm100_mla_mtp_decode/; here we additionally
verify the MODULE's wiring: the projections, the sliced kv_a_layernorm, the
GPT-J-interleaved RoPE on the absorbed [512:576) PE slice, the paged
append+gather, and the W_UV-fused o_proj. At num_splits=1 the oracle is a clean
per-head SDPA over the latent slab (K=full 576, V=first 512, deepseek scale).

Run:
  CUDA_VISIBLE_DEVICES=<free> python tests/runtime_python/test_mode/test_dsv3_mla_testmode.py
"""

import math
import os
import sys
from types import SimpleNamespace

import torch

import mirage
from mirage.mpk.persistent_kernel import PersistentKernel
from mirage.mpk.models.deepseek_v3.modeling import DeepseekV3MLA
from mirage.mpk.layers.mla.rope import _rotate_interleaved

# DeepSeek V3 MLA dims (hard-baked in the decode/reduce kernels).
H = 128            # num_attention_heads
HIDDEN = 7168
Q_LORA = 1536
KV_LORA = 512      # = D_V (absorbed v_head_dim) = c_latent width
D_PE = 64          # qk_rope_head_dim
D_K = KV_LORA + D_PE   # 576
ROPE_BASE = 10000.0
EPS = 1e-6


def _deepseek_softmax_scale() -> float:
    # Matches task_register.cc / pytorch_reference.py: q_head_dim=192, YARN mscale.
    mscale = 0.1 * math.log(40.0) + 1.0
    return (1.0 / math.sqrt(192.0)) * mscale * mscale


def _rmsnorm(x_f32, w, eps=EPS):
    var = x_f32.pow(2).mean(-1, keepdim=True)
    return x_f32 * torch.rsqrt(var + eps) * w.float()


def _build_cos_sin(max_seq, device, dtype):
    """Repeat-interleaved GPT-J cos/sin tables (max_seq, D_PE). Built so the
    kernel (reads even-index entry) and the eager `_rotate_interleaved` oracle
    agree: cos[2i] == cos[2i+1]."""
    half = D_PE // 2
    inv_freq = ROPE_BASE ** (-torch.arange(half, dtype=torch.float32, device=device) / half)
    pos = torch.arange(max_seq, dtype=torch.float32, device=device)
    ang = torch.outer(pos, inv_freq)                       # (max_seq, half)
    cos = torch.cos(ang).repeat_interleave(2, dim=-1)      # (max_seq, D_PE)
    sin = torch.sin(ang).repeat_interleave(2, dim=-1)
    return cos.to(dtype), sin.to(dtype)


def _rope_apply(pe_f32, cos_p, sin_p):
    """GPT-J interleaved RoPE (eager equivalent of the kernel), per rope.py."""
    return pe_f32 * cos_p + _rotate_interleaved(pe_f32) * sin_p


def test_dsv3_mla():
    device = "cuda"
    torch.manual_seed(0)

    kv_len = 128                # one full 128-token tile -> num_splits=1
    page_size = 128
    max_num_pages = 1
    max_seq_length = 128
    step_pos = kv_len - 1       # new token is at absolute position 127

    cfg = SimpleNamespace(
        hidden_size=HIDDEN, num_attention_heads=H, q_lora_rank=Q_LORA,
        kv_lora_rank=KV_LORA, qk_nope_head_dim=128, qk_rope_head_dim=D_PE,
        page_size=page_size,
    )

    # ---- Random weights (absorbed shapes the module stores) ----------------
    Wqa = torch.randn(Q_LORA, HIDDEN, dtype=torch.bfloat16, device=device) * 0.02
    Wqb = torch.randn(H * D_K, Q_LORA, dtype=torch.bfloat16, device=device) * 0.02
    Wkva = torch.randn(KV_LORA + D_PE, HIDDEN, dtype=torch.bfloat16, device=device) * 0.02
    Wo = torch.randn(HIDDEN, H * KV_LORA, dtype=torch.bfloat16, device=device) * 0.02
    q_a_ln = torch.randn(Q_LORA, dtype=torch.bfloat16, device=device) * 0.1 + 1.0
    kv_a_ln = torch.randn(KV_LORA, dtype=torch.bfloat16, device=device) * 0.1 + 1.0

    # ---- Inputs ------------------------------------------------------------
    x = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device=device) * 0.5
    residual = torch.randn(1, HIDDEN, dtype=torch.bfloat16, device=device) * 0.1
    # Prior context already stored in the cache (127 rows of [c_latent|k_pe]).
    prior_kv = torch.randn(kv_len - 1, D_K, dtype=torch.bfloat16, device=device) * 0.1

    cos, sin = _build_cos_sin(max_seq_length, device, torch.bfloat16)

    # ---- KV cache pool (num_layers=1, max_num_pages, page_size, D_K) --------
    ckv_kpe_cache = torch.zeros(
        1, max_num_pages, page_size, D_K, dtype=torch.bfloat16, device=device)
    ckv_kpe_cache[0, 0, :kv_len - 1, :] = prior_kv      # seed prior context

    # ---- Meta tensors (encode 1 new token, seq_len=128) --------------------
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    paged_kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    paged_kv_indices = torch.tensor([0], dtype=torch.int32, device=device)
    paged_kv_last = torch.tensor([page_size], dtype=torch.int32, device=device)  # last_page_len=128
    step = torch.tensor([step_pos], dtype=torch.int32, device=device)
    tokens = torch.zeros(1, max_seq_length, dtype=torch.int64, device=device)
    prompt_lengths = torch.tensor([kv_len], dtype=torch.int32, device=device)

    num_workers, num_schedulers = mirage.get_configurations_from_gpu(0)
    params = PersistentKernel.get_default_init_parameters()
    params.update(
        test_mode=True, num_workers=num_workers,
        num_local_schedulers=num_schedulers, mpi_rank=0, world_size=1,
        max_num_batched_tokens=1, max_num_batched_requests=1,
        max_seq_length=max_seq_length, max_num_pages=max_num_pages,
        page_size=page_size,
        meta_tensors={
            "step": step, "tokens": tokens, "prompt_lengths": prompt_lengths,
            "qo_indptr_buffer": qo_indptr,
            "paged_kv_indptr_buffer": paged_kv_indptr,
            "paged_kv_indices_buffer": paged_kv_indices,
            "paged_kv_last_page_len_buffer": paged_kv_last,
        },
        kv_cache={"k_cache": ckv_kpe_cache, "v_cache": ckv_kpe_cache},
    )
    pk = PersistentKernel(**params)

    out = torch.zeros(1, HIDDEN, dtype=torch.bfloat16, device=device)
    x_dt = pk.attach_input(x, name="x")
    res_dt = pk.attach_input(residual, name="residual")
    cos_dt = pk.attach_input(cos, name="cos")
    sin_dt = pk.attach_input(sin, name="sin")
    out_dt = pk.attach_input(out, name="out")

    with pk.compile_scope():
        m = DeepseekV3MLA(cfg, layer_idx=0, prefix="mla_").to(device)
        with torch.no_grad():
            m.q_a_proj_weight.copy_(Wqa)
            m.q_b_proj_weight.copy_(Wqb)
            m.kv_a_proj_with_mqa_weight.copy_(Wkva)
            m.o_proj_weight.copy_(Wo)
            m.q_a_layernorm.copy_(q_a_ln)
            m.kv_a_layernorm.copy_(kv_a_ln)
        # DeepseekV3MLA.compile() exposes probe_q / probe_kv debug outputs
        # (identity copies of the post-RoPE fused Q and the gathered KV slab)
        # for bisecting a wiring regression. They are OFF here: each probe adds
        # a second consumer of its buffer (the decode is the first), which makes
        # that buffer a fork-producer and trips the case-3 fork/join validator.
        # The cache-append sanity check below covers the gather without a probe.
        m.compile(x_dt, cos_dt, sin_dt, residual_dt=res_dt, output=out_dt,
                  probe_q=None, probe_kv=None)

    print("Compiling DeepseekV3MLA (absorbed decode)...")
    pk.compile(output_dir=os.path.dirname(__file__))
    pk()
    torch.cuda.synchronize()

    # ---- Oracle (fp32) -----------------------------------------------------
    ss = _deepseek_softmax_scale()
    cos_p = cos[step_pos].float()
    sin_p = sin[step_pos].float()
    x_f = x[0].float()
    q_a = _rmsnorm(x_f @ Wqa.float().t(), q_a_ln)                  # (Q_LORA,)
    q = (q_a @ Wqb.float().t()).reshape(H, D_K)                    # (H, 576)
    q_pe = _rope_apply(q[:, KV_LORA:].clone(), cos_p, sin_p)       # (H, 64)
    q = torch.cat([q[:, :KV_LORA], q_pe], dim=-1)                  # (H, 576)
    kv_a = x_f @ Wkva.float().t()                                  # (576,)
    c_latent = _rmsnorm(kv_a[:KV_LORA], kv_a_ln)                   # (512,)
    k_pe = _rope_apply(kv_a[KV_LORA:], cos_p, sin_p)               # (64,)
    new_kv = torch.cat([c_latent, k_pe])                          # (576,)
    kv_slab = torch.cat([prior_kv.float(), new_kv[None, :]], dim=0)  # (128, 576)
    scores = (q @ kv_slab.t()) * ss                              # (H, 128)
    attn_w = torch.softmax(scores, dim=-1)
    attn = attn_w @ kv_slab[:, :KV_LORA]                          # (H, 512)
    o = attn.reshape(-1) @ Wo.float().t() + residual[0].float()  # (7168,)
    ref = o[None, :]

    # ---- Paged-cache append sanity (no probe needed) -----------------------
    # The gather appends the new c_latent/k_pe into the paged cache in place.
    # Verify the new row (pos 127) matches the oracle's normed+rope'd new_kv.
    cache_now = ckv_kpe_cache[0, 0].float()                      # (128, 576)
    appended = cache_now[kv_len - 1]
    app_lat = (appended[:KV_LORA] - c_latent).abs().max().item()
    app_pe = (appended[KV_LORA:] - k_pe).abs().max().item()
    prior_ok = (cache_now[:kv_len - 1] - prior_kv.float()).abs().max().item()
    print(f"[CACHE] new_latent_absmax={app_lat:.5f} new_pe_absmax={app_pe:.5f} "
          f"prior_rows_absmax={prior_ok:.5f}")

    if out.isnan().any() or out.isinf().any():
        print(f"FAILED: NaN/Inf (out[0,:6]={out[0,:6]})")
        pk.finalize(); sys.exit(1)
    max_abs = (out.float() - ref).abs().max().item()
    max_rel = max_abs / max(ref.abs().max().item(), 1e-6)
    print(f"out[0,:6]: {out[0,:6]}")
    print(f"ref[0,:6]: {ref[0,:6]}")
    print(f"max abs {max_abs:.5f} rel {max_rel:.5f}")
    ok = max_rel < 0.10
    print("PASSED" if ok else "FAILED")
    pk.finalize()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    test_dsv3_mla()
