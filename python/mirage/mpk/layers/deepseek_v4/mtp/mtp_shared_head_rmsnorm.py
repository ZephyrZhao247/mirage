"""V4-Flash ``mtp_shared_head_rmsnorm`` -- REUSE alias for :class:`mirage.mpk.layers.RMSNorm`.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/mtp_shared_head_rmsnorm.md``.

Decision: **REUSE**.

Rationale
---------
The vLLM kernel ``mtp_shared_head_rmsnorm`` is the *plain* per-token RMSNorm
for ``SharedHead.norm`` in the MTP logits path:

  out[t, :] = (hidden[t, :].float() * rsqrt(var + eps) * weight).to(bf16)

with fp32 reduction and bf16 in / bf16 out. The existing MPK ``RMSNorm``
catalog (``layers.RMSNorm``, backed by the Hopper/Blackwell task
``rmsnorm_hopper`` at
``include/mirage/persistent_kernel/tasks/hopper/rmsnorm_hopper.cuh``)
matches the contract bit-for-bit:

* bf16 input / bf16 weight / bf16 output, row-major contiguous
  ``[num_tokens, HIDDEN]``.
* fp32 sum-of-squares via tree reduction, ``rsqrt(var + eps)``, weight
  multiply applied AFTER ``rsqrt`` (identical to ``_rmsnorm_row``).
* ``eps = 1e-6`` hard-coded in codegen -- matches V4-Flash's
  ``config.rms_norm_eps``.
* Grid: one CTA per token (kernel partitions on dim 0). Multi-batch is
  the default.

V4 ``TaskType`` enum slot ``TASK_MTP_SHARED_HEAD_RMSNORM_V4_SM100 = 390``
is RESERVED in ``runtime_header.h`` but UNUSED by this alias -- calls
flow through the existing ``TASK_RMS_NORM_HOPPER`` task (and the
corresponding ``rmsnorm_hopper`` registration in ``src/kernel/graph.cc``
/ ``task_register.cc``). Per the vLLM spec the only reason the kernel
exists as a separate Triton symbol upstream is so the MTP draft path
runs ONE consistent RMSNorm impl end-to-end + CUDA-graph friendliness --
neither concern applies inside MPK where the megakernel already provides
both. Future divergence (e.g., a fused MTP-logits-norm variant) can
reclaim slot 390.

Audit
-----
* dtype: producer = bf16 (post ``hc_head_fuse``), consumer = bf16
  (``LogitsProcessor`` lm_head GEMM). RMSNorm output dtype = input dtype.
  No silent cast.
* layout: row-major contiguous ``[num_tokens, hidden=4096]``. Matches.
* multi-batch: the kernel partitions on dim 0 with one CTA per token;
  multi-batch is the default and covered by the V4 test below
  (max_num_batched_requests=4).
* ``forward()``: provides the faithful PyTorch reference (fp32 reduction,
  bf16 output), inherited from :class:`mirage.mpk.layers.RMSNorm`.
"""
from __future__ import annotations

from ...norm.rmsnorm import RMSNorm as _RMSNorm


class V4MTPSharedHeadRMSNorm(_RMSNorm):
    """V4-Flash ``mtp_shared_head_rmsnorm`` alias for the generic MPK
    :class:`RMSNorm`.

    Subclasses :class:`mirage.mpk.layers.RMSNorm` with no behavioral
    changes; the only purpose of the subclass is to give the V4 MTP
    catalog a distinct name model authors can import as
    ``from mirage.mpk.layers.deepseek_v4.mtp import V4MTPSharedHeadRMSNorm``.

    Tensor contract (inherited):
      x:      (num_tokens, hidden_size) bf16, row-major contiguous.
      weight: (hidden_size,)            bf16, learnable per-channel scale.
      out:    (num_tokens, hidden_size) bf16, same layout as ``x``.

    Notes:
      * ``hidden_size`` must satisfy ``hidden_size % 256 == 0`` on
        Blackwell (NUM_THREADS=256) and ``hidden_size * 2 / 256 >= 4``
        (i.e. ``hidden_size >= 512``). V4 hidden=4096 is well above.
      * ``eps`` argument is honored by the PyTorch reference; the kernel
        hard-codes ``1e-6f`` in codegen, which matches V4-Flash's
        ``config.rms_norm_eps``.
    """

    pass
