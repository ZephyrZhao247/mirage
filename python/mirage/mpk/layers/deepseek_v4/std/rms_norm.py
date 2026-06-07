"""V4-Flash ``rms_norm`` — REUSE alias for :class:`mirage.mpk.layers.RMSNorm`.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/rms_norm.md``.

Decision: **REUSE**.

Rationale
---------
The vLLM spec is a bf16 residual-free RMSNorm with a learnable per-channel
scale, fp32 reduction, and ``eps = config.rms_norm_eps`` (V4-Flash uses
``1e-6``). The existing MPK ``RMSNorm`` catalog (``layers.RMSNorm``,
backed by the Hopper/Blackwell task ``rmsnorm_hopper`` at
``include/mirage/persistent_kernel/tasks/hopper/rmsnorm_hopper.cuh``)
matches the contract bit-for-bit:

* bf16 input / bf16 weight / bf16 output, row-major contiguous.
* fp32 sum-of-squares, ``rsqrt(var + eps)``, then ``(scalar_t)(x*scale)*w``
  (the kernel does the cast BEFORE the weight multiply, identical to the
  vLLM kernel's ``layernorm_kernels.cu:81`` convention).
* ``eps = 1e-6`` hard-coded in the codegen — matches V4-Flash's
  ``rms_norm_eps``.

V4 ``TaskType`` enum slot ``TASK_RMS_NORM_V4_SM100 = 350`` is RESERVED in
``runtime_header.h`` but UNUSED by this alias — calls flow through the
existing ``TASK_RMS_NORM_HOPPER`` task (and the corresponding
``rmsnorm_hopper`` registration in ``src/kernel/graph.cc`` /
``task_register.cc``).

Audit
-----
* dtype: producer = bf16 (post-mhc_head_fuse hidden state), consumer =
  bf16 (lm_head GEMM). RMSNorm output dtype = input dtype. No silent
  cast.
* layout: row-major contiguous, ``[num_tokens, hidden_size]``. Matches
  the existing kernel's contract.
* multi-batch: the kernel partitions on ``dim 0`` with one CTA per
  token; multi-batch is supported and covered by the V4 test below.
* ``forward()``: provides the faithful PyTorch reference (fp32 reduction,
  bf16 output).

V4 production call site (the *only* standalone launch site per spec):
``vllm/models/deepseek_v4/nvidia/model.py:1103`` — final trunk norm
before lm_head.
"""
from __future__ import annotations

from ...norm.rmsnorm import RMSNorm as _RMSNorm


class V4RMSNorm(_RMSNorm):
    """V4-Flash ``rms_norm`` alias for the generic MPK :class:`RMSNorm`.

    Subclasses :class:`mirage.mpk.layers.RMSNorm` with no behavioral
    changes; the only purpose of the subclass is to give the V4 catalog
    a distinct name model authors can import as
    ``from mirage.mpk.layers.deepseek_v4.std import V4RMSNorm``.

    Tensor contract (inherited):
      x:      (num_tokens, hidden_size) bf16, row-major contiguous.
      weight: (hidden_size,)            bf16, learnable per-channel scale.
      out:    (num_tokens, hidden_size) bf16, same layout as ``x``.

    Notes:
      * ``hidden_size`` must satisfy ``hidden_size % 256 == 0`` on
        Blackwell (NUM_THREADS=256) and ``hidden_size * 2 / 256 >= 4``
        (i.e. ``hidden_size >= 512``). V4 trunk hidden=4096 is well above
        the bar.
      * ``eps`` argument is honored by the PyTorch reference; the kernel
        hard-codes ``1e-6f`` in codegen, which matches V4-Flash's
        ``config.rms_norm_eps``.
    """

    pass
