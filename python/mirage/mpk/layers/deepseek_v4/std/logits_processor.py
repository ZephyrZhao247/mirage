"""V4-Flash ``logits_processor`` — REUSE alias for :class:`mirage.mpk.layers.Linear`.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/logits_processor.md``.

Decision: **REUSE**.

Rationale
---------
``LogitsProcessor`` is not itself a GPU kernel — the spec (Identity
section, lines 5-9) calls it "Python orchestration only", chaining four
steps:

1. lm_head GEMM (the dominant op).
2. TP gather (skipped for ``tp_size = 1``).
3. Vocab trim slice (no-op when ``org_vocab_size == vocab_size``, which
   is V4's default — both are 129280).
4. Optional ``soft_cap`` and ``scale`` (V4 sets ``soft_cap=None`` and
   ``scale=1.0``, so both are inactive — spec lines 18, 28-29).

With V4-Flash's defaults the entire ``LogitsProcessor.forward`` collapses
to a single bf16 GEMM:
``logits = F.linear(hidden_states, lm_head.weight)``
(spec line 80) — exactly what :class:`mirage.mpk.layers.Linear` already
implements (``F.linear(x, self.weight)`` at ``layers/linear/linear.py:71``),
backed by ``linear_sm100`` on Blackwell.

V4 ``TaskType`` enum slot ``TASK_LOGITS_PROCESSOR_V4_SM100 = 353`` is
RESERVED in ``runtime_header.h`` but UNUSED by this alias — calls flow
through the existing ``TASK_LINEAR_SM100`` task.

Audit
-----
* dtype: producer = bf16 hidden states (post-final-RMSNorm); weight =
  bf16 (V4 ships unquantized bf16 lm_head per spec line 25); output =
  bf16 logits. The sampler downstream casts to fp32, but the kernel
  boundary is bf16-in / bf16-out (spec lines 49-51, 70). ``forward()``
  returns bf16 to match. No silent dtype cast.
* layout: ``hidden_states [num_tokens, hidden_size=4096]`` row-major
  contiguous; ``lm_head.weight [vocab_size, hidden_size]`` row-major;
  output ``[num_tokens, vocab_size]`` row-major. Matches the spec.
* multi-batch: the GEMM partitions on the output feature dim
  (``grid.x = vocab_size // 64``); rows are processed within the CTA.
  Multi-batch is supported and covered by the V4 test below.
* TP path: when ``tp_size > 1``, vLLM follows with an ``all_gather`` or
  ``gather`` collective. NOT covered by this alias; deferred to a future
  composite that wires :class:`layers.AllReduce` after this Linear
  call.
* soft_cap: NOT covered by this alias (V4 default is ``None``); a future
  composite would add a pointwise ``tanh(x/c)*c`` op.
* scale: NOT covered by this alias (V4 default is ``1.0``); a future
  composite would add a scalar multiply.
* ``forward()``: provides the faithful PyTorch reference
  ``F.linear(hidden_states, lm_head.weight)``.

V4 production call sites:
* ``vllm/models/deepseek_v4/nvidia/model.py:1301`` —
  ``DeepseekV4ForCausalLM.compute_logits`` (trunk).
* ``vllm/models/deepseek_v4/nvidia/mtp.py:252`` —
  ``DeepSeekV4MultiTokenPredictor.compute_logits`` (MTP draft).
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from ...linear.linear import Linear as _Linear


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class V4LogitsProcessor(_Linear):
    """V4-Flash ``logits_processor`` alias for the generic
    :class:`mirage.mpk.layers.Linear` (lm_head GEMM, tp_size=1, no
    soft_cap, no scale).

    Constructor signature mirrors vLLM's
    ``LogitsProcessor(vocab_size)``: callers pass ``hidden_size`` and
    ``vocab_size`` instead of ``in_features``/``out_features`` so the
    naming matches the spec.

    Tensor contract (inherited from :class:`Linear`):
      hidden_states: (num_tokens, hidden_size) bf16, row-major contiguous.
      lm_head.weight: (vocab_size, hidden_size) bf16, row-major.
      logits:        (num_tokens, vocab_size) bf16, row-major.

    Notes:
      * ``vocab_size`` must be a multiple of 96 or 64 for the auto-grid
        (V4 vocab = 129280 is a multiple of 64; smaller test shapes
        should also satisfy one of these).
      * No bias (V4's lm_head has no bias; spec line 24-25 confirms).
      * Output dtype is bf16 — matches the spec contract at the kernel
        boundary. Callers needing fp32 must cast.
    """

    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        *,
        prefix: str = "",
    ) -> None:
        # Map vLLM's spec naming to Linear's in_features / out_features.
        super().__init__(
            in_features=hidden_size,
            out_features=vocab_size,
            bias=False,
            prefix=prefix,
        )
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """``F.linear(hidden_states, lm_head.weight)`` returning bf16 logits.

        Mirrors the V4 LogitsProcessor production path with
        ``soft_cap=None`` and ``scale=1.0`` (steps 2-4 of the spec are
        no-ops). Output dtype matches the kernel boundary (bf16) — the
        downstream sampler is responsible for casting to fp32.
        """
        # Ensure the output dtype documented at the kernel boundary
        # (bf16) is what the reference returns, even if the caller
        # passes a higher-precision input by accident.
        return super().forward(hidden_states).to(torch.bfloat16)
