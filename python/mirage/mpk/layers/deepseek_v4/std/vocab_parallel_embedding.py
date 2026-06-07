"""V4-Flash ``vocab_parallel_embedding`` — REUSE alias for :class:`mirage.mpk.layers.Embed`.

Spec: ``docs/mpk/deepseek_v4/vllm_kernels/vocab_parallel_embedding.md``.

Decision: **REUSE** (for the ``tp_size=1`` contract; sharded path is
deferred — see the audit below).

Rationale
---------
For the V4-Flash NVIDIA production call sites
(``vllm/models/deepseek_v4/nvidia/model.py:1064`` and
``vllm/models/deepseek_v4/nvidia/mtp.py:205``), the spec is exactly
``F.embedding(input_ids.long(), weight)``. The ``tp_size > 1`` branch
adds a fused mask + ``masked_fill_`` + ``all_reduce`` epilogue
(spec lines 62-72) — V4-Flash's MPK port runs on a single rank during
unit tests, so this alias targets the ``tp_size=1`` fast path the spec
explicitly calls out (spec line 83).

The existing MPK ``Embed`` catalog
(``include/mirage/persistent_kernel/tasks/{ampere,hopper}/embedding{,_hopper}.cuh``
registered as task ``embedding``) implements precisely that fast path:
single-CTA gather of ``weight[input_tokens[i]]`` into the output, bf16
weight / bf16 output / int64 token IDs.

V4 ``TaskType`` enum slot ``TASK_VOCAB_PARALLEL_EMBEDDING_V4_SM100 = 352``
is RESERVED in ``runtime_header.h`` but UNUSED by this alias — calls
flow through the existing ``TASK_EMBEDDING`` task.

Audit
-----
* dtype: producer = int64 token IDs (caller's responsibility — vLLM
  calls ``.long()`` at ``vocab_parallel_embedding.py:491``; our test
  feeds int64 directly); weight = bf16; output = bf16. Same as the
  existing ``Embed`` catalog and the spec.
* layout: ``input_tokens [num_tokens]`` 1-D contiguous; weight
  ``[num_embeddings, embedding_dim]`` row-major; output
  ``[num_tokens, embedding_dim]`` row-major. Matches the spec.
* multi-batch: the .cuh embedding_kernel loops ``batch_idx`` from 0 to
  BATCH_SIZE, which is set by the host wrapper to ``num_tokens``. The
  V4 test below exercises ``num_tokens >= 2``.
* TP path: when ``tp_size > 1``, vLLM wraps with a fused-mask Inductor
  kernel + ``all_reduce``. NOT covered by this alias; deferred to a
  future ``V4VocabParallelEmbeddingTP`` composite that wires
  :class:`layers.AllReduce` after this Embed call.
* ``forward()``: provides the faithful PyTorch reference
  ``F.embedding(input_tokens, weight)``.

V4 production call sites:
* ``vllm/models/deepseek_v4/nvidia/model.py:1064`` — main model
  embedding lookup (PP-first rank only).
* ``vllm/models/deepseek_v4/nvidia/mtp.py:205`` — MTP draft step.
"""
from __future__ import annotations

from ...embedding.embed import Embed as _Embed


class V4VocabParallelEmbedding(_Embed):
    """V4-Flash ``vocab_parallel_embedding`` alias for the generic
    :class:`mirage.mpk.layers.Embed` (tp_size=1 fast path).

    Tensor contract (inherited):
      input_dt: (num_tokens,) int64, token IDs in ``[0, num_embeddings)``.
      weight:   (num_embeddings, embedding_dim) bf16, row-major.
      output:   (num_tokens, embedding_dim) bf16.

    Notes:
      * Single-CTA kernel; ``auto_grid_dim`` returns ``(1, 1, 1)`` and
        scaling the grid is unproductive.
      * The ``input_source`` kwarg on ``compile()`` selects whether the
        kernel reads token IDs from this DTensor (``input_source=1``) or
        from the runtime's rolling ``runtime_config.tokens``
        (``input_source=0``). The standard catalog test exercises
        ``input_source=1``.
      * TP shard masking (``tp_size > 1``) is NOT covered by this alias.
    """

    pass
