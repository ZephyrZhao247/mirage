"""MLA split-K decode + reduce catalog modules.

Wraps two SM100 tasks (kernels under
``include/mirage/persistent_kernel/tasks/blackwell/``):

* :class:`MLADecode`  -> ``mla_decode_sm100``  (``mla_decode_sm100.cuh``)
* :class:`MLAReduce`  -> ``mla_reduce_sm100``  (``mla_reduce_sm100.cuh``)

Both kernels bake in ``NUM_HEADS=128``, ``D_K=576``, ``D_V=512`` for
DeepSeek V3. Each task instance handles ONE request (request_id comes
from ``task_desc->task_metadata``); paged KV is consumed via the MPK
runtime's ``paged_kv_indptr_buffer`` / ``paged_kv_last_page_len_buffer``.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .._base import MPKModule
from ...context import current_pk

from ....core import DTensor


GridDim = Tuple[int, int, int]
BlockDim = Tuple[int, int, int]


class MLADecode(MPKModule):
    """MLA split-K decode (partial output + partial LSE).

    Task ``mla_decode_sm100``. Pair with :class:`MLAReduce` to obtain
    the final per-head output. Inputs: ``q_input`` (B*Q_LEN*H, D_K)
    bf16 (TMA-desc) and ``kv_input`` (B*max_seq_len_pad, D_K) bf16
    (TMA-desc). Outputs split-K partial-O ``(B*Q_LEN*num_splits, H*D_V)``
    and partial-LSE ``(B*Q_LEN*num_splits, H)`` float32.
    """

    def __init__(
        self,
        num_heads: int,
        d_k: int,
        d_v: int,
        num_splits: int,
        kv_len: int,
        *,
        q_len: int = 1,
        prefix: str = "",
    ) -> None:
        # DISABLED — buggy kernel. MLADecode registers "mla_decode_sm100",
        # whose QK tcgen05 MMA writes only 64 of 128 score columns, so the
        # attention output collapses (verified: scores correct for K-cols 0-63,
        # exactly 0 for 64-127; not fixed by the SMEM-descriptor LBO field).
        # Use the demo-proven decode instead: pk.mla_mtp_decode_layer (registers
        # "mla_mtp_decode_sm100"), as in models/deepseek_v3/{builder.py,
        # modeling.py}. (MLAReduce was rerouted to "mla_mtp_reduce_sm100".)
        raise RuntimeError(
            "MLADecode is disabled: the 'mla_decode_sm100' kernel is buggy "
            "(its QK tcgen05 MMA produces only 64 of 128 score columns). Use "
            "pk.mla_mtp_decode_layer (the 'mla_mtp_decode_sm100' kernel) "
            "instead — that is the path the legacy demo uses for MLA decode."
        )
        super().__init__(prefix=prefix)
        self.num_heads = num_heads
        self.d_k = d_k
        self.d_v = d_v
        self.num_splits = num_splits
        self.kv_len = kv_len
        self.q_len = q_len

    def forward(self, *args, **kwargs):
        """Not implemented: depends on MPK runtime meta-tensors (paged KV
        indptr, qo indptr) and the split-K partition scheme."""
        raise NotImplementedError("MLADecode.forward(): use test-mode PK driver.")

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Grid ``(num_splits, num_head_groups, max_num_batched_requests)``.

        ``mla_decode_sm100`` (TASK_MLA_DECODE_SM100) maps grid coords to task
        metadata as ``kv_idx=bid.x`` (split), ``request_id=bid.y``
        (batch in single-query / head-group otherwise), ``merge_task_offset=
        bid.z``. For decode (q_len=1) the kernel packs ALL heads into one
        block (hpb=128/q_len, num_head_groups=128/hpb), so grid.y must be the
        BATCH dimension, not num_heads. Passing num_heads here made grid.y span
        0..127 -> request_id(bi)=0..127 -> Oout=Oa+bi*D_V*128 wrote far past
        the (mbr*num_splits, H*D_V) partial buffer (OOB / illegal instruction).
        """
        pk = current_pk()
        q_len = self.q_len
        hpb = min(128 // max(q_len, 1), self.num_heads)
        while hpb > 0 and self.num_heads % hpb != 0:
            hpb -= 1
        if hpb <= 0:
            hpb = 1
        num_head_groups = self.num_heads // hpb
        # Single-query: grid.y carries the batch (request_id=bid.y), so
        # num_head_groups must be 1 and grid.y == mbr. q_len>1: grid.y is the
        # head-group axis and the batch rides grid.z (merge_task_offset).
        if q_len == 1:
            return (self.num_splits, pk.max_num_batched_requests, 1)
        return (self.num_splits, num_head_groups, pk.max_num_batched_requests)

    def default_block_dim(self) -> BlockDim:
        return (128, 1, 1)

    def compile(
        self,
        q_input: DTensor,
        kv_input: DTensor,
        output_partial: DTensor,
        output_lse: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> Tuple[DTensor, DTensor]:
        """Register ``mla_decode_sm100`` (codegen routes to ``mla_mtp_decode_sm100_task_impl``).

        Tensor contract:
          q_input:        (R*Q_LEN*NUM_HEADS=128, D_K=576) bf16, row-major, TMA-desc (input_tma_desc_ptrs[0][0]).
          kv_input:       (R*KV_LEN, D_K=576) bf16, contiguous gathered paged-KV slab, TMA-desc (input_tma_desc_ptrs[1][0]).
          output_partial: (R*Q_LEN*NUM_SPLITS, NUM_HEADS*D_V=128*512) bf16, partial-O per split (kernel ``Oa``, output_ptrs[0]).
          output_lse:     (R*Q_LEN*NUM_SPLITS, NUM_HEADS=128) fp32, partial-LSE per split (kernel ``La``, output_ptrs[1]).

        Notes: paged-KV via ``paged_kv_indptr_buffer`` / ``paged_kv_last_page_len_buffer``;
        single-tile-per-split required (sk = ceil(kv_len/128)). For ``q_len > 1`` partial maps are
        ``(-1,-1,-1)`` so every block sees the full base and applies its own ``bi*nhg*sk + gi*sk + si`` offset.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        q_len = self.q_len
        params = [
            self.num_heads,
            self.d_k,
            self.d_v,
            self.num_splits,
            self.kv_len,
            q_len,
        ]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        tb_graph.new_input(q_input, (0, -1, -1), -1, True)
        tb_graph.new_input(kv_input, (0, -1, -1), -1, True)
        partial_map = (-1, -1, -1) if q_len > 1 else (0, -1, -1)
        tb_graph.new_input(output_partial, partial_map, -1, True)
        tb_graph.new_input(output_lse, partial_map, -1, True)
        pk.kn_graph.customized(
            [q_input, kv_input, output_partial, output_lse], tb_graph
        )
        pk.kn_graph.register_task(tb_graph, "mla_decode_sm100", params)
        return output_partial, output_lse


class MLAReduce(MPKModule):
    """MLA split-K reduce: merges per-split partials into final O.

    Task ``mla_reduce_sm100``. Each block reduces a ``d_count`` slice
    of the V dim (``D_V=512``) for one head and one batch. Inputs:
    ``input_partial`` and ``input_lse`` from the decode; output
    ``(B, NUM_HEADS, D_V)`` bf16 ready for ``o_proj``.
    """

    def __init__(
        self,
        num_heads: int,
        d_v: int,
        num_splits: int,
        d_start: int,
        d_count: int,
        *,
        q_len: int = 1,
        prefix: str = "",
    ) -> None:
        super().__init__(prefix=prefix)
        self.num_heads = num_heads
        self.d_v = d_v
        self.num_splits = num_splits
        self.d_start = d_start
        self.d_count = d_count
        self.q_len = q_len

    def forward(self, *args, **kwargs):
        """Not implemented: tied to the upstream decode's split-K layout."""
        raise NotImplementedError("MLAReduce.forward(): use test-mode PK driver.")

    # ``mla_mtp_reduce_sm100_task_impl<256>`` writes exactly two D_V columns per
    # block (``d = dv_base + lane`` with ``lane = threadIdx.x / 128`` in {0,1}),
    # so the D_V axis must be striped in steps of 2 across grid.x with
    # ``dv_base = kv_idx * RD_DV``. RD_DV=2 gives full D_V coverage.
    RD_DV = 2

    def auto_grid_dim(self, *_: DTensor) -> GridDim:
        """Grid ``(D_V / RD_DV, num_head_groups, max_num_batched_requests)``.

        ``mla_mtp_reduce_sm100`` (TASK_MLA_MTP_REDUCE_SM100) maps grid coords as
        ``kv_idx=bid.x`` (dv-block -> dv_base = kv_idx*RD_DV),
        ``request_id=bid.y`` (head group), ``merge_task_offset=bid.z`` (batch).
        The old config registered ``mla_reduce_sm100`` with grid.y=num_heads and
        a compile-time ``dv_base=d_start=0``, so it (a) ran 128x redundantly and
        (b) only ever wrote D_V columns 0 and 1 of each head, leaving 510/512
        columns zero (attn_out ~0 -> rel ~0.96). Route to the proven MTP reduce.
        """
        pk = current_pk()
        q_len = self.q_len
        hpb = min(128 // max(q_len, 1), self.num_heads)
        while hpb > 0 and self.num_heads % hpb != 0:
            hpb -= 1
        if hpb <= 0:
            hpb = 1
        num_head_groups = self.num_heads // hpb
        d_blocks = (self.d_v + self.RD_DV - 1) // self.RD_DV
        return (d_blocks, num_head_groups, pk.max_num_batched_requests)

    def default_block_dim(self) -> BlockDim:
        return (256, 1, 1)

    def compile(
        self,
        input_partial: DTensor,
        input_lse: DTensor,
        output: DTensor,
        *,
        grid_dim: Optional[GridDim] = None,
        block_dim: Optional[BlockDim] = None,
    ) -> DTensor:
        """Register ``mla_mtp_reduce_sm100`` (the runtime-proven split-K merge).

        Tensor contract:
          input_partial: (R*Q_LEN*NUM_SPLITS, NUM_HEADS*D_V=128*512) bf16, partial-O from MLADecode (input_ptrs[0]).
          input_lse:    (R*Q_LEN*NUM_SPLITS, NUM_HEADS=128) fp32, partial-LSE from MLADecode (input_ptrs[1]).
          output:       (B, NUM_HEADS=128, D_V=512) bf16, final attn output ready for ``o_proj`` (output_ptrs[0]).

        Notes: ``mla_mtp_reduce_sm100`` params are ``[num_head_groups, q_len,
        num_splits, RD_DV]``; each block reduces RD_DV(=2) D_V columns for one
        head group / batch (dv_base = kv_idx*RD_DV). ``d_start``/``d_count`` from
        ``__init__`` are accepted for API compatibility but the MTP reduce
        always covers the full D_V via the grid.x stripe.
        """
        pk = current_pk()
        if grid_dim is None:
            grid_dim = self.auto_grid_dim()
        if block_dim is None:
            block_dim = self.default_block_dim()

        from ....core import CyTBGraph
        from ....kernel import TBGraph

        q_len = self.q_len
        hpb = min(128 // max(q_len, 1), self.num_heads)
        while hpb > 0 and self.num_heads % hpb != 0:
            hpb -= 1
        if hpb <= 0:
            hpb = 1
        num_head_groups = self.num_heads // hpb
        params = [num_head_groups, q_len, self.num_splits, self.RD_DV]

        tb_graph = TBGraph(CyTBGraph(grid_dim, block_dim, 1, 64))
        # The MTP reduce kernel computes its own (Oa/La/O) offsets from the task
        # metadata (gi/bi/dv_base), so MPK must NOT auto-partition any axis —
        # every block needs the full base pointer (matches mla_mtp_reduce_layer).
        tb_graph.new_input(input_partial, (-1, -1, -1), -1, True)
        tb_graph.new_input(input_lse, (-1, -1, -1), -1, True)
        tb_graph.new_input(output, (-1, -1, -1), -1, True)
        pk.kn_graph.customized(
            [input_partial, input_lse, output], tb_graph
        )
        pk.kn_graph.register_task(tb_graph, "mla_mtp_reduce_sm100", params)
        return output
