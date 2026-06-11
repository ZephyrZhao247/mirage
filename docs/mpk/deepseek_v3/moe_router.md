# `moe_router` — DeepSeek V3 kernel spec

> Part of the [DSv3 kernel-spec category](./README.md). Config: **TP=4 × EP=2** (world_size=8).

**Semantics:** sigmoid gate, per-group top-k, bias-corrected expert selection; emits
local-expert routing.

**Phase:** both.

**grid_dim:** `(1,1,1)`; single CTA; block `(256,1,1)`.

**Inputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `gate_logits` | `[T,E=256]` | bf16 | router scores |
| `bias` | `[E]` | f32 | per-expert routing bias |

**Outputs**

| name | shape | dtype | layout / meaning |
|---|---|---|---|
| `topk_weight` | `[T,Ep=8]` | f32 | normalized gate weights per selected expert |
| `routing_idx` | `[E_loc,T]` | int32 | per-local-expert token lists |
| `expert_mask` | `[E_loc+1]` | int32 | active-expert / per-expert counts |

**Params:** `E`, `Ep`, `n_group=8`, `topk_group=4`, `ep_rank` (local-expert range =
`[ep_rank·128, (ep_rank+1)·128)`).

**Shape variants**

| variant | dims |
|---|---|
| config-fixed | `E=256, Ep=8, n_group=8, topk_group=4`; `E_loc=128`/rank |

**Reuse:** `moe_topk_sigmoid_routing_layer`.
