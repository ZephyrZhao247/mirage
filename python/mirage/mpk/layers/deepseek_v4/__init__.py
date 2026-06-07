"""DeepSeek V4-Flash catalog modules.

One subpackage per V4 subsystem (std layers, hyper-connections, attention,
compressor, indexer, MoE routing, MoE compute, MTP). Each subpackage holds
one module per vLLM-canonical kernel spec at
``docs/mpk/deepseek_v4/vllm_kernels/<kernel>.md``.

The modules in this subtree are thin wrappers; many alias an existing
generic MPK layer (e.g., :class:`mirage.mpk.layers.RMSNorm` covers V4's
``rms_norm``) and reserve the pre-allocated V4 ``TaskType`` enum slot for
documentation / future divergence only.
"""
