"""DeepSeek V4-Flash model composition for MPK.

Currently exports :class:`DeepseekV4Block` (per-layer composite) used by
the Wave-3 per-layer correctness gate. Higher-level assembly
(`DeepseekV4Model`, builder integration) is a follow-up.
"""

from .block import DeepseekV4Block

__all__ = ["DeepseekV4Block"]
