"""DeepSeek-V4-Flash Hyper-Connections (mHC) layers.

Backed by ``tasks/blackwell/mhc_{pre,post,head}_sm100.cuh`` plus
``sum_of_squares_sm100.cuh`` (used by mhc_prenorm_gemm v1).
See ``docs/mpk/deepseek_v4/hc.md`` for the full pipeline.
"""

from .mhc_prenorm_gemm import MhcPrenormGemm
from .mhc_pre import MhcPre
from .mhc_post import MhcPost
from .mhc_head import MhcHead

__all__ = ["MhcPrenormGemm", "MhcPre", "MhcPost", "MhcHead"]
