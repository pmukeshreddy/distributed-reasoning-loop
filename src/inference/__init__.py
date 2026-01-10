"""
Inference module with Speculative Decoding support.
Optimizes inference throughput for reasoning models.
"""

from .speculative_decoding import (
    SpeculativeDecoder,
    SpeculativeConfig,
    DraftTargetPair,
)

from .vllm_engine import (
    VLLMEngine,
    VLLMConfig,
)

from .sglang_engine import (
    SGLangEngine,
    SGLangConfig,
)

__all__ = [
    "SpeculativeDecoder",
    "SpeculativeConfig",
    "DraftTargetPair",
    "VLLMEngine",
    "VLLMConfig",
    "SGLangEngine",
    "SGLangConfig",
]
