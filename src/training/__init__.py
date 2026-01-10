"""
Training module for RL-based reasoning model training.
Supports DPO, GRPO, Rejection Sampling, and standard SFT.
"""

from .dpo_trainer import (
    DPOTrainerConfig,
    ReasoningDPOTrainer,
    RejectionSamplingDPO,
)

from .grpo_trainer import (
    GRPOConfig,
    ReasoningGRPOTrainer,
)

from .reward_model import (
    RewardModel,
    RewardModelConfig,
    ProcessRewardModel,
)

from .sft_trainer import (
    SFTTrainerConfig,
    ReasoningSFTTrainer,
    SFTFromSyntheticData,
)

__all__ = [
    # DPO
    "DPOTrainerConfig",
    "ReasoningDPOTrainer",
    "RejectionSamplingDPO",
    # GRPO
    "GRPOConfig",
    "ReasoningGRPOTrainer",
    # Reward Model
    "RewardModel",
    "RewardModelConfig",
    "ProcessRewardModel",
    # SFT
    "SFTTrainerConfig",
    "ReasoningSFTTrainer",
    "SFTFromSyntheticData",
]
