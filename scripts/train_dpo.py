#!/usr/bin/env python3
"""
Script to train a reasoning model using DPO.
Supports both standard DPO and rejection sampling DPO.
"""

import argparse
import logging
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training import DPOTrainerConfig, ReasoningDPOTrainer, RejectionSamplingDPO
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_dpo_data(data_path: str):
    """Load DPO training data from file."""
    data = []
    
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            # Ensure required keys exist
            if "prompt" in item and "chosen" in item and "rejected" in item:
                data.append(item)
    
    logger.info(f"Loaded {len(data)} DPO pairs from {data_path}")
    return data


def main():
    parser = argparse.ArgumentParser(description="Train reasoning model with DPO")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to DPO training data (JSONL)",
    )
    parser.add_argument(
        "--eval-data-path",
        type=str,
        default=None,
        help="Path to evaluation data (JSONL)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to train (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./dpo_output",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.1,
        help="DPO beta parameter",
    )
    parser.add_argument(
        "--use-lora",
        action="store_true",
        default=True,
        help="Use LoRA for training",
    )
    parser.add_argument(
        "--no-lora",
        action="store_true",
        help="Disable LoRA (full fine-tuning)",
    )
    parser.add_argument(
        "--rejection-sampling",
        action="store_true",
        help="Use rejection sampling DPO",
    )
    
    args = parser.parse_args()
    
    # Load config
    if Path(args.config).exists():
        config = OmegaConf.load(args.config)
    else:
        config = OmegaConf.create({})
    
    # Override with command line args
    model_name = args.model or config.get("data_generator", {}).get(
        "student_model", "Qwen/Qwen2.5-7B-Instruct"
    )
    
    # Create DPO config
    dpo_config = DPOTrainerConfig(
        model_name=model_name,
        beta=args.beta,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        output_dir=args.output_dir,
        use_lora=args.use_lora and not args.no_lora,
    )
    
    # Load data
    train_data = load_dpo_data(args.data_path)
    eval_data = load_dpo_data(args.eval_data_path) if args.eval_data_path else None
    
    logger.info(f"Training configuration:")
    logger.info(f"  Model: {model_name}")
    logger.info(f"  Beta: {args.beta}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Epochs: {args.num_epochs}")
    logger.info(f"  LoRA: {dpo_config.use_lora}")
    logger.info(f"  Output: {args.output_dir}")
    
    # Train
    if args.rejection_sampling:
        trainer = RejectionSamplingDPO(dpo_config)
        # For rejection sampling, we need problems not pairs
        logger.warning("Rejection sampling mode - data should contain problems")
    else:
        trainer = ReasoningDPOTrainer(dpo_config)
        trainer.train(train_data, eval_data)
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
