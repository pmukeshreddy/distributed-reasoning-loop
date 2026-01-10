#!/usr/bin/env python3
"""
Main script to run the full distributed reasoning loop pipeline.
Orchestrates data generation, training, and evaluation.
"""

import argparse
import logging
import sys
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_data_generation(config, args):
    """Run synthetic data generation phase."""
    from data_generator import SyntheticDataPipeline, GenerationConfig
    from data_generator.cot_generator import InferenceBackend
    
    logger.info("=" * 50)
    logger.info("Phase 1: Synthetic Data Generation")
    logger.info("=" * 50)
    
    gen_config = GenerationConfig(
        model_name=config.data_generator.teacher_model,
        backend=InferenceBackend.VLLM,
        num_paths=config.data_generator.num_cot_paths,
        max_new_tokens=config.data_generator.max_new_tokens,
        temperature=config.data_generator.temperature,
    )
    
    pipeline = SyntheticDataPipeline(
        generator_config=gen_config,
        dataset_name=args.dataset,
        output_dir=f"{config.general.output_dir}/synthetic_data",
    )
    
    samples, pairs = pipeline.run(
        subset_size=args.subset_size,
        batch_size=args.batch_size,
    )
    
    logger.info(f"Generated {len(samples)} samples, {len(pairs)} DPO pairs")
    return samples, pairs


def run_sft_training(config, data_path):
    """Run supervised fine-tuning phase."""
    from training import SFTTrainerConfig, SFTFromSyntheticData
    
    logger.info("=" * 50)
    logger.info("Phase 2a: Supervised Fine-Tuning")
    logger.info("=" * 50)
    
    sft_config = SFTTrainerConfig(
        model_name=config.data_generator.student_model,
        learning_rate=config.training.learning_rate * 2,  # Higher LR for SFT
        batch_size=config.training.batch_size,
        num_epochs=1,  # Quick SFT pass
        output_dir=f"{config.general.output_dir}/sft_model",
    )
    
    trainer = SFTFromSyntheticData(sft_config, data_path)
    trainer.train()
    
    return sft_config.output_dir


def run_dpo_training(config, data_path, base_model=None):
    """Run DPO training phase."""
    from training import DPOTrainerConfig, ReasoningDPOTrainer
    import json
    
    logger.info("=" * 50)
    logger.info("Phase 2b: DPO Training")
    logger.info("=" * 50)
    
    # Load DPO data
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    
    model_name = base_model or config.data_generator.student_model
    
    dpo_config = DPOTrainerConfig(
        model_name=model_name,
        beta=config.training.dpo.beta,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        num_epochs=config.training.num_epochs,
        max_length=config.training.dpo.max_length,
        output_dir=f"{config.general.output_dir}/dpo_model",
    )
    
    trainer = ReasoningDPOTrainer(dpo_config)
    trainer.train(data)
    
    return dpo_config.output_dir


def run_grpo_training(config, data_path, base_model=None):
    """Run GRPO training phase."""
    from training.grpo_trainer import GRPOConfig, ReasoningGRPOTrainer
    import json
    
    logger.info("=" * 50)
    logger.info("Phase 2b: GRPO Training")
    logger.info("=" * 50)
    
    # Load data
    data = []
    with open(data_path) as f:
        for line in f:
            data.append(json.loads(line))
    
    model_name = base_model or config.data_generator.student_model
    
    grpo_config = GRPOConfig(
        model_name=model_name,
        learning_rate=config.training.learning_rate,
        batch_size=config.training.batch_size,
        num_epochs=config.training.num_epochs,
        max_length=config.training.dpo.max_length,
        output_dir=f"{config.general.output_dir}/grpo_model",
    )
    
    trainer = ReasoningGRPOTrainer(grpo_config)
    trainer.train(data)
    
    return grpo_config.output_dir


def run_evaluation(config, model_path, args):
    """Run evaluation phase."""
    from evaluation import GSM8KEvaluator, HumanEvalEvaluator
    
    logger.info("=" * 50)
    logger.info("Phase 3: Evaluation")
    logger.info("=" * 50)
    
    results = {}
    
    if args.dataset in ["gsm8k", "math"]:
        evaluator = GSM8KEvaluator(
            model_name=model_path,
            use_test_time_compute=args.use_ttc,
            ttc_samples=config.training.evaluation.num_paths,
        )
        result = evaluator.evaluate(subset_size=args.eval_subset_size)
        results["gsm8k"] = result
        logger.info(f"GSM8K Accuracy: {result.accuracy:.2%}")
    
    if args.dataset in ["humaneval", "mbpp"]:
        evaluator = HumanEvalEvaluator(model_name=model_path)
        result = evaluator.evaluate(subset_size=args.eval_subset_size)
        results["humaneval"] = result
        logger.info(f"HumanEval Pass@1: {result.accuracy:.2%}")
    
    # Save results
    for name, result in results.items():
        result.save(f"{config.general.output_dir}/{name}_results.json")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run full reasoning loop pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "humaneval", "math", "mbpp"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=100,
        help="Number of problems to use",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip data generation phase",
    )
    parser.add_argument(
        "--skip-sft",
        action="store_true",
        default=True,
        help="Skip SFT phase (default: True)",
    )
    parser.add_argument(
        "--run-sft",
        action="store_true",
        help="Run SFT phase (overrides --skip-sft)",
    )
    parser.add_argument(
        "--skip-dpo",
        action="store_true",
        help="Skip DPO training phase",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip evaluation phase",
    )
    parser.add_argument(
        "--use-ttc",
        action="store_true",
        help="Use test-time compute for evaluation",
    )
    parser.add_argument(
        "--eval-subset-size",
        type=int,
        default=None,
        help="Number of problems for evaluation (None for all, overrides --subset-size for eval)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to existing synthetic data (skips generation)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to existing trained model (skips training)",
    )
    parser.add_argument(
        "--training-method",
        type=str,
        default="dpo",
        choices=["dpo", "grpo"],
        help="Training method: dpo or grpo",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = OmegaConf.load(args.config)
    
    # Create output directory
    Path(config.general.output_dir).mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    logger.info("Starting Distributed Reasoning Loop Pipeline")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Config: {args.config}")
    
    # Phase 1: Data Generation
    if not args.skip_generation and not args.data_path:
        samples, pairs = run_data_generation(config, args)
        data_path = f"{config.general.output_dir}/synthetic_data/dpo_pairs.jsonl"
        correct_data_path = f"{config.general.output_dir}/synthetic_data/correct_samples.jsonl"
    else:
        data_path = args.data_path or f"{config.general.output_dir}/synthetic_data/dpo_pairs.jsonl"
        correct_data_path = args.data_path.replace("dpo_pairs", "correct_samples") if args.data_path else None
    
    # Phase 2a: SFT (optional, disabled by default)
    sft_model = None
    if args.run_sft and correct_data_path and Path(correct_data_path).exists():
        sft_model = run_sft_training(config, correct_data_path)
    
    # Phase 2b: DPO/GRPO Training
    if not args.skip_dpo and not args.model_path:
        if args.training_method == "grpo":
            model_path = run_grpo_training(config, data_path, sft_model)
        else:
            model_path = run_dpo_training(config, data_path, sft_model)
    else:
        model_path = args.model_path or config.data_generator.student_model
    
    # Phase 3: Evaluation
    if not args.skip_eval:
        results = run_evaluation(config, model_path, args)
    
    elapsed = time.time() - start_time
    logger.info("=" * 50)
    logger.info(f"Pipeline complete! Total time: {elapsed/60:.1f} minutes")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()