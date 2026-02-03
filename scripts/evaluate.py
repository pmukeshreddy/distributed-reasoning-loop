#!/usr/bin/env python3
"""
Script to evaluate reasoning models on benchmarks.
Supports GSM8K, HumanEval, and MATH benchmarks.
"""

import argparse
import logging
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation import (
    GSM8KEvaluator,
    HumanEvalEvaluator,
    MATHEvaluator,
    run_all_benchmarks,
)
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate reasoning model on benchmarks")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to evaluate",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="all",
        choices=["gsm8k", "humaneval", "math", "all"],
        help="Benchmark to run",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Number of problems to evaluate (None for all)",
    )
    parser.add_argument(
        "--use-ttc",
        action="store_true",
        help="Use test-time compute (Best-of-N sampling)",
    )
    parser.add_argument(
        "--ttc-samples",
        type=int,
        default=16,
        help="Number of samples for test-time compute",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Evaluating model: {args.model}")
    logger.info(f"Benchmark: {args.benchmark}")
    logger.info(f"Test-time compute: {args.use_ttc}")
    
    if args.benchmark == "all":
        results = run_all_benchmarks(
            model_name=args.model,
            output_dir=args.output_dir,
            use_ttc=args.use_ttc,
            gsm8k_subset_size=args.subset_size,
            humaneval_subset_size=args.subset_size,
            ttc_samples=args.ttc_samples,
        )
        
        for name, result in results.items():
            logger.info(f"{name}: {result.accuracy:.2%} accuracy")
    
    elif args.benchmark == "gsm8k":
        evaluator = GSM8KEvaluator(
            model_name=args.model,
            use_test_time_compute=args.use_ttc,
            ttc_samples=args.ttc_samples,
        )
        result = evaluator.evaluate(
            split=args.split,
            subset_size=args.subset_size,
        )
        result.save(f"{args.output_dir}/gsm8k_results.json")
        
        logger.info(f"GSM8K Results:")
        logger.info(f"  Accuracy: {result.accuracy:.2%}")
        logger.info(f"  Correct: {result.correct}/{result.total_problems}")
        logger.info(f"  Avg time: {result.avg_time_per_problem:.2f}s")
    
    elif args.benchmark == "humaneval":
        evaluator = HumanEvalEvaluator(
            model_name=args.model,
        )
        result = evaluator.evaluate(subset_size=args.subset_size)
        result.save(f"{args.output_dir}/humaneval_results.json")
        
        logger.info(f"HumanEval Results:")
        logger.info(f"  Pass@1: {result.accuracy:.2%}")
        logger.info(f"  Correct: {result.correct}/{result.total_problems}")
    
    elif args.benchmark == "math":
        evaluator = MATHEvaluator(
            model_name=args.model,
            use_test_time_compute=args.use_ttc,
            ttc_samples=args.ttc_samples,
        )
        result = evaluator.evaluate(
            split=args.split,
            subset_size=args.subset_size,
        )
        result.save(f"{args.output_dir}/math_results.json")
        
        logger.info(f"MATH Results:")
        logger.info(f"  Accuracy: {result.accuracy:.2%}")
        logger.info(f"  Correct: {result.correct}/{result.total_problems}")
    
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
