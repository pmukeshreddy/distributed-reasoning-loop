#!/usr/bin/env python3
"""
Script to generate synthetic reasoning data using the pipeline.
Generates Chain-of-Thought solutions and creates DPO training pairs.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generator import SyntheticDataPipeline, GenerationConfig
from omegaconf import OmegaConf

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic reasoning data")
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
        "--model",
        type=str,
        default=None,
        help="Model to use for generation (overrides config)",
    )
    parser.add_argument(
        "--num-paths",
        type=int,
        default=10,
        help="Number of reasoning paths per problem",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Number of problems to process (None for all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./synthetic_data",
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "sglang", "transformers"],
        help="Inference backend to use",
    )
    
    args = parser.parse_args()
    
    # Load config
    if Path(args.config).exists():
        config = OmegaConf.load(args.config)
    else:
        config = OmegaConf.create({})
    
    # Override with command line args
    model_name = args.model or config.get("data_generator", {}).get(
        "teacher_model", "meta-llama/Llama-3-70B-Instruct"
    )
    
    # Create generation config
    from data_generator.cot_generator import InferenceBackend
    
    backend_map = {
        "vllm": InferenceBackend.VLLM,
        "sglang": InferenceBackend.SGLANG,
        "transformers": InferenceBackend.TRANSFORMERS,
    }
    
    gen_config = GenerationConfig(
        model_name=model_name,
        backend=backend_map[args.backend],
        num_paths=args.num_paths,
        max_new_tokens=config.get("data_generator", {}).get("max_new_tokens", 2048),
        temperature=config.get("data_generator", {}).get("temperature", 0.8),
    )
    
    # Create pipeline
    pipeline = SyntheticDataPipeline(
        generator_config=gen_config,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
    )
    
    # Run pipeline
    logger.info(f"Starting synthetic data generation for {args.dataset}")
    logger.info(f"Model: {model_name}")
    logger.info(f"Paths per problem: {args.num_paths}")
    logger.info(f"Output directory: {args.output_dir}")
    
    samples, pairs = pipeline.run(
        subset_size=args.subset_size,
        batch_size=args.batch_size,
    )
    
    logger.info(f"Generation complete!")
    logger.info(f"Total samples: {len(samples)}")
    logger.info(f"Correct samples: {sum(1 for s in samples if s.is_correct)}")
    logger.info(f"DPO pairs: {len(pairs)}")


if __name__ == "__main__":
    main()
