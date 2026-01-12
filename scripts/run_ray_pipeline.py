#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import ray
import logging
from omegaconf import OmegaConf
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import os
os.environ["PYTHONPATH"] = "/workspace/distributed-reasoning-loop/src"


ray.init(ignore_reinit_error=True)
logger.info(f"Ray: {ray.cluster_resources()}")

from data_generator import SyntheticDataPipeline, GenerationConfig
from data_generator.cot_generator import InferenceBackend
from orchestration.ray_workers import DistributedDataProcessor, RayClusterConfig
from training.grpo_trainer import GRPOConfig, ReasoningGRPOTrainer

config = OmegaConf.load("config/default.yaml")

logger.info("Phase 1: SGLang Generation")
gen_config = GenerationConfig(
    model_name="Qwen/Qwen2.5-1.5B-Instruct",
    backend=InferenceBackend.SGLANG,
    num_paths=10,
    max_new_tokens=2048,
    temperature=0.8,
)
pipeline = SyntheticDataPipeline(gen_config, "gsm8k", "./outputs/synthetic_data")
samples, pairs = pipeline.run(subset_size=500, batch_size=10)

logger.info("Phase 2: Ray Verification")
ray_config = RayClusterConfig(num_workers=4)
processor = DistributedDataProcessor(ray_config, "math")
processor.initialize()
verified = processor.process_data([s.to_dict() for s in samples], verify=True, tokenize=False)
logger.info(f"Ray stats: {processor.get_stats()}")
processor.shutdown()

logger.info("Phase 3: GRPO Training")
data = [json.loads(l) for l in open("./outputs/synthetic_data/dpo_pairs.jsonl")]
trainer = ReasoningGRPOTrainer(GRPOConfig(model_name="Qwen/Qwen2.5-1.5B-Instruct", output_dir="./outputs/grpo_model", num_epochs=5, batch_size=2))
trainer.train(data)

ray.shutdown()
logger.info("Done: SGLang -> Ray -> GRPO")