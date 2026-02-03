#!/usr/bin/env python3
"""
Continuous Distributed GRPO Training Pipeline.

This script implements the full continuous training loop:
1. Parallel rollout generation across multiple SGLang workers
2. Ray-based verification
3. GRPO training step
4. LoRA broadcast to all workers
5. Repeat

Usage:
    # First, start SGLang workers (see start_workers.sh)
    
    # Then run continuous training:
    python scripts/run_continuous_grpo.py --iterations 100

    # Or with custom ports:
    python scripts/run_continuous_grpo.py --ports 30001 30002 30003
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import logging
import json
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

import ray
from omegaconf import OmegaConf
from tqdm import tqdm

# Local imports
from data_generator.dataset_loader import get_loader
from orchestration.worker_pool import WorkerPoolConfig, SyncWorkerPool
from orchestration.coordinator import TrainingCoordinator, CoordinatorConfig, LoRACheckpoint
from training.grpo_trainer import GRPOConfig, ReasoningGRPOTrainer
from verifier import GSM8KVerifier, VerificationStatus

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ContinuousGRPOPipeline:
    """
    Complete continuous distributed GRPO training pipeline.
    
    Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │              CONTINUOUS DISTRIBUTED GRPO                         │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │  STEP 1: Rollout (Parallel across workers)                      │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
    │  │ SGLang 0 │  │ SGLang 1 │  │ SGLang 2 │  ← All use LoRA vN   │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘                       │
    │       └─────────────┼─────────────┘                              │
    │                     ↓                                            │
    │  STEP 2: Verify (Ray workers)                                   │
    │              ┌──────────────┐                                   │
    │              │ Ray Verify   │  ← SymPy check                    │
    │              └──────┬───────┘                                   │
    │                     ↓                                            │
    │  STEP 3: Train (GRPO on verified batch)                         │
    │              ┌──────────────┐                                   │
    │              │ GRPO Trainer │  ← Compute advantages, update     │
    │              └──────┬───────┘                                   │
    │                     ↓                                            │
    │  STEP 4: Broadcast LoRA vN+1 to ALL workers                     │
    │              ┌──────────────┐                                   │
    │              │ Coordinator  │  ← Hot-reload without restart     │
    │              └──────┬───────┘                                   │
    │                     ↓                                            │
    │            REPEAT FROM STEP 1                                    │
    └─────────────────────────────────────────────────────────────────┘
    """
    
    MATH_SYSTEM_PROMPT = """You are a helpful math tutor. Solve the following problem step by step.
Show your work clearly, explaining each step of your reasoning.
At the end, provide your final answer after '#### '."""
    
    def __init__(
        self,
        worker_ports: List[int],
        model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
        output_dir: str = "./outputs/continuous_grpo",
        num_verification_workers: int = 4,
    ):
        self.worker_ports = worker_ports
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize coordinator
        coordinator_config = CoordinatorConfig(
            worker_ports=worker_ports,
            worker_host="127.0.0.1",
            lora_checkpoint_dir=str(self.output_dir / "lora_checkpoints"),
            num_verification_workers=num_verification_workers,
            log_dir=str(self.output_dir / "logs"),
        )
        self.coordinator = TrainingCoordinator(coordinator_config)
        
        # Initialize GRPO trainer (lazy - only setup when needed)
        grpo_config = GRPOConfig(
            model_name=model_name,
            output_dir=str(self.output_dir / "grpo_model"),
            lora_checkpoint_dir=str(self.output_dir / "lora_checkpoints"),
            save_lora_only=True,  # For hot-reload
            continuous_mode=True,
            num_epochs=1,
            batch_size=2,
            gradient_accumulation_steps=4,
        )
        self.grpo_trainer = ReasoningGRPOTrainer(grpo_config)
        self._trainer_initialized = False
        
        # Initialize verifier
        self.verifier = GSM8KVerifier()
        
        # Load problems
        self.dataset_loader = get_loader("gsm8k")
        self.problems = None
        
        # Metrics tracking
        self.iteration = 0
        self.metrics_history = []
        
    def initialize(self) -> bool:
        """Initialize all components."""
        logger.info("=" * 60)
        logger.info("INITIALIZING CONTINUOUS GRPO PIPELINE")
        logger.info("=" * 60)
        
        # Initialize coordinator and workers
        if not self.coordinator.initialize():
            logger.error("Failed to initialize coordinator")
            return False
        
        # Load problems
        logger.info("Loading GSM8K problems...")
        problems_raw = self.dataset_loader.load()
        self.problems = [
            {"id": p.id, "problem": p.problem, "answer": p.answer}
            for p in problems_raw
        ]
        logger.info(f"Loaded {len(self.problems)} problems")
        
        # Initialize Ray for verification
        ray.init(ignore_reinit_error=True)
        logger.info(f"Ray initialized: {ray.cluster_resources()}")
        
        return True
    
    def _init_trainer(self):
        """Lazy initialization of trainer."""
        if not self._trainer_initialized:
            logger.info("Initializing GRPO trainer...")
            self.grpo_trainer.setup()
            self._trainer_initialized = True
    
    def generate_rollouts(
        self,
        num_problems: int = 50,
        paths_per_problem: int = 10,
    ) -> List[Dict[str, Any]]:
        """Generate rollouts using the worker pool."""
        # Sample problems
        import random
        problems = random.sample(self.problems, min(num_problems, len(self.problems)))
        
        logger.info(f"Generating {num_problems * paths_per_problem} rollouts...")
        
        rollouts = self.coordinator.generate_rollouts(
            problems=problems,
            system_prompt=self.MATH_SYSTEM_PROMPT,
            paths_per_problem=paths_per_problem,
        )
        
        return rollouts
    
    def verify_rollouts(
        self,
        rollouts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Verify rollouts using SymPy verification."""
        logger.info(f"Verifying {len(rollouts)} rollouts...")
        
        verified = []
        for rollout in tqdm(rollouts, desc="Verifying"):
            try:
                result = self.verifier.verify_reasoning_path(
                    rollout["reasoning"],
                    rollout["expected_answer"],
                )
                rollout["is_correct"] = result.status == VerificationStatus.CORRECT
                rollout["verification_confidence"] = result.confidence
                rollout["final_answer"] = result.predicted
            except Exception as e:
                logger.debug(f"Verification error: {e}")
                rollout["is_correct"] = False
                rollout["verification_confidence"] = 0.0
            
            verified.append(rollout)
        
        correct = sum(1 for r in verified if r.get("is_correct", False))
        logger.info(f"Verification: {correct}/{len(verified)} correct ({100*correct/len(verified):.1f}%)")
        
        return verified
    
    def train_step(
        self,
        verified_rollouts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Perform a GRPO training step."""
        self._init_trainer()
        
        # Convert to training format
        training_data = []
        for rollout in verified_rollouts:
            training_data.append({
                "prompt": rollout["problem"],
                "reasoning": rollout["reasoning"],
                "is_correct": rollout.get("is_correct", False),
            })
        
        logger.info(f"Training on {len(training_data)} samples...")
        
        # Train step
        loss = self.grpo_trainer.train_step(training_data)
        
        # Save LoRA checkpoint
        checkpoint_path = self.grpo_trainer.save_lora_only()
        
        return {
            "loss": loss,
            "samples": len(training_data),
            "checkpoint_path": checkpoint_path,
        }
    
    def broadcast_lora(self, checkpoint_path: str) -> bool:
        """Broadcast new LoRA to all workers."""
        checkpoint = self.coordinator.save_lora_checkpoint(
            lora_path=checkpoint_path,
            metrics={"iteration": self.iteration},
        )
        
        results = self.coordinator.broadcast_lora_update(checkpoint)
        success = all(results.values())
        
        if success:
            logger.info(f"All workers updated to LoRA v{checkpoint.version}")
        else:
            failed = [k for k, v in results.items() if not v]
            logger.warning(f"Workers {failed} failed to update")
        
        return success
    
    def run_iteration(self) -> Dict[str, Any]:
        """Run a single training iteration."""
        self.iteration += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {self.iteration}")
        logger.info(f"{'='*60}")
        
        start_time = time.time()
        
        # Step 1: Generate rollouts
        rollout_start = time.time()
        rollouts = self.generate_rollouts(
            num_problems=50,
            paths_per_problem=10,
        )
        rollout_time = time.time() - rollout_start
        
        # Step 2: Verify
        verify_start = time.time()
        verified = self.verify_rollouts(rollouts)
        verify_time = time.time() - verify_start
        
        # Step 3: Train
        train_start = time.time()
        train_result = self.train_step(verified)
        train_time = time.time() - train_start
        
        # Step 4: Broadcast LoRA
        broadcast_start = time.time()
        if train_result.get("checkpoint_path"):
            self.broadcast_lora(train_result["checkpoint_path"])
        broadcast_time = time.time() - broadcast_start
        
        total_time = time.time() - start_time
        
        # Compute metrics
        correct = sum(1 for r in verified if r.get("is_correct", False))
        accuracy = correct / len(verified) if verified else 0
        
        metrics = {
            "iteration": self.iteration,
            "timestamp": datetime.now().isoformat(),
            "lora_version": self.coordinator.current_lora_version,
            "rollouts": {
                "total": len(rollouts),
                "time_seconds": rollout_time,
            },
            "verification": {
                "total": len(verified),
                "correct": correct,
                "accuracy": accuracy,
                "time_seconds": verify_time,
            },
            "training": {
                "loss": train_result.get("loss", 0),
                "samples": train_result.get("samples", 0),
                "time_seconds": train_time,
            },
            "broadcast": {
                "time_seconds": broadcast_time,
            },
            "total_time_seconds": total_time,
        }
        
        self.metrics_history.append(metrics)
        self._save_metrics(metrics)
        
        # Print summary
        logger.info(f"\nIteration {self.iteration} Summary:")
        logger.info(f"  Rollouts: {len(rollouts)} in {rollout_time:.1f}s")
        logger.info(f"  Accuracy: {accuracy:.1%} ({correct}/{len(verified)})")
        logger.info(f"  Loss: {train_result.get('loss', 0):.4f}")
        logger.info(f"  LoRA version: v{self.coordinator.current_lora_version}")
        logger.info(f"  Total time: {total_time:.1f}s")
        
        return metrics
    
    def _save_metrics(self, metrics: Dict[str, Any]):
        """Save metrics to file."""
        metrics_file = self.output_dir / "training_metrics.jsonl"
        with open(metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
    
    def run(
        self,
        num_iterations: int = 10,
        target_accuracy: Optional[float] = None,
    ):
        """
        Run the continuous training loop.
        
        Args:
            num_iterations: Number of iterations to run
            target_accuracy: Stop early if accuracy exceeds this
        """
        logger.info(f"\n{'#'*60}")
        logger.info(f"STARTING CONTINUOUS GRPO TRAINING")
        logger.info(f"  Iterations: {num_iterations}")
        logger.info(f"  Workers: {len(self.worker_ports)}")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"{'#'*60}\n")
        
        try:
            for i in range(num_iterations):
                metrics = self.run_iteration()
                
                # Check for early stopping
                accuracy = metrics["verification"]["accuracy"]
                if target_accuracy and accuracy >= target_accuracy:
                    logger.info(f"\n🎉 Target accuracy {target_accuracy:.1%} reached!")
                    break
                
        except KeyboardInterrupt:
            logger.info("\nTraining interrupted by user")
        
        # Final summary
        self._print_final_summary()
    
    def _print_final_summary(self):
        """Print final training summary."""
        if not self.metrics_history:
            return
        
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING COMPLETE")
        logger.info(f"{'='*60}")
        
        # Compute summary stats
        accuracies = [m["verification"]["accuracy"] for m in self.metrics_history]
        losses = [m["training"]["loss"] for m in self.metrics_history]
        
        logger.info(f"Total iterations: {len(self.metrics_history)}")
        logger.info(f"Final LoRA version: v{self.coordinator.current_lora_version}")
        logger.info(f"Final accuracy: {accuracies[-1]:.1%}")
        logger.info(f"Best accuracy: {max(accuracies):.1%}")
        logger.info(f"Final loss: {losses[-1]:.4f}")
        
        # Save final summary
        summary = {
            "total_iterations": len(self.metrics_history),
            "final_lora_version": self.coordinator.current_lora_version,
            "final_accuracy": accuracies[-1],
            "best_accuracy": max(accuracies),
            "final_loss": losses[-1],
            "worker_stats": self.coordinator.worker_pool.get_stats(),
        }
        
        with open(self.output_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"\nResults saved to: {self.output_dir}")
    
    def shutdown(self):
        """Shutdown all components."""
        self.coordinator.shutdown()
        ray.shutdown()
        logger.info("Pipeline shutdown complete")


def main():
    parser = argparse.ArgumentParser(
        description="Continuous Distributed GRPO Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default workers on ports 30001-30003:
    python scripts/run_continuous_grpo.py --iterations 50

    # With custom worker ports:
    python scripts/run_continuous_grpo.py --ports 30001 30002 --iterations 100

    # With target accuracy for early stopping:
    python scripts/run_continuous_grpo.py --iterations 100 --target-accuracy 0.9
        """,
    )
    
    parser.add_argument(
        "--ports",
        type=int,
        nargs="+",
        default=[30001, 30002, 30003],
        help="SGLang worker ports (default: 30001 30002 30003)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of training iterations (default: 10)",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=None,
        help="Stop early if accuracy exceeds this value",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs/continuous_grpo",
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    # Create and run pipeline
    pipeline = ContinuousGRPOPipeline(
        worker_ports=args.ports,
        model_name=args.model,
        output_dir=args.output_dir,
    )
    
    try:
        if not pipeline.initialize():
            logger.error("Failed to initialize pipeline")
            sys.exit(1)
        
        pipeline.run(
            num_iterations=args.iterations,
            target_accuracy=args.target_accuracy,
        )
    finally:
        pipeline.shutdown()


if __name__ == "__main__":
    main()
