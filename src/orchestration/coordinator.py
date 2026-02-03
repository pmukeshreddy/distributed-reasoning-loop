"""
Coordinator for Continuous Distributed GRPO Training.

Orchestrates the continuous loop:
1. Rollout generation (parallel across workers)
2. Verification (Ray workers)
3. GRPO Training
4. LoRA broadcast to all workers
5. Repeat
"""

import asyncio
import logging
import time
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
import shutil

from .worker_pool import (
    SGLangWorkerPool,
    SyncWorkerPool,
    WorkerPoolConfig,
    WorkerInfo,
    WorkerStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class LoRACheckpoint:
    """Information about a LoRA checkpoint."""
    version: int
    path: str
    created_at: float
    training_step: int
    metrics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "path": self.path,
            "created_at": self.created_at,
            "training_step": self.training_step,
            "metrics": self.metrics,
        }


@dataclass
class CoordinatorConfig:
    """Configuration for the training coordinator."""
    # Worker pool settings
    worker_ports: List[int] = field(default_factory=lambda: [30001, 30002, 30003])
    worker_host: str = "127.0.0.1"
    worker_gpu_ids: Optional[List[int]] = None
    
    # LoRA management
    lora_checkpoint_dir: str = "./checkpoints/lora"
    max_lora_versions: int = 5  # Keep last N versions
    
    # Training loop
    rollouts_per_iteration: int = 500
    paths_per_problem: int = 10
    batch_size: int = 10
    
    # Verification
    num_verification_workers: int = 4
    
    # Timeouts
    health_check_interval: float = 30.0
    broadcast_timeout: float = 60.0
    
    # Logging
    log_dir: str = "./logs/coordinator"


class TrainingCoordinator:
    """
    Coordinates continuous distributed GRPO training.
    
    Manages the loop:
    - Parallel rollout generation across SGLang workers
    - Ray verification
    - GRPO training step
    - LoRA broadcast to all workers
    """
    
    def __init__(self, config: CoordinatorConfig):
        self.config = config
        
        # Initialize worker pool
        pool_config = WorkerPoolConfig.from_ports(
            ports=config.worker_ports,
            gpu_ids=config.worker_gpu_ids,
            host=config.worker_host,
        )
        self.worker_pool = SyncWorkerPool(pool_config)
        
        # LoRA version tracking
        self.current_lora_version = 0
        self.lora_checkpoints: List[LoRACheckpoint] = []
        
        # Training state
        self.training_step = 0
        self.iteration = 0
        self.is_running = False
        
        # Metrics
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Setup directories
        self.checkpoint_dir = Path(config.lora_checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
    def initialize(self) -> bool:
        """Initialize coordinator and worker pool."""
        logger.info("Initializing training coordinator...")
        
        self.worker_pool.initialize()
        health = self.worker_pool.check_health()
        
        ready_count = sum(1 for h in health.values() if h)
        logger.info(f"Workers ready: {ready_count}/{len(health)}")
        
        if ready_count == 0:
            logger.error("No workers available!")
            return False
        
        return True
    
    def save_lora_checkpoint(
        self,
        lora_path: str,
        metrics: Optional[Dict[str, float]] = None,
    ) -> LoRACheckpoint:
        """
        Save a LoRA checkpoint with versioning.
        
        Args:
            lora_path: Path to the trained LoRA adapter
            metrics: Training metrics for this checkpoint
            
        Returns:
            LoRACheckpoint with version info
        """
        self.current_lora_version += 1
        version = self.current_lora_version
        
        # Copy to versioned directory
        versioned_path = self.checkpoint_dir / f"lora_v{version}"
        if Path(lora_path).exists():
            if versioned_path.exists():
                shutil.rmtree(versioned_path)
            shutil.copytree(lora_path, versioned_path)
        
        checkpoint = LoRACheckpoint(
            version=version,
            path=str(versioned_path),
            created_at=time.time(),
            training_step=self.training_step,
            metrics=metrics or {},
        )
        
        self.lora_checkpoints.append(checkpoint)
        
        # Cleanup old versions
        self._cleanup_old_checkpoints()
        
        # Save checkpoint metadata
        self._save_checkpoint_metadata()
        
        logger.info(f"Saved LoRA checkpoint v{version} at step {self.training_step}")
        return checkpoint
    
    def _cleanup_old_checkpoints(self):
        """Remove old LoRA checkpoints beyond max_versions."""
        while len(self.lora_checkpoints) > self.config.max_lora_versions:
            old = self.lora_checkpoints.pop(0)
            old_path = Path(old.path)
            if old_path.exists():
                shutil.rmtree(old_path)
                logger.info(f"Removed old checkpoint v{old.version}")
    
    def _save_checkpoint_metadata(self):
        """Save checkpoint metadata to disk."""
        metadata = {
            "current_version": self.current_lora_version,
            "checkpoints": [c.to_dict() for c in self.lora_checkpoints],
        }
        with open(self.checkpoint_dir / "checkpoints.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def broadcast_lora_update(
        self,
        checkpoint: LoRACheckpoint,
        wait_for_all: bool = True,
    ) -> Dict[int, bool]:
        """
        Broadcast LoRA update to all workers.
        
        Args:
            checkpoint: LoRA checkpoint to load
            wait_for_all: Whether to wait for all workers to reload
            
        Returns:
            Dict mapping worker_id to success status
        """
        logger.info(f"Broadcasting LoRA v{checkpoint.version} to all workers...")
        start_time = time.time()
        
        results = self.worker_pool.reload_lora_all(
            lora_path=checkpoint.path,
            version=checkpoint.version,
        )
        
        elapsed = time.time() - start_time
        success_count = sum(1 for r in results.values() if r)
        
        logger.info(
            f"LoRA broadcast complete: {success_count}/{len(results)} workers "
            f"updated in {elapsed:.2f}s"
        )
        
        # Log any failures
        for worker_id, success in results.items():
            if not success:
                logger.warning(f"Worker {worker_id} failed to reload LoRA")
        
        return results
    
    def broadcast_merged_weights(
        self,
        merged_path: str,
        version: int,
    ) -> Dict[int, bool]:
        """
        Broadcast merged model weights to all workers using /update_weights_from_disk.
        
        This is the correct approach for SGLang - it expects merged weights,
        not raw PEFT LoRA adapters.
        
        Args:
            merged_path: Path to merged model checkpoint
            version: Version number for tracking
            
        Returns:
            Dict mapping worker_id to success status
        """
        logger.info(f"Broadcasting merged weights v{version} to all workers...")
        start_time = time.time()
        
        results = self.worker_pool.update_weights_all(
            model_path=merged_path,
            version=version,
        )
        
        elapsed = time.time() - start_time
        success_count = sum(1 for r in results.values() if r)
        
        logger.info(
            f"Weight update complete: {success_count}/{len(results)} workers "
            f"updated in {elapsed:.2f}s"
        )
        
        # Log any failures
        for worker_id, success in results.items():
            if not success:
                logger.warning(f"Worker {worker_id} failed to reload LoRA")
        
        return results
    
    def verify_all_workers_synced(self, expected_version: int) -> bool:
        """Verify all workers are on the expected LoRA version."""
        for worker in self.worker_pool.workers:
            if worker.current_lora_version != expected_version:
                logger.warning(
                    f"Worker {worker.worker_id} on v{worker.current_lora_version}, "
                    f"expected v{expected_version}"
                )
                return False
        return True
    
    def generate_rollouts(
        self,
        problems: List[Dict[str, Any]],
        system_prompt: str,
        paths_per_problem: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate rollouts using the worker pool.
        
        Args:
            problems: List of problems with 'id' and 'problem' keys
            system_prompt: System prompt for generation
            paths_per_problem: Number of paths per problem
            
        Returns:
            List of rollouts with problem info and generated reasoning
        """
        paths_per_problem = paths_per_problem or self.config.paths_per_problem
        
        # Build all messages
        all_messages = []
        message_to_problem = []
        
        for prob in problems:
            for _ in range(paths_per_problem):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prob["problem"]},
                ]
                all_messages.append(messages)
                message_to_problem.append(prob)
        
        logger.info(f"Generating {len(all_messages)} rollouts across {len(self.worker_pool.workers)} workers")
        
        # Generate with thread pool for parallelism
        results = self.worker_pool.generate_batch_threaded(
            all_messages,
            max_workers=32,
            temperature=0.8,
            max_tokens=2048,
        )
        
        # Collect rollouts
        rollouts = []
        for (text, metadata), prob in zip(results, message_to_problem):
            if text is not None:
                rollout = {
                    "problem_id": prob["id"],
                    "problem": prob["problem"],
                    "expected_answer": prob.get("answer", ""),
                    "reasoning": text,
                    "policy_version": metadata.get("lora_version", 0),
                    "worker_id": metadata.get("worker_id", -1),
                }
                rollouts.append(rollout)
        
        logger.info(f"Generated {len(rollouts)} successful rollouts")
        return rollouts
    
    def log_metrics(
        self,
        iteration: int,
        rollout_metrics: Dict[str, Any],
        verification_metrics: Dict[str, Any],
        training_metrics: Dict[str, Any],
    ):
        """Log metrics for this iteration."""
        metrics = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "lora_version": self.current_lora_version,
            "rollouts": rollout_metrics,
            "verification": verification_metrics,
            "training": training_metrics,
            "worker_stats": self.worker_pool.get_stats(),
        }
        
        self.metrics_history.append(metrics)
        
        # Save to file
        with open(self.log_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
        
        # Log summary
        logger.info(
            f"Iteration {iteration}: "
            f"rollouts={rollout_metrics.get('total', 0)}, "
            f"correct={verification_metrics.get('correct', 0)}, "
            f"loss={training_metrics.get('loss', 0):.4f}"
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current coordinator status."""
        return {
            "is_running": self.is_running,
            "iteration": self.iteration,
            "training_step": self.training_step,
            "lora_version": self.current_lora_version,
            "workers": self.worker_pool.get_stats(),
            "checkpoints": [c.to_dict() for c in self.lora_checkpoints],
        }
    
    def shutdown(self):
        """Shutdown coordinator."""
        self.is_running = False
        self.worker_pool.close()
        logger.info("Coordinator shutdown complete")


class ContinuousTrainingLoop:
    """
    Implements the continuous GRPO training loop.
    
    Loop:
    1. Generate rollouts (parallel workers)
    2. Verify with Ray
    3. Train GRPO step
    4. Save and broadcast new LoRA
    5. Repeat
    """
    
    def __init__(
        self,
        coordinator: TrainingCoordinator,
        grpo_trainer,  # ReasoningGRPOTrainer instance
        verifier,  # Verifier instance
        problem_loader: Callable[[], List[Dict[str, Any]]],
    ):
        self.coordinator = coordinator
        self.grpo_trainer = grpo_trainer
        self.verifier = verifier
        self.problem_loader = problem_loader
        
        # System prompt for math reasoning
        self.system_prompt = """You are a helpful math tutor. Solve the following problem step by step.
Show your work clearly, explaining each step of your reasoning.
At the end, provide your final answer after '#### '."""
    
    def run_iteration(self) -> Dict[str, Any]:
        """Run a single training iteration."""
        iteration = self.coordinator.iteration
        logger.info(f"=== Starting iteration {iteration} ===")
        
        # Step 1: Load problems
        problems = self.problem_loader()
        problems = problems[:self.coordinator.config.rollouts_per_iteration]
        
        # Step 2: Generate rollouts
        start_time = time.time()
        rollouts = self.coordinator.generate_rollouts(
            problems=problems,
            system_prompt=self.system_prompt,
        )
        rollout_time = time.time() - start_time
        
        rollout_metrics = {
            "total": len(rollouts),
            "time_seconds": rollout_time,
            "rollouts_per_second": len(rollouts) / rollout_time if rollout_time > 0 else 0,
        }
        
        # Step 3: Verify rollouts
        start_time = time.time()
        verified_rollouts = self._verify_rollouts(rollouts)
        verify_time = time.time() - start_time
        
        correct = sum(1 for r in verified_rollouts if r.get("is_correct", False))
        verification_metrics = {
            "total": len(verified_rollouts),
            "correct": correct,
            "incorrect": len(verified_rollouts) - correct,
            "accuracy": correct / len(verified_rollouts) if verified_rollouts else 0,
            "time_seconds": verify_time,
        }
        
        # Step 4: Train GRPO
        start_time = time.time()
        training_result = self._train_step(verified_rollouts)
        train_time = time.time() - start_time
        
        training_metrics = {
            "loss": training_result.get("loss", 0),
            "time_seconds": train_time,
            **training_result,
        }
        
        # Step 5: Save checkpoint and broadcast
        if training_result.get("checkpoint_path"):
            checkpoint = self.coordinator.save_lora_checkpoint(
                lora_path=training_result["checkpoint_path"],
                metrics=training_metrics,
            )
            self.coordinator.broadcast_lora_update(checkpoint)
        
        # Log metrics
        self.coordinator.log_metrics(
            iteration=iteration,
            rollout_metrics=rollout_metrics,
            verification_metrics=verification_metrics,
            training_metrics=training_metrics,
        )
        
        self.coordinator.iteration += 1
        
        return {
            "iteration": iteration,
            "rollouts": rollout_metrics,
            "verification": verification_metrics,
            "training": training_metrics,
        }
    
    def _verify_rollouts(
        self,
        rollouts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Verify rollouts using the verifier."""
        from src.verifier import VerificationStatus
        
        for rollout in rollouts:
            try:
                result = self.verifier.verify_reasoning_path(
                    rollout["reasoning"],
                    rollout["expected_answer"],
                )
                rollout["is_correct"] = result.status == VerificationStatus.CORRECT
                rollout["verification_confidence"] = result.confidence
                rollout["final_answer"] = result.predicted
            except Exception as e:
                logger.warning(f"Verification error: {e}")
                rollout["is_correct"] = False
                rollout["verification_confidence"] = 0.0
        
        return rollouts
    
    def _train_step(
        self,
        verified_rollouts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Perform a GRPO training step."""
        # Convert to training format
        training_data = []
        for rollout in verified_rollouts:
            training_data.append({
                "prompt": rollout["problem"],
                "reasoning": rollout["reasoning"],
                "is_correct": rollout.get("is_correct", False),
            })
        
        # Train (single step/epoch)
        try:
            # This would call the trainer's train method
            # For now, return placeholder metrics
            loss = self.grpo_trainer.train_step(training_data)
            
            checkpoint_path = self.grpo_trainer.save_lora_only()
            
            return {
                "loss": loss,
                "samples": len(training_data),
                "checkpoint_path": checkpoint_path,
            }
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {"loss": 0, "error": str(e)}
    
    def run(
        self,
        num_iterations: int,
        early_stop_accuracy: Optional[float] = None,
    ):
        """
        Run the continuous training loop.
        
        Args:
            num_iterations: Number of iterations to run
            early_stop_accuracy: Stop if verification accuracy exceeds this
        """
        self.coordinator.is_running = True
        logger.info(f"Starting continuous training for {num_iterations} iterations")
        
        try:
            for i in range(num_iterations):
                if not self.coordinator.is_running:
                    logger.info("Training stopped by user")
                    break
                
                result = self.run_iteration()
                
                # Early stopping check
                accuracy = result["verification"]["accuracy"]
                if early_stop_accuracy and accuracy >= early_stop_accuracy:
                    logger.info(
                        f"Early stopping: accuracy {accuracy:.2%} >= {early_stop_accuracy:.2%}"
                    )
                    break
                
        except KeyboardInterrupt:
            logger.info("Training interrupted")
        finally:
            self.coordinator.is_running = False
        
        logger.info(f"Training complete after {self.coordinator.iteration} iterations")
        return self.coordinator.metrics_history
