"""
Ray distributed workers for data processing pipeline.
Handles tokenization, batching, and distributed compute.
"""

import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Callable
import time
from queue import Queue
import numpy as np

try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    ray = None
    RAY_AVAILABLE = False

logger = logging.getLogger(__name__)


def _ray_remote_or_class(cls):
    """Decorator that applies @_ray_remote_or_class if ray is available, else returns class as-is."""
    if RAY_AVAILABLE and ray is not None:
        return ray.remote(cls)
    return cls


@dataclass
class RayClusterConfig:
    """Configuration for Ray cluster."""
    num_workers: int = 4
    num_cpus_per_worker: int = 2
    num_gpus_per_worker: float = 0.0
    object_store_memory: int = 4 * 1024 * 1024 * 1024  # 4GB
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8265


class RayClusterManager:
    """
    Manages Ray cluster lifecycle and worker coordination.
    """
    
    def __init__(self, config: RayClusterConfig):
        self.config = config
        self._initialized = False
        
    def initialize(self, address: Optional[str] = None):
        """
        Initialize Ray cluster.
        
        Args:
            address: Ray cluster address (None for local, "auto" for existing)
        """
        if self._initialized:
            return
            
        if address:
            ray.init(
                address=address,
                ignore_reinit_error=True,
            )
        else:
            ray.init(
                num_cpus=self.config.num_workers * self.config.num_cpus_per_worker,
                object_store_memory=self.config.object_store_memory,
                dashboard_host=self.config.dashboard_host,
                dashboard_port=self.config.dashboard_port,
                ignore_reinit_error=True,
            )
        
        self._initialized = True
        logger.info(f"Ray cluster initialized: {ray.cluster_resources()}")
    
    def shutdown(self):
        """Shutdown Ray cluster."""
        if self._initialized:
            ray.shutdown()
            self._initialized = False
    
    def get_cluster_info(self) -> Dict[str, Any]:
        """Get cluster resource information."""
        if not self._initialized:
            return {}
            
        return {
            "resources": ray.cluster_resources(),
            "available": ray.available_resources(),
            "nodes": len(ray.nodes()),
        }


@_ray_remote_or_class
class DataProcessingWorker:
    """
    Ray actor for distributed data processing.
    Handles verification and filtering of reasoning paths.
    """
    
    def __init__(self, worker_id: int, problem_type: str = "math"):
        self.worker_id = worker_id
        self.problem_type = problem_type
        self.processed_count = 0
        self._init_verifier()
    
    def _init_verifier(self):
        """Initialize the appropriate verifier."""
        if self.problem_type == "math":
            try:
                from verifier import GSM8KVerifier
            except ImportError:
                from ..verifier import GSM8KVerifier
            self.verifier = GSM8KVerifier()
        else:
            try:
                from verifier import HumanEvalVerifier
            except ImportError:
                from ..verifier import HumanEvalVerifier
            self.verifier = HumanEvalVerifier()
    
    def process_batch(
        self,
        batch: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Process a batch of reasoning paths.
        
        Args:
            batch: List of reasoning paths to verify
            
        Returns:
            List of verified paths with is_correct flag
        """
        results = []
        
        for item in batch:
            try:
                if self.problem_type == "math":
                    try:
                        from verifier import VerificationStatus
                    except ImportError:
                        from ..verifier import VerificationStatus
                    result = self.verifier.verify_reasoning_path(
                        item["reasoning"],
                        item["expected_answer"],
                    )
                    is_correct = result.status == VerificationStatus.CORRECT
                    
                    item["is_correct"] = is_correct
                    item["verification_confidence"] = result.confidence
                    item["final_answer"] = result.predicted
                else:
                    try:
                        from verifier import ExecutionStatus
                    except ImportError:
                        from ..verifier import ExecutionStatus
                    code = self.verifier.extract_code(item["reasoning"])
                    result = self.verifier.verify_humaneval(
                        code,
                        item.get("entry_point", "solution"),
                        item.get("test", ""),
                    )
                    is_correct = result.status == ExecutionStatus.SUCCESS
                    
                    item["is_correct"] = is_correct
                    item["final_answer"] = code
                
                results.append(item)
                self.processed_count += 1
                
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                item["is_correct"] = False
                item["error"] = str(e)
                results.append(item)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics."""
        return {
            "worker_id": self.worker_id,
            "processed_count": self.processed_count,
        }


@_ray_remote_or_class
class TokenizationWorker:
    """
    Ray actor for distributed tokenization.
    Prepares data for training.
    """
    
    def __init__(self, worker_id: int, model_name: str):
        self.worker_id = worker_id
        self.model_name = model_name
        self.tokenizer = None
        self._init_tokenizer()
    
    def _init_tokenizer(self):
        """Initialize tokenizer."""
        try:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
    
    def tokenize_batch(
        self,
        batch: List[Dict[str, Any]],
        max_length: int = 2048,
    ) -> List[Dict[str, Any]]:
        """
        Tokenize a batch of samples.
        
        Args:
            batch: List of samples with 'prompt' and 'chosen'/'rejected' keys
            max_length: Maximum sequence length
            
        Returns:
            List of tokenized samples
        """
        if self.tokenizer is None:
            return batch
        
        results = []
        
        for item in batch:
            try:
                # For DPO format
                if "chosen" in item and "rejected" in item:
                    prompt_tokens = self.tokenizer(
                        item["prompt"],
                        truncation=True,
                        max_length=max_length // 2,
                    )
                    chosen_tokens = self.tokenizer(
                        item["chosen"],
                        truncation=True,
                        max_length=max_length,
                    )
                    rejected_tokens = self.tokenizer(
                        item["rejected"],
                        truncation=True,
                        max_length=max_length,
                    )
                    
                    item["prompt_input_ids"] = prompt_tokens["input_ids"]
                    item["chosen_input_ids"] = chosen_tokens["input_ids"]
                    item["rejected_input_ids"] = rejected_tokens["input_ids"]
                else:
                    # Standard tokenization
                    tokens = self.tokenizer(
                        item.get("text", item.get("reasoning", "")),
                        truncation=True,
                        max_length=max_length,
                    )
                    item["input_ids"] = tokens["input_ids"]
                    item["attention_mask"] = tokens["attention_mask"]
                
                results.append(item)
                
            except Exception as e:
                logger.error(f"Tokenization error: {e}")
                results.append(item)
        
        return results


@_ray_remote_or_class
class BatchPreparationWorker:
    """
    Ray actor for preparing training batches.
    Handles padding, batching, and data collation.
    """
    
    def __init__(self, worker_id: int, batch_size: int = 8):
        self.worker_id = worker_id
        self.batch_size = batch_size
        self.buffer = []
    
    def add_samples(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Add samples to buffer and return completed batches.
        
        Args:
            samples: List of tokenized samples
            
        Returns:
            List of completed batches
        """
        self.buffer.extend(samples)
        
        batches = []
        while len(self.buffer) >= self.batch_size:
            batch_samples = self.buffer[:self.batch_size]
            self.buffer = self.buffer[self.batch_size:]
            
            batch = self._collate_batch(batch_samples)
            batches.append(batch)
        
        return batches
    
    def _collate_batch(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate samples into a batch."""
        batch = {
            "samples": samples,
            "batch_size": len(samples),
        }
        
        # Check if DPO format
        if samples and "chosen_input_ids" in samples[0]:
            batch["format"] = "dpo"
            batch["prompt_input_ids"] = [s["prompt_input_ids"] for s in samples]
            batch["chosen_input_ids"] = [s["chosen_input_ids"] for s in samples]
            batch["rejected_input_ids"] = [s["rejected_input_ids"] for s in samples]
        elif samples and "input_ids" in samples[0]:
            batch["format"] = "standard"
            batch["input_ids"] = [s["input_ids"] for s in samples]
            batch["attention_mask"] = [s.get("attention_mask", []) for s in samples]
        
        return batch
    
    def flush(self) -> Optional[Dict[str, Any]]:
        """Flush remaining buffer as partial batch."""
        if not self.buffer:
            return None
        
        batch = self._collate_batch(self.buffer)
        self.buffer = []
        return batch


class DistributedDataProcessor:
    """
    Orchestrates distributed data processing using Ray workers.
    """
    
    def __init__(
        self,
        cluster_config: RayClusterConfig,
        problem_type: str = "math",
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    ):
        self.cluster_config = cluster_config
        self.problem_type = problem_type
        self.model_name = model_name
        
        self.cluster_manager = RayClusterManager(cluster_config)
        self.workers = []
        self.tokenizers = []
        self.batch_preparers = []
        
    def initialize(self, ray_address: Optional[str] = None):
        """Initialize cluster and workers."""
        self.cluster_manager.initialize(ray_address)
        
        # Create workers
        num_workers = self.cluster_config.num_workers
        
        self.workers = [
            DataProcessingWorker.remote(i, self.problem_type)
            for i in range(num_workers)
        ]
        
        self.tokenizers = [
            TokenizationWorker.remote(i, self.model_name)
            for i in range(num_workers)
        ]
        
        self.batch_preparers = [
            BatchPreparationWorker.remote(i)
            for i in range(num_workers)
        ]
        
        logger.info(f"Initialized {num_workers} workers of each type")
    
    def process_data(
        self,
        data: List[Dict[str, Any]],
        verify: bool = True,
        tokenize: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Process data through the pipeline.
        
        Args:
            data: List of raw data items
            verify: Whether to verify reasoning paths
            tokenize: Whether to tokenize for training
            
        Returns:
            Processed data
        """
        if not self.workers:
            raise RuntimeError("Workers not initialized. Call initialize() first.")
        
        # Split data among workers
        num_workers = len(self.workers)
        chunk_size = (len(data) + num_workers - 1) // num_workers
        chunks = [
            data[i:i + chunk_size]
            for i in range(0, len(data), chunk_size)
        ]
        
        results = []
        
        # Verify
        if verify:
            verify_futures = [
                worker.process_batch.remote(chunk)
                for worker, chunk in zip(self.workers, chunks)
            ]
            verified_results = ray.get(verify_futures)
            for chunk_result in verified_results:
                results.extend(chunk_result)
        else:
            results = data
        
        # Tokenize
        if tokenize:
            chunk_size = (len(results) + num_workers - 1) // num_workers
            chunks = [
                results[i:i + chunk_size]
                for i in range(0, len(results), chunk_size)
            ]
            
            tokenize_futures = [
                tokenizer.tokenize_batch.remote(chunk)
                for tokenizer, chunk in zip(self.tokenizers, chunks)
            ]
            tokenized_results = ray.get(tokenize_futures)
            
            results = []
            for chunk_result in tokenized_results:
                results.extend(chunk_result)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        stats_futures = [worker.get_stats.remote() for worker in self.workers]
        worker_stats = ray.get(stats_futures)
        
        return {
            "cluster": self.cluster_manager.get_cluster_info(),
            "workers": worker_stats,
            "total_processed": sum(s["processed_count"] for s in worker_stats),
        }
    
    def shutdown(self):
        """Shutdown workers and cluster."""
        self.workers = []
        self.tokenizers = []
        self.batch_preparers = []
        self.cluster_manager.shutdown()


@_ray_remote_or_class
class KafkaRayBridge:
    """
    Bridge between Kafka and Ray for streaming processing.
    Consumes from Kafka, processes with Ray, produces results.
    """
    
    def __init__(
        self,
        kafka_config_dict: Dict[str, Any],
        input_topic: str,
        output_topic: str,
        processor_type: str = "verify",
    ):
        from .kafka_streaming import KafkaConfig, KafkaConsumer, KafkaProducer
        
        self.kafka_config = KafkaConfig(**kafka_config_dict)
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.processor_type = processor_type
        
        self.consumer = KafkaConsumer(self.kafka_config, [input_topic])
        self.producer = KafkaProducer(self.kafka_config)
        
        self.running = False
        self.processed_count = 0
    
    def start(self, batch_size: int = 10):
        """Start processing loop."""
        self.consumer.initialize()
        self.producer.initialize()
        self.running = True
        
        batch = []
        
        while self.running:
            messages = self.consumer.poll(timeout=1.0)
            
            for msg in messages:
                batch.append(msg["value"])
                
                if len(batch) >= batch_size:
                    results = self._process_batch(batch)
                    for result in results:
                        self.producer.send(self.output_topic, result)
                    self.producer.flush()
                    batch = []
                    self.processed_count += len(results)
            
            # Process remaining
            if batch and not messages:
                results = self._process_batch(batch)
                for result in results:
                    self.producer.send(self.output_topic, result)
                self.producer.flush()
                batch = []
    
    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of messages."""
        # Simplified processing - in real use would dispatch to other workers
        return [{"processed": True, "data": item} for item in batch]
    
    def stop(self):
        """Stop processing."""
        self.running = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "processed_count": self.processed_count,
            "running": self.running,
        }
