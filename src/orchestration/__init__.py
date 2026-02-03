"""
Orchestration module for distributed data processing.
Kafka for streaming, Ray for distributed compute, Worker Pool for continuous training.
"""

# Kafka imports (optional)
try:
    from .kafka_streaming import (
        KafkaProducer,
        KafkaConsumer,
        KafkaConfig,
        ReasoningDataProducer,
        ReasoningDataConsumer,
    )
    _kafka_available = True
except ImportError:
    KafkaProducer = None
    KafkaConsumer = None
    KafkaConfig = None
    ReasoningDataProducer = None
    ReasoningDataConsumer = None
    _kafka_available = False

# Ray imports (optional)
try:
    from .ray_workers import (
        RayClusterManager,
        RayClusterConfig,
        DataProcessingWorker,
        TokenizationWorker,
        BatchPreparationWorker,
        DistributedDataProcessor,
    )
    _ray_available = True
except ImportError:
    RayClusterManager = None
    RayClusterConfig = None
    DataProcessingWorker = None
    TokenizationWorker = None
    BatchPreparationWorker = None
    DistributedDataProcessor = None
    _ray_available = False

# Worker Pool for continuous distributed GRPO
from .worker_pool import (
    SGLangWorkerPool,
    SyncWorkerPool,
    WorkerPoolConfig,
    WorkerInfo,
    WorkerStatus,
)

# Coordinator for continuous training
from .coordinator import (
    TrainingCoordinator,
    CoordinatorConfig,
    LoRACheckpoint,
    ContinuousTrainingLoop,
)

# KV Cache (always available)
from .kv_cache_manager import (
    KVCacheManager,
    DistributedKVCache,
    CacheEntry,
    CacheStats,
)

__all__ = [
    # Kafka
    "KafkaProducer",
    "KafkaConsumer",
    "KafkaConfig",
    "ReasoningDataProducer",
    "ReasoningDataConsumer",
    # Ray
    "RayClusterManager",
    "RayClusterConfig",
    "DataProcessingWorker",
    "TokenizationWorker",
    "BatchPreparationWorker",
    "DistributedDataProcessor",
    # Worker Pool (Continuous GRPO)
    "SGLangWorkerPool",
    "SyncWorkerPool",
    "WorkerPoolConfig",
    "WorkerInfo",
    "WorkerStatus",
    # Coordinator (Continuous GRPO)
    "TrainingCoordinator",
    "CoordinatorConfig",
    "LoRACheckpoint",
    "ContinuousTrainingLoop",
    # KV Cache
    "KVCacheManager",
    "DistributedKVCache",
    "CacheEntry",
    "CacheStats",
    # Availability flags
    "_kafka_available",
    "_ray_available",
]
