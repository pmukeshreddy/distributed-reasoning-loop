"""
Orchestration module for distributed data processing.
Kafka for streaming, Ray for distributed compute.
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
    )
    _ray_available = True
except ImportError:
    RayClusterManager = None
    RayClusterConfig = None
    DataProcessingWorker = None
    TokenizationWorker = None
    BatchPreparationWorker = None
    _ray_available = False

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
    # KV Cache
    "KVCacheManager",
    "DistributedKVCache",
    "CacheEntry",
    "CacheStats",
    # Availability flags
    "_kafka_available",
    "_ray_available",
]
