"""
Kafka streaming for reasoning data pipeline.
Handles ingestion, processing, and distribution of reasoning paths.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Iterator
from abc import ABC, abstractmethod
import threading
import time
from queue import Queue
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class KafkaConfig:
    """Kafka configuration."""
    bootstrap_servers: List[str] = field(default_factory=lambda: ["localhost:9092"])
    client_id: str = "reasoning-loop"
    group_id: str = "reasoning-consumer-group"
    
    # Topics
    raw_reasoning_topic: str = "raw_reasoning_data"
    verified_paths_topic: str = "verified_paths"
    training_data_topic: str = "training_data"
    
    # Producer settings
    acks: str = "all"
    retries: int = 3
    batch_size: int = 16384
    linger_ms: int = 10
    compression_type: str = "gzip"
    
    # Consumer settings
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = True
    auto_commit_interval_ms: int = 5000
    max_poll_records: int = 500
    session_timeout_ms: int = 30000


class KafkaProducer:
    """
    High-throughput Kafka producer for reasoning data.
    Supports batching and async delivery.
    """
    
    def __init__(self, config: KafkaConfig):
        self.config = config
        self.producer = None
        self._initialized = False
        self._delivery_reports = Queue()
        
    def initialize(self):
        """Initialize Kafka producer."""
        if self._initialized:
            return
            
        try:
            from confluent_kafka import Producer
            
            conf = {
                'bootstrap.servers': ','.join(self.config.bootstrap_servers),
                'client.id': self.config.client_id,
                'acks': self.config.acks,
                'retries': self.config.retries,
                'batch.size': self.config.batch_size,
                'linger.ms': self.config.linger_ms,
                'compression.type': self.config.compression_type,
            }
            
            self.producer = Producer(conf)
            self._initialized = True
            logger.info("Kafka producer initialized")
            
        except ImportError:
            logger.warning("confluent-kafka not installed, using kafka-python fallback")
            self._init_kafka_python()
    
    def _init_kafka_python(self):
        """Fallback to kafka-python library."""
        try:
            from kafka import KafkaProducer as KP
            
            self.producer = KP(
                bootstrap_servers=self.config.bootstrap_servers,
                client_id=self.config.client_id,
                acks=self.config.acks,
                retries=self.config.retries,
                batch_size=self.config.batch_size,
                linger_ms=self.config.linger_ms,
                compression_type=self.config.compression_type,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
            )
            self._initialized = True
            self._use_confluent = False
            logger.info("Kafka producer initialized (kafka-python)")
            
        except ImportError:
            raise ImportError("Neither confluent-kafka nor kafka-python installed")
    
    def _delivery_callback(self, err, msg):
        """Callback for message delivery reports."""
        if err:
            logger.error(f"Message delivery failed: {err}")
            self._delivery_reports.put(("error", str(err)))
        else:
            self._delivery_reports.put(("success", msg.topic()))
    
    def send(
        self,
        topic: str,
        value: Dict[str, Any],
        key: Optional[str] = None,
        partition: Optional[int] = None,
    ):
        """
        Send a message to Kafka topic.
        
        Args:
            topic: Target topic name
            value: Message value (will be JSON serialized)
            key: Optional message key for partitioning
            partition: Optional explicit partition
        """
        self.initialize()
        
        if hasattr(self.producer, 'produce'):  # confluent-kafka
            self.producer.produce(
                topic=topic,
                value=json.dumps(value).encode('utf-8'),
                key=key.encode('utf-8') if key else None,
                partition=partition,
                callback=self._delivery_callback,
            )
            self.producer.poll(0)  # Trigger callbacks
        else:  # kafka-python
            future = self.producer.send(
                topic=topic,
                value=value,
                key=key,
                partition=partition,
            )
            return future
    
    def send_batch(
        self,
        topic: str,
        messages: List[Dict[str, Any]],
        key_func: Optional[Callable[[Dict], str]] = None,
    ):
        """Send a batch of messages."""
        self.initialize()
        
        for msg in messages:
            key = key_func(msg) if key_func else None
            self.send(topic, msg, key)
        
        self.flush()
    
    def flush(self, timeout: float = 10.0):
        """Flush all pending messages."""
        if self.producer:
            if hasattr(self.producer, 'flush'):
                self.producer.flush(timeout)
            else:
                self.producer.flush(timeout=timeout * 1000)
    
    def close(self):
        """Close the producer."""
        self.flush()
        if hasattr(self.producer, 'close'):
            self.producer.close()


class KafkaConsumer:
    """
    Kafka consumer for processing reasoning data.
    Supports consumer groups and parallel processing.
    """
    
    def __init__(self, config: KafkaConfig, topics: List[str]):
        self.config = config
        self.topics = topics
        self.consumer = None
        self._initialized = False
        self._running = False
        
    def initialize(self):
        """Initialize Kafka consumer."""
        if self._initialized:
            return
            
        try:
            from confluent_kafka import Consumer
            
            conf = {
                'bootstrap.servers': ','.join(self.config.bootstrap_servers),
                'group.id': self.config.group_id,
                'auto.offset.reset': self.config.auto_offset_reset,
                'enable.auto.commit': self.config.enable_auto_commit,
                'auto.commit.interval.ms': self.config.auto_commit_interval_ms,
                'max.poll.records': self.config.max_poll_records,
                'session.timeout.ms': self.config.session_timeout_ms,
            }
            
            self.consumer = Consumer(conf)
            self.consumer.subscribe(self.topics)
            self._initialized = True
            self._use_confluent = True
            logger.info(f"Kafka consumer initialized, subscribed to {self.topics}")
            
        except ImportError:
            logger.warning("confluent-kafka not installed, using kafka-python fallback")
            self._init_kafka_python()
    
    def _init_kafka_python(self):
        """Fallback to kafka-python library."""
        try:
            from kafka import KafkaConsumer as KC
            
            self.consumer = KC(
                *self.topics,
                bootstrap_servers=self.config.bootstrap_servers,
                group_id=self.config.group_id,
                auto_offset_reset=self.config.auto_offset_reset,
                enable_auto_commit=self.config.enable_auto_commit,
                auto_commit_interval_ms=self.config.auto_commit_interval_ms,
                max_poll_records=self.config.max_poll_records,
                session_timeout_ms=self.config.session_timeout_ms,
                value_deserializer=lambda v: json.loads(v.decode('utf-8')),
                key_deserializer=lambda k: k.decode('utf-8') if k else None,
            )
            self._initialized = True
            self._use_confluent = False
            logger.info(f"Kafka consumer initialized (kafka-python), subscribed to {self.topics}")
            
        except ImportError:
            raise ImportError("Neither confluent-kafka nor kafka-python installed")
    
    def poll(self, timeout: float = 1.0) -> List[Dict[str, Any]]:
        """Poll for new messages."""
        self.initialize()
        
        messages = []
        
        if self._use_confluent:
            msg = self.consumer.poll(timeout)
            if msg is None:
                return messages
            if msg.error():
                logger.error(f"Consumer error: {msg.error()}")
                return messages
            
            try:
                value = json.loads(msg.value().decode('utf-8'))
                messages.append({
                    'value': value,
                    'key': msg.key().decode('utf-8') if msg.key() else None,
                    'topic': msg.topic(),
                    'partition': msg.partition(),
                    'offset': msg.offset(),
                })
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode message: {e}")
        else:
            # kafka-python returns messages differently
            records = self.consumer.poll(timeout_ms=int(timeout * 1000))
            for tp, msgs in records.items():
                for msg in msgs:
                    messages.append({
                        'value': msg.value,
                        'key': msg.key,
                        'topic': msg.topic,
                        'partition': msg.partition,
                        'offset': msg.offset,
                    })
        
        return messages
    
    def consume(
        self,
        callback: Callable[[Dict[str, Any]], None],
        batch_size: int = 100,
        poll_timeout: float = 1.0,
    ):
        """
        Continuously consume messages and process with callback.
        
        Args:
            callback: Function to process each message
            batch_size: Number of messages to batch before processing
            poll_timeout: Timeout for each poll
        """
        self.initialize()
        self._running = True
        
        batch = []
        
        while self._running:
            messages = self.poll(poll_timeout)
            
            for msg in messages:
                batch.append(msg['value'])
                
                if len(batch) >= batch_size:
                    for item in batch:
                        try:
                            callback(item)
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                    batch = []
            
            # Process remaining batch
            if batch and not messages:
                for item in batch:
                    try:
                        callback(item)
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                batch = []
    
    def stop(self):
        """Stop consuming."""
        self._running = False
    
    def close(self):
        """Close the consumer."""
        self.stop()
        if self.consumer:
            self.consumer.close()


class ReasoningDataProducer:
    """
    Specialized producer for reasoning pipeline data.
    Handles different message types in the pipeline.
    """
    
    def __init__(self, config: KafkaConfig):
        self.config = config
        self.producer = KafkaProducer(config)
        
    def send_raw_reasoning(self, reasoning_path: Dict[str, Any]):
        """Send raw generated reasoning path."""
        self.producer.send(
            topic=self.config.raw_reasoning_topic,
            value={
                "type": "raw_reasoning",
                "data": reasoning_path,
                "timestamp": time.time(),
            },
            key=reasoning_path.get("problem_id"),
        )
    
    def send_verified_path(self, verified_path: Dict[str, Any]):
        """Send verified reasoning path."""
        self.producer.send(
            topic=self.config.verified_paths_topic,
            value={
                "type": "verified_path",
                "data": verified_path,
                "timestamp": time.time(),
            },
            key=verified_path.get("problem_id"),
        )
    
    def send_training_sample(self, sample: Dict[str, Any]):
        """Send prepared training sample."""
        self.producer.send(
            topic=self.config.training_data_topic,
            value={
                "type": "training_sample",
                "data": sample,
                "timestamp": time.time(),
            },
            key=sample.get("problem_id"),
        )
    
    def send_batch_reasoning(self, paths: List[Dict[str, Any]]):
        """Send batch of reasoning paths."""
        for path in paths:
            self.send_raw_reasoning(path)
        self.producer.flush()
    
    def close(self):
        self.producer.close()


class ReasoningDataConsumer:
    """
    Specialized consumer for reasoning pipeline data.
    Handles different stages of the pipeline.
    """
    
    def __init__(
        self,
        config: KafkaConfig,
        stage: str = "raw",  # raw, verified, training
    ):
        self.config = config
        
        topic_map = {
            "raw": config.raw_reasoning_topic,
            "verified": config.verified_paths_topic,
            "training": config.training_data_topic,
        }
        
        if stage not in topic_map:
            raise ValueError(f"Unknown stage: {stage}")
        
        self.topic = topic_map[stage]
        self.consumer = KafkaConsumer(config, [self.topic])
        self.stage = stage
    
    def process_messages(
        self,
        processor: Callable[[Dict[str, Any]], Optional[Dict[str, Any]]],
        output_producer: Optional[ReasoningDataProducer] = None,
        output_stage: Optional[str] = None,
    ):
        """
        Process messages and optionally forward to next stage.
        
        Args:
            processor: Function to process each message, returns processed data or None
            output_producer: Producer for forwarding results
            output_stage: Stage to forward to (verified, training)
        """
        def callback(message: Dict[str, Any]):
            data = message.get("data", message)
            result = processor(data)
            
            if result and output_producer and output_stage:
                if output_stage == "verified":
                    output_producer.send_verified_path(result)
                elif output_stage == "training":
                    output_producer.send_training_sample(result)
        
        self.consumer.consume(callback)
    
    def iter_messages(self, max_messages: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Iterate over messages."""
        self.consumer.initialize()
        count = 0
        
        while max_messages is None or count < max_messages:
            messages = self.consumer.poll(timeout=1.0)
            
            for msg in messages:
                yield msg['value'].get('data', msg['value'])
                count += 1
                
                if max_messages and count >= max_messages:
                    break
            
            if not messages:
                break
    
    def close(self):
        self.consumer.close()


class KafkaAdminClient:
    """Admin client for topic management."""
    
    def __init__(self, config: KafkaConfig):
        self.config = config
        self.admin = None
        
    def initialize(self):
        """Initialize admin client."""
        try:
            from confluent_kafka.admin import AdminClient, NewTopic
            
            self.admin = AdminClient({
                'bootstrap.servers': ','.join(self.config.bootstrap_servers),
            })
            self._NewTopic = NewTopic
            
        except ImportError:
            from kafka.admin import KafkaAdminClient as KAC, NewTopic
            
            self.admin = KAC(
                bootstrap_servers=self.config.bootstrap_servers,
            )
            self._NewTopic = NewTopic
    
    def create_topics(
        self,
        topics: List[str],
        num_partitions: int = 3,
        replication_factor: int = 1,
    ):
        """Create topics if they don't exist."""
        self.initialize()
        
        new_topics = [
            self._NewTopic(
                topic,
                num_partitions=num_partitions,
                replication_factor=replication_factor,
            )
            for topic in topics
        ]
        
        if hasattr(self.admin, 'create_topics'):
            # confluent-kafka
            fs = self.admin.create_topics(new_topics)
            for topic, f in fs.items():
                try:
                    f.result()
                    logger.info(f"Topic {topic} created")
                except Exception as e:
                    logger.warning(f"Topic {topic} creation failed: {e}")
        else:
            # kafka-python
            try:
                self.admin.create_topics(new_topics)
                logger.info(f"Topics created: {topics}")
            except Exception as e:
                logger.warning(f"Topic creation failed: {e}")
    
    def setup_pipeline_topics(self):
        """Create all topics needed for the reasoning pipeline."""
        topics = [
            self.config.raw_reasoning_topic,
            self.config.verified_paths_topic,
            self.config.training_data_topic,
        ]
        self.create_topics(topics)
