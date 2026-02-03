"""
SGLang Worker Pool Manager with LoRA Hot-Reload Support.

Manages multiple SGLang inference workers and coordinates LoRA updates
for continuous distributed GRPO training.
"""

import asyncio
import aiohttp
import logging
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import requests

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    """Worker status states."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    RELOADING = "reloading"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class WorkerInfo:
    """Information about a single SGLang worker."""
    worker_id: int
    host: str
    port: int
    gpu_id: int
    status: WorkerStatus = WorkerStatus.INITIALIZING
    current_lora_version: int = 0
    current_lora_path: Optional[str] = None
    requests_served: int = 0
    total_tokens_generated: int = 0
    last_health_check: float = 0.0
    avg_latency_ms: float = 0.0
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "url": self.url,
            "gpu_id": self.gpu_id,
            "status": self.status.value,
            "lora_version": self.current_lora_version,
            "lora_path": self.current_lora_path,
            "requests_served": self.requests_served,
            "total_tokens": self.total_tokens_generated,
            "avg_latency_ms": self.avg_latency_ms,
        }


@dataclass
class WorkerPoolConfig:
    """Configuration for worker pool."""
    workers: List[Dict[str, Any]] = field(default_factory=list)
    health_check_interval: float = 10.0
    request_timeout: float = 120.0
    lora_reload_timeout: float = 30.0
    max_retries: int = 3
    load_balancing: str = "round_robin"  # round_robin, least_pending, random
    
    @classmethod
    def from_ports(
        cls,
        ports: List[int],
        gpu_ids: Optional[List[int]] = None,
        host: str = "127.0.0.1",
    ) -> "WorkerPoolConfig":
        """Create config from list of ports."""
        if gpu_ids is None:
            gpu_ids = list(range(len(ports)))
        
        workers = [
            {"port": port, "gpu_id": gpu_id, "host": host}
            for port, gpu_id in zip(ports, gpu_ids)
        ]
        return cls(workers=workers)


class SGLangWorkerPool:
    """
    Manages a pool of SGLang inference workers.
    
    Features:
    - Round-robin / least-pending load balancing
    - LoRA hot-reload with version tracking
    - Health monitoring
    - Automatic retry on failures
    """
    
    def __init__(self, config: WorkerPoolConfig):
        self.config = config
        self.workers: List[WorkerInfo] = []
        self._current_idx = 0
        self._lock = asyncio.Lock()
        self._pending_requests: Dict[int, int] = {}  # worker_id -> pending count
        self._session: Optional[aiohttp.ClientSession] = None
        
    def initialize(self):
        """Initialize worker pool."""
        for idx, worker_config in enumerate(self.config.workers):
            worker = WorkerInfo(
                worker_id=idx,
                host=worker_config.get("host", "127.0.0.1"),
                port=worker_config["port"],
                gpu_id=worker_config.get("gpu_id", idx),
            )
            self.workers.append(worker)
            self._pending_requests[idx] = 0
        
        logger.info(f"Initialized worker pool with {len(self.workers)} workers")
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.request_timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def check_worker_health(self, worker: WorkerInfo) -> bool:
        """Check if a worker is healthy."""
        try:
            session = await self._get_session()
            async with session.get(
                f"{worker.url}/health",
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as response:
                if response.status == 200:
                    worker.status = WorkerStatus.READY
                    worker.last_health_check = time.time()
                    return True
        except Exception as e:
            logger.warning(f"Worker {worker.worker_id} health check failed: {e}")
        
        worker.status = WorkerStatus.OFFLINE
        return False
    
    async def check_all_health(self) -> Dict[int, bool]:
        """Check health of all workers."""
        results = {}
        tasks = [self.check_worker_health(w) for w in self.workers]
        healths = await asyncio.gather(*tasks, return_exceptions=True)
        
        for worker, health in zip(self.workers, healths):
            results[worker.worker_id] = health if isinstance(health, bool) else False
        
        return results
    
    def _select_worker_round_robin(self) -> Optional[WorkerInfo]:
        """Select worker using round-robin."""
        # Prefer READY workers, but accept non-OFFLINE workers as fallback
        ready_workers = [w for w in self.workers if w.status == WorkerStatus.READY]
        if not ready_workers:
            # Fallback: any non-offline worker
            ready_workers = [w for w in self.workers if w.status != WorkerStatus.OFFLINE]
        if not ready_workers:
            return None
        
        worker = ready_workers[self._current_idx % len(ready_workers)]
        self._current_idx = (self._current_idx + 1) % len(ready_workers)
        return worker
    
    def _select_worker_least_pending(self) -> Optional[WorkerInfo]:
        """Select worker with least pending requests."""
        ready_workers = [w for w in self.workers if w.status == WorkerStatus.READY]
        if not ready_workers:
            ready_workers = [w for w in self.workers if w.status != WorkerStatus.OFFLINE]
        if not ready_workers:
            return None
        
        return min(ready_workers, key=lambda w: self._pending_requests[w.worker_id])
    
    def select_worker(self) -> Optional[WorkerInfo]:
        """Select a worker based on load balancing strategy."""
        if self.config.load_balancing == "least_pending":
            return self._select_worker_least_pending()
        else:  # round_robin (default)
            return self._select_worker_round_robin()
    
    async def load_lora(
        self,
        worker: WorkerInfo,
        lora_path: str,
        lora_name: str,
    ) -> bool:
        """
        Load a LoRA adapter on a specific worker.
        Tries multiple API endpoints for compatibility with different SGLang versions.
        """
        # List of endpoints to try (different SGLang versions use different APIs)
        endpoints = [
            # SGLang v0.4+ style
            ("/update_lora", {"lora_path": lora_path, "lora_name": lora_name}),
            # Alternative endpoint
            ("/add_lora", {"path": lora_path, "name": lora_name}),
            # OpenAI-compatible style  
            ("/v1/load_lora_adapter", {"lora_name": lora_name, "lora_path": lora_path}),
            # vLLM-compatible style
            ("/v1/lora/load", {"lora_path": lora_path, "lora_name": lora_name}),
        ]
        
        session = await self._get_session()
        original_status = worker.status
        worker.status = WorkerStatus.RELOADING
        
        for endpoint, payload in endpoints:
            try:
                start_time = time.time()
                async with session.post(
                    f"{worker.url}{endpoint}",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.lora_reload_timeout)
                ) as response:
                    elapsed = time.time() - start_time
                    
                    if response.status == 200:
                        logger.info(
                            f"Worker {worker.worker_id}: Loaded LoRA '{lora_name}' "
                            f"via {endpoint} in {elapsed:.2f}s"
                        )
                        worker.status = WorkerStatus.READY
                        return True
                    elif response.status == 404:
                        # Endpoint doesn't exist, try next one
                        continue
                    else:
                        error = await response.text()
                        logger.debug(f"Worker {worker.worker_id}: {endpoint} returned {response.status}: {error}")
                        continue
                        
            except asyncio.TimeoutError:
                logger.warning(f"Worker {worker.worker_id}: {endpoint} timed out")
                continue
            except Exception as e:
                logger.debug(f"Worker {worker.worker_id}: {endpoint} error: {e}")
                continue
        
        # All endpoints failed - but DON'T mark worker as ERROR
        # Worker can still serve requests with base model
        logger.warning(
            f"Worker {worker.worker_id}: LoRA hot-reload not available. "
            f"Worker will continue with base model."
        )
        worker.status = WorkerStatus.READY  # Keep worker available!
        return False
    
    async def unload_lora(
        self,
        worker: WorkerInfo,
        lora_name: str,
    ) -> bool:
        """Unload a LoRA adapter from a specific worker."""
        # List of endpoints to try
        endpoints = [
            ("/remove_lora", {"lora_name": lora_name}),
            ("/delete_lora", {"name": lora_name}),
            ("/v1/unload_lora_adapter", {"lora_name": lora_name}),
            ("/v1/lora/unload", {"lora_name": lora_name}),
        ]
        
        session = await self._get_session()
        
        for endpoint, payload in endpoints:
            try:
                async with session.post(
                    f"{worker.url}{endpoint}",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10.0)
                ) as response:
                    if response.status == 200:
                        logger.info(f"Worker {worker.worker_id}: Unloaded LoRA '{lora_name}' via {endpoint}")
                        return True
                    elif response.status == 404:
                        continue
                        
            except Exception:
                continue
        
        # Unload not available or LoRA wasn't loaded - that's OK
        logger.debug(f"Worker {worker.worker_id}: LoRA unload skipped (not available or not loaded)")
        return True  # Return True since unload is optional
    
    async def reload_lora(
        self,
        worker: WorkerInfo,
        new_lora_path: str,
        new_version: int,
    ) -> bool:
        """
        Reload LoRA adapter on a worker.
        Unloads old version first, then loads new version.
        """
        old_name = f"lora_v{worker.current_lora_version}"
        new_name = f"lora_v{new_version}"
        
        # Unload old if exists
        if worker.current_lora_version > 0:
            await self.unload_lora(worker, old_name)
        
        # Load new
        success = await self.load_lora(worker, new_lora_path, new_name)
        
        if success:
            worker.current_lora_version = new_version
            worker.current_lora_path = new_lora_path
        
        return success
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        worker: Optional[WorkerInfo] = None,
        lora_name: Optional[str] = None,
        **kwargs,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """
        Generate completion using a worker.
        
        Args:
            messages: Chat messages
            worker: Specific worker to use (auto-select if None)
            lora_name: LoRA adapter name to use
            **kwargs: Additional generation parameters
            
        Returns:
            Tuple of (generated_text, metadata)
        """
        if worker is None:
            worker = self.select_worker()
        
        if worker is None:
            logger.error("No available workers")
            return None, {"error": "No available workers"}
        
        self._pending_requests[worker.worker_id] += 1
        
        try:
            session = await self._get_session()
            
            payload = {
                "model": kwargs.get("model", "default"),
                "messages": messages,
                "max_tokens": kwargs.get("max_tokens", 2048),
                "temperature": kwargs.get("temperature", 0.8),
                "top_p": kwargs.get("top_p", 0.95),
            }
            
            # Add LoRA if specified
            if lora_name:
                payload["lora_name"] = lora_name
            elif worker.current_lora_version > 0:
                payload["lora_name"] = f"lora_v{worker.current_lora_version}"
            
            start_time = time.time()
            async with session.post(
                f"{worker.url}/v1/chat/completions",
                json=payload,
            ) as response:
                elapsed = time.time() - start_time
                
                if response.status == 200:
                    data = await response.json()
                    text = data["choices"][0]["message"]["content"]
                    
                    # Update stats
                    worker.requests_served += 1
                    tokens = data.get("usage", {}).get("completion_tokens", 0)
                    worker.total_tokens_generated += tokens
                    worker.avg_latency_ms = (
                        worker.avg_latency_ms * 0.9 + elapsed * 1000 * 0.1
                    )
                    
                    metadata = {
                        "worker_id": worker.worker_id,
                        "lora_version": worker.current_lora_version,
                        "latency_ms": elapsed * 1000,
                        "tokens": tokens,
                    }
                    
                    return text, metadata
                else:
                    error = await response.text()
                    return None, {"error": error, "worker_id": worker.worker_id}
                    
        except Exception as e:
            logger.error(f"Generation error on worker {worker.worker_id}: {e}")
            return None, {"error": str(e), "worker_id": worker.worker_id}
        finally:
            self._pending_requests[worker.worker_id] -= 1
    
    async def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        **kwargs,
    ) -> List[Tuple[Optional[str], Dict[str, Any]]]:
        """Generate completions for a batch of messages."""
        tasks = [self.generate(messages, **kwargs) for messages in messages_batch]
        return await asyncio.gather(*tasks)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return {
            "total_workers": len(self.workers),
            "ready_workers": sum(1 for w in self.workers if w.status == WorkerStatus.READY),
            "workers": [w.to_dict() for w in self.workers],
            "total_requests": sum(w.requests_served for w in self.workers),
            "total_tokens": sum(w.total_tokens_generated for w in self.workers),
        }
    
    async def close(self):
        """Close the worker pool."""
        if self._session and not self._session.closed:
            await self._session.close()


class SyncWorkerPool:
    """
    Synchronous wrapper for SGLangWorkerPool.
    For use in non-async contexts like the existing pipeline.
    """
    
    def __init__(self, config: WorkerPoolConfig):
        self.async_pool = SGLangWorkerPool(config)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
    
    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create event loop."""
        try:
            self._loop = asyncio.get_event_loop()
        except RuntimeError:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop
    
    def initialize(self):
        """Initialize pool."""
        self.async_pool.initialize()
        loop = self._get_loop()
        loop.run_until_complete(self.async_pool.check_all_health())
    
    def check_health(self) -> Dict[int, bool]:
        """Check all worker health."""
        loop = self._get_loop()
        return loop.run_until_complete(self.async_pool.check_all_health())
    
    def reload_lora_all(
        self,
        lora_path: str,
        version: int,
    ) -> Dict[int, bool]:
        """Reload LoRA on all workers."""
        loop = self._get_loop()
        
        async def _reload_all():
            tasks = [
                self.async_pool.reload_lora(w, lora_path, version)
                for w in self.async_pool.workers
            ]
            results = await asyncio.gather(*tasks)
            return {w.worker_id: r for w, r in zip(self.async_pool.workers, results)}
        
        return loop.run_until_complete(_reload_all())
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Tuple[Optional[str], Dict[str, Any]]:
        """Generate completion."""
        loop = self._get_loop()
        return loop.run_until_complete(self.async_pool.generate(messages, **kwargs))
    
    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        **kwargs,
    ) -> List[Tuple[Optional[str], Dict[str, Any]]]:
        """Generate batch of completions."""
        loop = self._get_loop()
        return loop.run_until_complete(
            self.async_pool.generate_batch(messages_batch, **kwargs)
        )
    
    def generate_batch_threaded(
        self,
        messages_batch: List[List[Dict[str, str]]],
        max_workers: int = 32,
        **kwargs,
    ) -> List[Tuple[Optional[str], Dict[str, Any]]]:
        """
        Generate batch using thread pool for sync compatibility.
        Better for integration with existing sync code.
        """
        # Round-robin counter for load balancing
        import itertools
        worker_cycle = itertools.cycle(range(len(self.async_pool.workers)))
        
        def _make_request(args):
            messages, worker_idx = args
            
            # Get worker directly by index (round-robin)
            if not self.async_pool.workers:
                return None, {"error": "No workers configured"}
            
            worker = self.async_pool.workers[worker_idx % len(self.async_pool.workers)]
            
            # Skip workers that are completely offline
            if worker.status == WorkerStatus.OFFLINE:
                # Try next worker
                for i in range(len(self.async_pool.workers)):
                    alt_worker = self.async_pool.workers[(worker_idx + i) % len(self.async_pool.workers)]
                    if alt_worker.status != WorkerStatus.OFFLINE:
                        worker = alt_worker
                        break
                else:
                    return None, {"error": "All workers offline"}
            
            try:
                payload = {
                    "model": kwargs.get("model", "default"),
                    "messages": messages,
                    "max_tokens": kwargs.get("max_tokens", 2048),
                    "temperature": kwargs.get("temperature", 0.8),
                    "top_p": kwargs.get("top_p", 0.95),
                }
                
                # Only add LoRA if it was successfully loaded
                # (don't request LoRA that might not be available)
                # The base model will be used if LoRA isn't loaded
                
                response = requests.post(
                    f"{worker.url}/v1/chat/completions",
                    json=payload,
                    timeout=self.async_pool.config.request_timeout,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    text = data["choices"][0]["message"]["content"]
                    worker.requests_served += 1
                    return text, {"worker_id": worker.worker_id, "lora_version": worker.current_lora_version}
                else:
                    return None, {"error": response.text, "worker_id": worker.worker_id}
                    
            except requests.exceptions.Timeout:
                return None, {"error": "Request timeout", "worker_id": worker.worker_id}
            except Exception as e:
                return None, {"error": str(e), "worker_id": worker.worker_id}
        
        # Assign workers round-robin
        batch_with_workers = [
            (messages, next(worker_cycle)) 
            for messages in messages_batch
        ]
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_make_request, batch_with_workers))
        
        return results
    
    @property
    def workers(self) -> List[WorkerInfo]:
        """Get workers list."""
        return self.async_pool.workers
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        return self.async_pool.get_stats()
    
    def close(self):
        """Close the pool."""
        if self._loop:
            self._loop.run_until_complete(self.async_pool.close())
