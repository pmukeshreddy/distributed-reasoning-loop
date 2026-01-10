"""
Distributed KV-Cache Manager for inference optimization.
Implements RadixAttention-style prefix caching for efficient generation.
"""

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry with KV tensors."""
    key_hash: str
    prefix_tokens: List[int]
    kv_tensors: Optional[Any] = None  # Actual KV cache tensors
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.time()


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_entries: int = 0
    memory_used_bytes: int = 0
    
    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class KVCacheManager:
    """
    Local KV-Cache manager with LRU eviction.
    Caches computed KV tensors for prefix reuse.
    """
    
    def __init__(
        self,
        max_memory_bytes: int = 4 * 1024 * 1024 * 1024,  # 4GB
        eviction_policy: str = "lru",
    ):
        self.max_memory_bytes = max_memory_bytes
        self.eviction_policy = eviction_policy
        
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.stats = CacheStats()
        self._lock = threading.Lock()
    
    def _compute_hash(self, tokens: List[int]) -> str:
        """Compute hash for token sequence."""
        token_str = ",".join(map(str, tokens))
        return hashlib.sha256(token_str.encode()).hexdigest()[:32]
    
    def _find_longest_prefix(self, tokens: List[int]) -> Tuple[Optional[CacheEntry], int]:
        """
        Find the longest cached prefix for the given tokens.
        Returns the cache entry and the number of tokens matched.
        """
        best_entry = None
        best_length = 0
        
        # Try progressively shorter prefixes
        for length in range(len(tokens), 0, -1):
            prefix = tokens[:length]
            prefix_hash = self._compute_hash(prefix)
            
            if prefix_hash in self.cache:
                entry = self.cache[prefix_hash]
                if entry.prefix_tokens == prefix:
                    best_entry = entry
                    best_length = length
                    break
        
        return best_entry, best_length
    
    def get(self, tokens: List[int]) -> Tuple[Optional[Any], int]:
        """
        Get cached KV tensors for token sequence.
        
        Args:
            tokens: Input token sequence
            
        Returns:
            Tuple of (kv_tensors, num_cached_tokens)
            kv_tensors is None if no cache hit
        """
        with self._lock:
            entry, matched_length = self._find_longest_prefix(tokens)
            
            if entry is not None:
                # Move to end (most recently used)
                self.cache.move_to_end(entry.key_hash)
                entry.update_access()
                self.stats.hits += 1
                return entry.kv_tensors, matched_length
            else:
                self.stats.misses += 1
                return None, 0
    
    def put(
        self,
        tokens: List[int],
        kv_tensors: Any,
        size_bytes: Optional[int] = None,
    ):
        """
        Cache KV tensors for token sequence.
        
        Args:
            tokens: Token sequence
            kv_tensors: KV cache tensors to store
            size_bytes: Size of tensors in bytes (estimated if not provided)
        """
        key_hash = self._compute_hash(tokens)
        
        # Estimate size if not provided
        if size_bytes is None:
            size_bytes = self._estimate_size(kv_tensors)
        
        with self._lock:
            # Evict if necessary
            while (self.stats.memory_used_bytes + size_bytes > self.max_memory_bytes
                   and self.cache):
                self._evict_one()
            
            # Store entry
            entry = CacheEntry(
                key_hash=key_hash,
                prefix_tokens=tokens.copy(),
                kv_tensors=kv_tensors,
                size_bytes=size_bytes,
            )
            
            self.cache[key_hash] = entry
            self.stats.memory_used_bytes += size_bytes
            self.stats.total_entries = len(self.cache)
    
    def _estimate_size(self, kv_tensors: Any) -> int:
        """Estimate size of KV tensors in bytes."""
        if kv_tensors is None:
            return 0
        
        try:
            import torch
            if isinstance(kv_tensors, torch.Tensor):
                return kv_tensors.element_size() * kv_tensors.nelement()
            elif isinstance(kv_tensors, (list, tuple)):
                return sum(self._estimate_size(t) for t in kv_tensors)
        except ImportError:
            pass
        
        try:
            if isinstance(kv_tensors, np.ndarray):
                return kv_tensors.nbytes
        except:
            pass
        
        # Default estimate
        return 1024 * 1024  # 1MB default
    
    def _evict_one(self):
        """Evict one entry based on eviction policy."""
        if not self.cache:
            return
        
        if self.eviction_policy == "lru":
            # Remove oldest (first) entry
            key, entry = self.cache.popitem(last=False)
        elif self.eviction_policy == "lfu":
            # Remove least frequently used
            min_count = min(e.access_count for e in self.cache.values())
            for key, entry in self.cache.items():
                if entry.access_count == min_count:
                    del self.cache[key]
                    break
        else:
            # Default to LRU
            key, entry = self.cache.popitem(last=False)
        
        self.stats.memory_used_bytes -= entry.size_bytes
        self.stats.evictions += 1
        self.stats.total_entries = len(self.cache)
    
    def clear(self):
        """Clear the cache."""
        with self._lock:
            self.cache.clear()
            self.stats = CacheStats()
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats


class RadixTreeNode:
    """Node in radix tree for prefix matching."""
    
    def __init__(self):
        self.children: Dict[int, 'RadixTreeNode'] = {}
        self.kv_tensors: Optional[Any] = None
        self.token_count: int = 0
        self.is_complete: bool = False


class RadixKVCache:
    """
    RadixAttention-style KV cache using radix tree for efficient prefix matching.
    Enables O(n) prefix lookup instead of trying all prefixes.
    """
    
    def __init__(
        self,
        max_memory_bytes: int = 16 * 1024 * 1024 * 1024,  # 16GB
    ):
        self.max_memory_bytes = max_memory_bytes
        self.root = RadixTreeNode()
        self.stats = CacheStats()
        self._lock = threading.Lock()
        self._total_size = 0
    
    def get_prefix_cache(self, tokens: List[int]) -> Tuple[Optional[Any], int]:
        """
        Find longest cached prefix.
        
        Args:
            tokens: Input token sequence
            
        Returns:
            Tuple of (kv_tensors, num_cached_tokens)
        """
        with self._lock:
            node = self.root
            last_cached_node = None
            cached_length = 0
            
            for i, token in enumerate(tokens):
                if token not in node.children:
                    break
                
                node = node.children[token]
                
                if node.is_complete and node.kv_tensors is not None:
                    last_cached_node = node
                    cached_length = i + 1
            
            if last_cached_node is not None:
                self.stats.hits += 1
                return last_cached_node.kv_tensors, cached_length
            else:
                self.stats.misses += 1
                return None, 0
    
    def insert(self, tokens: List[int], kv_tensors: Any, size_bytes: int = 0):
        """
        Insert KV cache for token sequence.
        
        Args:
            tokens: Token sequence
            kv_tensors: KV cache tensors
            size_bytes: Size of tensors
        """
        with self._lock:
            node = self.root
            
            for token in tokens:
                if token not in node.children:
                    node.children[token] = RadixTreeNode()
                node = node.children[token]
            
            node.kv_tensors = kv_tensors
            node.token_count = len(tokens)
            node.is_complete = True
            
            self._total_size += size_bytes
            self.stats.total_entries += 1
            self.stats.memory_used_bytes = self._total_size
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats


class DistributedKVCache:
    """
    Distributed KV cache using Ray for cross-node sharing.
    Enables prefix sharing across multiple inference workers.
    """
    
    def __init__(
        self,
        num_shards: int = 4,
        max_memory_per_shard: int = 4 * 1024 * 1024 * 1024,
    ):
        self.num_shards = num_shards
        self.max_memory_per_shard = max_memory_per_shard
        self.shards = []
        self._initialized = False
    
    def initialize(self):
        """Initialize distributed cache shards."""
        if self._initialized:
            return
        
        try:
            import ray
            
            @ray.remote
            class CacheShard:
                def __init__(self, shard_id: int, max_memory: int):
                    self.shard_id = shard_id
                    self.cache = KVCacheManager(max_memory_bytes=max_memory)
                
                def get(self, tokens: List[int]) -> Tuple[Optional[Any], int]:
                    return self.cache.get(tokens)
                
                def put(self, tokens: List[int], kv_tensors: Any, size_bytes: int):
                    self.cache.put(tokens, kv_tensors, size_bytes)
                
                def get_stats(self) -> Dict[str, Any]:
                    stats = self.cache.get_stats()
                    return {
                        "shard_id": self.shard_id,
                        "hits": stats.hits,
                        "misses": stats.misses,
                        "hit_rate": stats.hit_rate,
                        "memory_used": stats.memory_used_bytes,
                    }
            
            self.shards = [
                CacheShard.remote(i, self.max_memory_per_shard)
                for i in range(self.num_shards)
            ]
            
            self._initialized = True
            logger.info(f"Initialized {self.num_shards} distributed cache shards")
            
        except ImportError:
            logger.warning("Ray not available, using local cache only")
            self.shards = [
                KVCacheManager(max_memory_bytes=self.max_memory_per_shard)
            ]
            self._initialized = True
    
    def _get_shard(self, tokens: List[int]) -> int:
        """Get shard index for tokens."""
        if not tokens:
            return 0
        # Hash first token for consistent sharding
        return tokens[0] % self.num_shards
    
    def get(self, tokens: List[int]) -> Tuple[Optional[Any], int]:
        """Get from distributed cache."""
        self.initialize()
        
        shard_idx = self._get_shard(tokens)
        shard = self.shards[shard_idx]
        
        try:
            import ray
            if ray.is_initialized():
                return ray.get(shard.get.remote(tokens))
        except:
            pass
        
        return shard.get(tokens)
    
    def put(self, tokens: List[int], kv_tensors: Any, size_bytes: int = 0):
        """Put into distributed cache."""
        self.initialize()
        
        shard_idx = self._get_shard(tokens)
        shard = self.shards[shard_idx]
        
        try:
            import ray
            if ray.is_initialized():
                ray.get(shard.put.remote(tokens, kv_tensors, size_bytes))
                return
        except:
            pass
        
        shard.put(tokens, kv_tensors, size_bytes)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get stats from all shards."""
        self.initialize()
        
        all_stats = []
        
        try:
            import ray
            if ray.is_initialized():
                stats_futures = [shard.get_stats.remote() for shard in self.shards]
                all_stats = ray.get(stats_futures)
        except:
            for shard in self.shards:
                stats = shard.get_stats()
                all_stats.append({
                    "hits": stats.hits,
                    "misses": stats.misses,
                    "hit_rate": stats.hit_rate,
                    "memory_used": stats.memory_used_bytes,
                })
        
        total_hits = sum(s.get("hits", 0) for s in all_stats)
        total_misses = sum(s.get("misses", 0) for s in all_stats)
        total_memory = sum(s.get("memory_used", 0) for s in all_stats)
        
        return {
            "shards": all_stats,
            "total_hits": total_hits,
            "total_misses": total_misses,
            "overall_hit_rate": total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0,
            "total_memory_used": total_memory,
        }
