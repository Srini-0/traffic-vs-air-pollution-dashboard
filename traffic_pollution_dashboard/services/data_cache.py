"""
Data caching service for improving performance and reducing API calls.

This module provides the DataCache class that implements in-memory caching
with TTL (Time To Live) management and cache invalidation strategies.
"""

import logging
import time
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
from threading import Lock

logger = logging.getLogger(__name__)


class DataCache:
    """
    In-memory data cache with TTL management and cache invalidation strategies.
    
    Provides caching for API responses, processed datasets, and computed results
    to minimize redundant API calls and improve dashboard performance.
    """
    
    def __init__(self, default_ttl: int = 900):  # 15 minutes default
        """
        Initialize the data cache.
        
        Args:
            default_ttl: Default time-to-live in seconds
        """
        self.default_ttl = default_ttl
        self._cache: Dict[str, Dict] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = Lock()
        self.logger = logging.getLogger(__name__)
        
        # Cache statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_requests': 0
        }
    
    def get_cached_data(self, key: str) -> Optional[pd.DataFrame]:
        """
        Retrieve cached data by key.
        
        Args:
            key: Cache key
            
        Returns:
            Cached DataFrame if found and not expired, None otherwise
        """
        with self._lock:
            self.stats['total_requests'] += 1
            
            if key not in self._cache:
                self.stats['misses'] += 1
                logger.debug(f"Cache miss for key: {key}")
                return None
            
            cache_entry = self._cache[key]
            current_time = time.time()
            
            # Check if entry has expired
            if current_time > cache_entry['expires_at']:
                logger.debug(f"Cache entry expired for key: {key}")
                self._remove_entry(key)
                self.stats['misses'] += 1
                return None
            
            # Update access time
            self._access_times[key] = current_time
            self.stats['hits'] += 1
            
            logger.debug(f"Cache hit for key: {key}")
            return cache_entry['data'].copy()  # Return copy to prevent modification
    
    def store_data(self, key: str, data: pd.DataFrame, ttl: Optional[int] = None) -> None:
        """
        Store data in cache with specified TTL.
        
        Args:
            key: Cache key
            data: DataFrame to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        if data is None or data.empty:
            logger.warning(f"Attempting to cache empty data for key: {key}")
            return
        
        ttl = ttl or self.default_ttl
        current_time = time.time()
        expires_at = current_time + ttl
        
        with self._lock:
            # Store data copy to prevent external modifications
            self._cache[key] = {
                'data': data.copy(),
                'created_at': current_time,
                'expires_at': expires_at,
                'ttl': ttl,
                'size_bytes': self._estimate_dataframe_size(data)
            }
            self._access_times[key] = current_time
            
            logger.info(f"Cached data for key: {key}, TTL: {ttl}s, Size: {len(data)} records")
            
            # Cleanup expired entries periodically
            self._cleanup_expired_entries()
    
    def invalidate_cache(self, pattern: str) -> int:
        """
        Invalidate cache entries matching a pattern.
        
        Args:
            pattern: Pattern to match against cache keys
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            keys_to_remove = [key for key in self._cache.keys() if pattern in key]
            
            for key in keys_to_remove:
                self._remove_entry(key)
                self.stats['evictions'] += 1
            
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries matching pattern: {pattern}")
            return len(keys_to_remove)
    
    def clear_cache(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            evicted_count = len(self._cache)
            self._cache.clear()
            self._access_times.clear()
            self.stats['evictions'] += evicted_count
            
            logger.info(f"Cleared all cache entries: {evicted_count} entries removed")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_size_bytes = sum(entry['size_bytes'] for entry in self._cache.values())
            hit_rate = (self.stats['hits'] / self.stats['total_requests'] 
                       if self.stats['total_requests'] > 0 else 0)
            
            return {
                'entries': len(self._cache),
                'total_size_bytes': total_size_bytes,
                'total_size_mb': total_size_bytes / (1024 * 1024),
                'hit_rate': hit_rate,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'total_requests': self.stats['total_requests']
            }
    
    def get_cache_info(self) -> List[Dict]:
        """
        Get information about all cached entries.
        
        Returns:
            List of dictionaries with cache entry information
        """
        with self._lock:
            current_time = time.time()
            cache_info = []
            
            for key, entry in self._cache.items():
                cache_info.append({
                    'key': key,
                    'created_at': datetime.fromtimestamp(entry['created_at']).isoformat(),
                    'expires_at': datetime.fromtimestamp(entry['expires_at']).isoformat(),
                    'ttl': entry['ttl'],
                    'size_bytes': entry['size_bytes'],
                    'records': len(entry['data']),
                    'time_to_expiry': max(0, entry['expires_at'] - current_time),
                    'last_accessed': datetime.fromtimestamp(self._access_times[key]).isoformat()
                })
            
            return sorted(cache_info, key=lambda x: x['last_accessed'], reverse=True)
    
    def generate_cache_key(self, *args, **kwargs) -> str:
        """
        Generate a cache key from arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            Generated cache key
        """
        # Create a string representation of all arguments
        key_parts = []
        
        for arg in args:
            if isinstance(arg, (str, int, float, bool)):
                key_parts.append(str(arg))
            elif isinstance(arg, datetime):
                key_parts.append(arg.isoformat())
            else:
                key_parts.append(str(type(arg).__name__))
        
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (str, int, float, bool)):
                key_parts.append(f"{k}={v}")
            elif isinstance(v, datetime):
                key_parts.append(f"{k}={v.isoformat()}")
            else:
                key_parts.append(f"{k}={type(v).__name__}")
        
        # Create hash of the key parts
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _remove_entry(self, key: str) -> None:
        """Remove a cache entry and its access time."""
        if key in self._cache:
            del self._cache[key]
        if key in self._access_times:
            del self._access_times[key]
    
    def _cleanup_expired_entries(self) -> None:
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, entry in self._cache.items()
            if current_time > entry['expires_at']
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
            self.stats['evictions'] += 1
        
        if expired_keys:
            logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def _estimate_dataframe_size(self, df: pd.DataFrame) -> int:
        """
        Estimate the memory size of a DataFrame in bytes.
        
        Args:
            df: DataFrame to estimate
            
        Returns:
            Estimated size in bytes
        """
        try:
            return df.memory_usage(deep=True).sum()
        except Exception:
            # Fallback estimation
            return len(df) * len(df.columns) * 8  # Rough estimate


class CacheManager:
    """
    Manager for multiple cache instances with different configurations.
    
    Provides separate caches for different data types with appropriate TTL settings.
    """
    
    def __init__(self):
        """Initialize cache manager with different cache instances."""
        self.caches = {
            'traffic_data': DataCache(default_ttl=900),      # 15 minutes
            'pollution_data': DataCache(default_ttl=900),    # 15 minutes
            'processed_data': DataCache(default_ttl=1800),   # 30 minutes
            'correlation_results': DataCache(default_ttl=3600),  # 1 hour
            'api_responses': DataCache(default_ttl=600),     # 10 minutes
        }
        self.logger = logging.getLogger(__name__)
    
    def get_cache(self, cache_type: str) -> DataCache:
        """
        Get a specific cache instance.
        
        Args:
            cache_type: Type of cache to retrieve
            
        Returns:
            DataCache instance
            
        Raises:
            ValueError: If cache type is not recognized
        """
        if cache_type not in self.caches:
            raise ValueError(f"Unknown cache type: {cache_type}. "
                           f"Available types: {list(self.caches.keys())}")
        
        return self.caches[cache_type]
    
    def get_all_stats(self) -> Dict[str, Dict]:
        """
        Get statistics for all cache instances.
        
        Returns:
            Dictionary with statistics for each cache type
        """
        return {
            cache_type: cache.get_cache_stats()
            for cache_type, cache in self.caches.items()
        }
    
    def clear_all_caches(self) -> None:
        """Clear all cache instances."""
        for cache_type, cache in self.caches.items():
            cache.clear_cache()
            self.logger.info(f"Cleared {cache_type} cache")
    
    def invalidate_all_matching(self, pattern: str) -> int:
        """
        Invalidate entries matching pattern across all caches.
        
        Args:
            pattern: Pattern to match
            
        Returns:
            Total number of entries invalidated
        """
        total_invalidated = 0
        for cache in self.caches.values():
            total_invalidated += cache.invalidate_cache(pattern)
        
        return total_invalidated


# Global cache manager instance
cache_manager = CacheManager()


def cached_api_call(cache_type: str, ttl: Optional[int] = None):
    """
    Decorator for caching API call results.
    
    Args:
        cache_type: Type of cache to use
        ttl: Time-to-live override
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            cache = cache_manager.get_cache(cache_type)
            
            # Generate cache key from function name and arguments
            cache_key = f"{func.__name__}_{cache.generate_cache_key(*args, **kwargs)}"
            
            # Try to get from cache first
            cached_result = cache.get_cached_data(cache_key)
            if cached_result is not None:
                logger.debug(f"Using cached result for {func.__name__}")
                return cached_result
            
            # Call function and cache result
            result = func(*args, **kwargs)
            if isinstance(result, pd.DataFrame) and not result.empty:
                cache.store_data(cache_key, result, ttl)
                logger.debug(f"Cached result for {func.__name__}")
            
            return result
        
        return wrapper
    return decorator