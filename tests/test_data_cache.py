"""
Property-based tests for data caching functionality.

**Feature: traffic-pollution-dashboard, Property 11: Caching behavior for large datasets**
"""

import pandas as pd
import pytest
import time
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, settings as hypothesis_settings, HealthCheck
import numpy as np

from traffic_pollution_dashboard.services.data_cache import DataCache, CacheManager, cache_manager
from traffic_pollution_dashboard.data.traffic_client import TrafficDataClient
from traffic_pollution_dashboard.data.pollution_client import PollutionDataClient
from traffic_pollution_dashboard.data.models import DateRange


class TestDataCache:
    """Property-based tests for DataCache."""
    
    def create_cache(self, ttl=60):
        """Create a DataCache instance for testing."""
        return DataCache(default_ttl=ttl)
    
    def create_sample_dataframe(self, rows=100, cols=5):
        """Create a sample DataFrame for testing."""
        data = {}
        for i in range(cols):
            data[f'col_{i}'] = np.random.randn(rows)
        return pd.DataFrame(data)
    
    @given(
        data_size=st.integers(min_value=10, max_value=1000),
        ttl=st.integers(min_value=1, max_value=300)
    )
    @hypothesis_settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_caching_behavior_for_large_datasets(self, data_size, ttl):
        """
        Property test: For any large dataset processing operation, subsequent identical 
        requests should use cached data and avoid redundant API calls.
        
        **Feature: traffic-pollution-dashboard, Property 11: Caching behavior for large datasets**
        **Validates: Requirements 7.3**
        """
        cache = self.create_cache(ttl)
        
        # Create test data
        test_data = self.create_sample_dataframe(data_size, 10)
        cache_key = f"test_data_{data_size}"
        
        # First store operation
        cache.store_data(cache_key, test_data, ttl)
        
        # Retrieve data
        retrieved_data = cache.get_cached_data(cache_key)
        
        # Assert caching behavior
        assert retrieved_data is not None, "Should retrieve cached data"
        assert isinstance(retrieved_data, pd.DataFrame), "Should return DataFrame"
        assert len(retrieved_data) == data_size, f"Should have {data_size} rows"
        assert test_data.equals(retrieved_data), "Retrieved data should match original"
        
        # Check cache statistics
        stats = cache.get_cache_stats()
        assert stats['entries'] >= 1, "Should have at least one cache entry"
        assert stats['hits'] >= 1, "Should have at least one cache hit"
        assert stats['hit_rate'] > 0, "Hit rate should be positive"
        
        # Test cache key generation consistency
        key1 = cache.generate_cache_key("test", data_size, ttl=ttl)
        key2 = cache.generate_cache_key("test", data_size, ttl=ttl)
        assert key1 == key2, "Same arguments should generate same cache key"
        
        # Test different arguments generate different keys
        key3 = cache.generate_cache_key("test", data_size + 1, ttl=ttl)
        assert key1 != key3, "Different arguments should generate different cache keys"
    
    @given(
        num_entries=st.integers(min_value=5, max_value=50)
    )
    @hypothesis_settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_cache_invalidation_patterns(self, num_entries):
        """
        Property test: Cache invalidation should work correctly with pattern matching.
        
        **Feature: traffic-pollution-dashboard, Property 11: Caching behavior for large datasets**
        **Validates: Requirements 7.3**
        """
        cache = self.create_cache()
        
        # Store multiple entries with different patterns
        for i in range(num_entries):
            data = self.create_sample_dataframe(10, 3)
            if i % 2 == 0:
                key = f"traffic_data_{i}"
            else:
                key = f"pollution_data_{i}"
            cache.store_data(key, data)
        
        initial_entries = cache.get_cache_stats()['entries']
        assert initial_entries == num_entries, f"Should have {num_entries} entries"
        
        # Invalidate traffic data entries
        invalidated = cache.invalidate_cache("traffic_data")
        expected_invalidated = (num_entries + 1) // 2  # Half rounded up
        
        assert invalidated == expected_invalidated, \
            f"Should invalidate {expected_invalidated} traffic entries"
        
        remaining_entries = cache.get_cache_stats()['entries']
        expected_remaining = num_entries - expected_invalidated
        assert remaining_entries == expected_remaining, \
            f"Should have {expected_remaining} entries remaining"
        
        # Verify pollution data entries still exist
        for i in range(1, num_entries, 2):  # Odd indices (pollution data)
            key = f"pollution_data_{i}"
            data = cache.get_cached_data(key)
            assert data is not None, f"Pollution data entry {i} should still exist"
    
    def test_cache_expiration_behavior(self):
        """
        Test that cache entries expire correctly based on TTL.
        
        **Feature: traffic-pollution-dashboard, Property 11: Caching behavior for large datasets**
        **Validates: Requirements 7.3**
        """
        cache = self.create_cache(ttl=1)  # 1 second TTL
        
        test_data = self.create_sample_dataframe(50, 5)
        cache.store_data("short_ttl_data", test_data, ttl=1)
        
        # Should be available immediately
        retrieved = cache.get_cached_data("short_ttl_data")
        assert retrieved is not None, "Data should be available immediately"
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Should be expired now
        expired_data = cache.get_cached_data("short_ttl_data")
        assert expired_data is None, "Data should be expired after TTL"
        
        # Check that expired entries are cleaned up
        stats = cache.get_cache_stats()
        assert stats['entries'] == 0, "Expired entries should be cleaned up"
    
    @given(
        cache_operations=st.integers(min_value=5, max_value=20)
    )
    @hypothesis_settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_cache_performance_under_load(self, cache_operations):
        """
        Property test: Cache should maintain performance under multiple operations.
        
        **Feature: traffic-pollution-dashboard, Property 11: Caching behavior for large datasets**
        **Validates: Requirements 7.3**
        """
        cache = self.create_cache()
        
        # Perform multiple cache operations
        start_time = time.time()
        
        for i in range(cache_operations):
            data = self.create_sample_dataframe(20, 5)
            key = f"perf_test_{i}"
            
            # Store data
            cache.store_data(key, data)
            
            # Retrieve data
            retrieved = cache.get_cached_data(key)
            assert retrieved is not None, f"Should retrieve data for operation {i}"
        
        total_time = time.time() - start_time
        avg_time_per_op = total_time / (cache_operations * 2)  # 2 ops per iteration
        
        # Performance assertion - should complete operations quickly
        assert avg_time_per_op < 0.1, f"Average operation time too slow: {avg_time_per_op:.4f}s"
        
        # Check final statistics
        stats = cache.get_cache_stats()
        assert stats['entries'] == cache_operations, f"Should have {cache_operations} entries"
        assert stats['hits'] == cache_operations, f"Should have {cache_operations} hits"
        # Total requests should be at least cache_operations (one for each retrieve)
        assert stats['total_requests'] >= cache_operations, "Should track retrieve requests"
    
    def test_cache_thread_safety(self):
        """
        Test cache thread safety with concurrent operations.
        
        **Feature: traffic-pollution-dashboard, Property 11: Caching behavior for large datasets**
        **Validates: Requirements 7.3**
        """
        import threading
        
        cache = self.create_cache()
        results = []
        errors = []
        
        def cache_worker(worker_id):
            try:
                for i in range(10):
                    data = self.create_sample_dataframe(10, 3)
                    key = f"worker_{worker_id}_data_{i}"
                    
                    # Store and retrieve
                    cache.store_data(key, data)
                    retrieved = cache.get_cached_data(key)
                    
                    results.append((worker_id, i, retrieved is not None))
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Create and start multiple threads
        threads = []
        for worker_id in range(5):
            thread = threading.Thread(target=cache_worker, args=(worker_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Should have no errors, got: {errors}"
        assert len(results) == 50, "Should have 50 successful operations"
        assert all(success for _, _, success in results), "All operations should succeed"
        
        # Check final cache state
        stats = cache.get_cache_stats()
        assert stats['entries'] == 50, "Should have 50 entries from all workers"


class TestCacheManager:
    """Tests for CacheManager functionality."""
    
    def test_cache_manager_multiple_caches(self):
        """
        Test that cache manager handles multiple cache types correctly.
        
        **Feature: traffic-pollution-dashboard, Property 11: Caching behavior for large datasets**
        **Validates: Requirements 7.3**
        """
        manager = CacheManager()
        
        # Test getting different cache types
        traffic_cache = manager.get_cache('traffic_data')
        pollution_cache = manager.get_cache('pollution_data')
        
        assert isinstance(traffic_cache, DataCache), "Should return DataCache instance"
        assert isinstance(pollution_cache, DataCache), "Should return DataCache instance"
        assert traffic_cache is not pollution_cache, "Should return different instances"
        
        # Test storing data in different caches
        traffic_data = pd.DataFrame({'traffic': [1, 2, 3]})
        pollution_data = pd.DataFrame({'pollution': [4, 5, 6]})
        
        traffic_cache.store_data('test_traffic', traffic_data)
        pollution_cache.store_data('test_pollution', pollution_data)
        
        # Verify data isolation
        assert traffic_cache.get_cached_data('test_traffic') is not None
        assert traffic_cache.get_cached_data('test_pollution') is None
        assert pollution_cache.get_cached_data('test_pollution') is not None
        assert pollution_cache.get_cached_data('test_traffic') is None
        
        # Test global operations
        all_stats = manager.get_all_stats()
        assert 'traffic_data' in all_stats
        assert 'pollution_data' in all_stats
        assert all_stats['traffic_data']['entries'] == 1
        assert all_stats['pollution_data']['entries'] == 1
    
    def test_invalid_cache_type(self):
        """Test error handling for invalid cache types."""
        manager = CacheManager()
        
        with pytest.raises(ValueError, match="Unknown cache type"):
            manager.get_cache('invalid_cache_type')


class TestCachedAPIIntegration:
    """Tests for cached API integration."""
    
    def create_date_range(self):
        """Create a test date range."""
        start_date = datetime(2024, 1, 1, 0, 0, 0)
        end_date = datetime(2024, 1, 1, 6, 0, 0)
        return DateRange(start_date=start_date, end_date=end_date)
    
    def test_cached_traffic_api_calls(self):
        """
        Test that traffic API calls are properly cached.
        
        **Feature: traffic-pollution-dashboard, Property 11: Caching behavior for large datasets**
        **Validates: Requirements 7.3**
        """
        client = TrafficDataClient()
        date_range = self.create_date_range()
        
        # Clear cache to start fresh
        traffic_cache = cache_manager.get_cache('traffic_data')
        traffic_cache.clear_cache()
        
        # First call
        start_time = time.time()
        df1 = client.fetch_traffic_data('Delhi', date_range)
        first_call_time = time.time() - start_time
        
        # Second call (should be cached)
        start_time = time.time()
        df2 = client.fetch_traffic_data('Delhi', date_range)
        second_call_time = time.time() - start_time
        
        # Assertions
        assert df1.equals(df2), "Cached data should match original"
        assert second_call_time < first_call_time, "Cached call should be faster"
        
        # Check cache statistics
        stats = traffic_cache.get_cache_stats()
        assert stats['hits'] >= 1, "Should have at least one cache hit"
        assert stats['entries'] >= 1, "Should have at least one cache entry"
    
    def test_cached_pollution_api_calls(self):
        """
        Test that pollution API calls are properly cached.
        
        **Feature: traffic-pollution-dashboard, Property 11: Caching behavior for large datasets**
        **Validates: Requirements 7.3**
        """
        client = PollutionDataClient()
        date_range = self.create_date_range()
        
        # Clear cache to start fresh
        pollution_cache = cache_manager.get_cache('pollution_data')
        pollution_cache.clear_cache()
        
        # First call
        start_time = time.time()
        df1 = client.fetch_pollution_data('Delhi', date_range)
        first_call_time = time.time() - start_time
        
        # Second call (should be cached)
        start_time = time.time()
        df2 = client.fetch_pollution_data('Delhi', date_range)
        second_call_time = time.time() - start_time
        
        # Assertions
        assert df1.equals(df2), "Cached data should match original"
        assert second_call_time < first_call_time, "Cached call should be faster"
        
        # Check cache statistics
        stats = pollution_cache.get_cache_stats()
        assert stats['hits'] >= 1, "Should have at least one cache hit"
        assert stats['entries'] >= 1, "Should have at least one cache entry"
    
    def test_cache_isolation_between_cities(self):
        """
        Test that cache properly isolates data between different cities.
        
        **Feature: traffic-pollution-dashboard, Property 11: Caching behavior for large datasets**
        **Validates: Requirements 7.3**
        """
        client = TrafficDataClient()
        date_range = self.create_date_range()
        
        # Clear cache
        traffic_cache = cache_manager.get_cache('traffic_data')
        traffic_cache.clear_cache()
        
        # Fetch data for different cities
        delhi_data = client.fetch_traffic_data('Delhi', date_range)
        bengaluru_data = client.fetch_traffic_data('Bengaluru', date_range)
        
        # Data should be different (different cities have different patterns)
        assert not delhi_data.equals(bengaluru_data), "Different cities should have different data"
        
        # Both should be cached separately
        stats = traffic_cache.get_cache_stats()
        assert stats['entries'] >= 2, "Should cache data for both cities separately"
        
        # Verify cache isolation by fetching again
        delhi_data_2 = client.fetch_traffic_data('Delhi', date_range)
        bengaluru_data_2 = client.fetch_traffic_data('Bengaluru', date_range)
        
        assert delhi_data.equals(delhi_data_2), "Delhi data should be consistent"
        assert bengaluru_data.equals(bengaluru_data_2), "Bengaluru data should be consistent"
        assert not delhi_data_2.equals(bengaluru_data_2), "Cities should remain different"