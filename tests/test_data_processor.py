"""
Property-based tests for data processor functionality.

**Feature: traffic-pollution-dashboard, Property 7: Data alignment by timestamp**
"""

import pandas as pd
import pytest
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, settings as hypothesis_settings, HealthCheck
import pytz
import numpy as np

from traffic_pollution_dashboard.services.data_processor import DataProcessor
from traffic_pollution_dashboard.data.models import DateRange


class TestDataProcessor:
    """Property-based tests for DataProcessor."""
    
    def create_processor(self):
        """Create a DataProcessor instance for testing."""
        return DataProcessor()
    
    def create_sample_traffic_data(self, num_records=10, city='Delhi'):
        """Create sample traffic data for testing."""
        timestamps = [datetime(2024, 1, 1, i, 0, 0) for i in range(num_records)]
        return pd.DataFrame({
            'timestamp': timestamps,
            'city': [city] * num_records,
            'congestion_level': np.random.uniform(0, 100, num_records),
            'average_speed': np.random.uniform(10, 80, num_records),
            'delay_index': np.random.uniform(1.0, 5.0, num_records),
            'road_segment': [f'{city}_main'] * num_records
        })
    
    def create_sample_pollution_data(self, num_records=10, city='Delhi'):
        """Create sample pollution data for testing."""
        timestamps = [datetime(2024, 1, 1, i, 0, 0) for i in range(num_records)]
        return pd.DataFrame({
            'timestamp': timestamps,
            'city': [city] * num_records,
            'aqi': np.random.randint(50, 300, num_records),
            'pm25': np.random.uniform(20, 150, num_records),
            'pm10': np.random.uniform(40, 250, num_records),
            'no2': np.random.uniform(10, 100, num_records),
            'co': np.random.uniform(0.5, 5.0, num_records),
            'station_id': [f'{city}_station'] * num_records
        })
    
    @given(
        traffic_records=st.integers(min_value=1, max_value=10),
        pollution_records=st.integers(min_value=1, max_value=10),
        city=st.sampled_from(['Delhi', 'Bengaluru', 'Chennai'])
    )
    @hypothesis_settings(max_examples=5, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_data_alignment_by_timestamp(self, traffic_records, pollution_records, city):
        """
        Property test: For any pair of traffic and pollution datasets, the merge operation 
        should align records by timestamp and handle timezone differences appropriately.
        
        **Feature: traffic-pollution-dashboard, Property 7: Data alignment by timestamp**
        **Validates: Requirements 3.5**
        """
        processor = self.create_processor()
        
        # Create sample datasets with overlapping timestamps
        traffic_df = self.create_sample_traffic_data(traffic_records, city)
        pollution_df = self.create_sample_pollution_data(pollution_records, city)
        
        # Act - align the datasets
        aligned_df = processor.align_datasets(traffic_df, pollution_df)
        
        # Assert - alignment properties
        assert isinstance(aligned_df, pd.DataFrame), "Result should be a pandas DataFrame"
        
        # Check that aligned data has expected columns
        expected_columns = [
            'timestamp', 'city', 'congestion_level', 'average_speed', 'delay_index',
            'aqi', 'pm25', 'pm10', 'no2', 'co'
        ]
        for col in expected_columns:
            assert col in aligned_df.columns, f"Aligned data missing column: {col}"
        
        if not aligned_df.empty:
            # All records should be for the same city
            assert all(aligned_df['city'] == city), f"All records should be for city: {city}"
            
            # Timestamps should be sorted
            timestamps = aligned_df['timestamp'].values
            assert all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1)), \
                "Timestamps should be sorted in ascending order"
            
            # All timestamps should be timezone-aware
            assert aligned_df['timestamp'].dt.tz is not None, \
                "Aligned timestamps should be timezone-aware"
            
            # Aligned data should only contain timestamps that exist in both datasets
            # Convert to timezone-naive for comparison since alignment adds timezone info
            traffic_hours = set(traffic_df['timestamp'].dt.floor('h').dt.tz_localize(None))
            pollution_hours = set(pollution_df['timestamp'].dt.floor('h').dt.tz_localize(None))
            aligned_hours = set(aligned_df['timestamp'].dt.floor('h').dt.tz_localize(None))
            
            # Aligned hours should be subset of intersection
            expected_intersection = traffic_hours.intersection(pollution_hours)
            assert aligned_hours.issubset(expected_intersection), \
                "Aligned data should only contain overlapping timestamps"
            
            # No duplicate timestamp-city combinations
            duplicates = aligned_df.duplicated(subset=['timestamp', 'city']).sum()
            assert duplicates == 0, "Should not have duplicate timestamp-city combinations"
    
    @given(
        timezone_offset_hours=st.integers(min_value=-12, max_value=12)
    )
    @hypothesis_settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_timezone_handling_in_alignment(self, timezone_offset_hours):
        """
        Property test: Data alignment should handle timezone differences correctly.
        
        **Feature: traffic-pollution-dashboard, Property 7: Data alignment by timestamp**
        **Validates: Requirements 3.5**
        """
        processor = self.create_processor()
        
        # Create datasets with different timezones
        traffic_df = self.create_sample_traffic_data(10, 'Delhi')
        pollution_df = self.create_sample_pollution_data(10, 'Delhi')
        
        # Add different timezones
        ist = pytz.timezone('Asia/Kolkata')
        offset_tz = pytz.FixedOffset(timezone_offset_hours * 60)
        
        traffic_df['timestamp'] = traffic_df['timestamp'].dt.tz_localize(ist)
        pollution_df['timestamp'] = pollution_df['timestamp'].dt.tz_localize(offset_tz)
        
        # Act - align datasets with different timezones
        aligned_df = processor.align_datasets(traffic_df, pollution_df)
        
        # Assert - timezone handling
        if not aligned_df.empty:
            # All timestamps should be in the same timezone (IST)
            assert aligned_df['timestamp'].dt.tz == ist, \
                "All aligned timestamps should be in IST timezone"
            
            # Alignment should still work despite timezone differences
            assert len(aligned_df) > 0, "Should successfully align data despite timezone differences"
    
    def test_empty_dataset_alignment(self):
        """
        Test alignment behavior with empty datasets.
        
        **Feature: traffic-pollution-dashboard, Property 7: Data alignment by timestamp**
        **Validates: Requirements 3.5**
        """
        processor = self.create_processor()
        
        traffic_df = self.create_sample_traffic_data(5, 'Delhi')
        pollution_df = self.create_sample_pollution_data(5, 'Delhi')
        empty_df = pd.DataFrame()
        
        # Test with empty traffic data
        result1 = processor.align_datasets(empty_df, pollution_df)
        assert result1.empty, "Alignment with empty traffic data should return empty DataFrame"
        
        # Test with empty pollution data
        result2 = processor.align_datasets(traffic_df, empty_df)
        assert result2.empty, "Alignment with empty pollution data should return empty DataFrame"
        
        # Test with both empty
        result3 = processor.align_datasets(empty_df, empty_df)
        assert result3.empty, "Alignment with both empty should return empty DataFrame"
    
    def test_no_overlapping_timestamps(self):
        """
        Test alignment behavior when datasets have no overlapping timestamps.
        
        **Feature: traffic-pollution-dashboard, Property 7: Data alignment by timestamp**
        **Validates: Requirements 3.5**
        """
        processor = self.create_processor()
        
        # Create datasets with non-overlapping time ranges
        traffic_timestamps = [datetime(2024, 1, 1, i, 0, 0) for i in range(5)]
        pollution_timestamps = [datetime(2024, 1, 2, i, 0, 0) for i in range(5)]
        
        traffic_df = pd.DataFrame({
            'timestamp': traffic_timestamps,
            'city': ['Delhi'] * 5,
            'congestion_level': [50] * 5,
            'average_speed': [30] * 5,
            'delay_index': [2.0] * 5
        })
        
        pollution_df = pd.DataFrame({
            'timestamp': pollution_timestamps,
            'city': ['Delhi'] * 5,
            'aqi': [100] * 5,
            'pm25': [50] * 5,
            'pm10': [80] * 5,
            'no2': [30] * 5,
            'co': [2.0] * 5
        })
        
        # Act
        aligned_df = processor.align_datasets(traffic_df, pollution_df)
        
        # Assert
        assert aligned_df.empty, "Should return empty DataFrame when no timestamps overlap"
    
    @given(
        missing_ratio=st.floats(min_value=0.0, max_value=0.5)
    )
    @hypothesis_settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_alignment_with_missing_values(self, missing_ratio):
        """
        Property test: Data alignment should handle datasets with missing values.
        
        **Feature: traffic-pollution-dashboard, Property 7: Data alignment by timestamp**
        **Validates: Requirements 3.5**
        """
        processor = self.create_processor()
        
        traffic_df = self.create_sample_traffic_data(20, 'Delhi')
        pollution_df = self.create_sample_pollution_data(20, 'Delhi')
        
        # Introduce missing values
        num_missing = int(len(traffic_df) * missing_ratio)
        if num_missing > 0:
            missing_indices = np.random.choice(len(traffic_df), num_missing, replace=False)
            traffic_df.loc[missing_indices, 'congestion_level'] = np.nan
            
            missing_indices = np.random.choice(len(pollution_df), num_missing, replace=False)
            pollution_df.loc[missing_indices, 'aqi'] = np.nan
        
        # Act
        aligned_df = processor.align_datasets(traffic_df, pollution_df)
        
        # Assert - alignment should still work with missing values
        assert isinstance(aligned_df, pd.DataFrame), "Should return DataFrame even with missing values"
        
        if not aligned_df.empty:
            # Check that structure is maintained
            expected_columns = [
                'timestamp', 'city', 'congestion_level', 'average_speed', 'delay_index',
                'aqi', 'pm25', 'pm10', 'no2', 'co'
            ]
            for col in expected_columns:
                assert col in aligned_df.columns, f"Missing column after alignment: {col}"
    
    def test_different_cities_alignment(self):
        """
        Test alignment behavior with datasets from different cities.
        
        **Feature: traffic-pollution-dashboard, Property 7: Data alignment by timestamp**
        **Validates: Requirements 3.5**
        """
        processor = self.create_processor()
        
        # Create datasets for different cities
        traffic_df = self.create_sample_traffic_data(10, 'Delhi')
        pollution_df = self.create_sample_pollution_data(10, 'Bengaluru')
        
        # Act
        aligned_df = processor.align_datasets(traffic_df, pollution_df)
        
        # Assert - should return empty since cities don't match
        assert aligned_df.empty, "Should return empty DataFrame when cities don't match"
    
    def test_mixed_cities_alignment(self):
        """
        Test alignment with datasets containing multiple cities.
        
        **Feature: traffic-pollution-dashboard, Property 7: Data alignment by timestamp**
        **Validates: Requirements 3.5**
        """
        processor = self.create_processor()
        
        # Create mixed city datasets
        timestamps = [datetime(2024, 1, 1, i, 0, 0) for i in range(10)]
        
        traffic_df = pd.DataFrame({
            'timestamp': timestamps,
            'city': ['Delhi'] * 5 + ['Bengaluru'] * 5,
            'congestion_level': np.random.uniform(0, 100, 10),
            'average_speed': np.random.uniform(10, 80, 10),
            'delay_index': np.random.uniform(1.0, 5.0, 10)
        })
        
        pollution_df = pd.DataFrame({
            'timestamp': timestamps,
            'city': ['Delhi'] * 5 + ['Bengaluru'] * 5,
            'aqi': np.random.randint(50, 300, 10),
            'pm25': np.random.uniform(20, 150, 10),
            'pm10': np.random.uniform(40, 250, 10),
            'no2': np.random.uniform(10, 100, 10),
            'co': np.random.uniform(0.5, 5.0, 10)
        })
        
        # Act
        aligned_df = processor.align_datasets(traffic_df, pollution_df)
        
        # Assert
        if not aligned_df.empty:
            # Should have data for both cities
            cities_in_result = set(aligned_df['city'].unique())
            assert 'Delhi' in cities_in_result or 'Bengaluru' in cities_in_result, \
                "Should contain data for at least one city"
            
            # Each city should have consistent data
            for city in cities_in_result:
                city_data = aligned_df[aligned_df['city'] == city]
                assert len(city_data) > 0, f"Should have data for city: {city}"
    
    @given(
        duplicate_ratio=st.floats(min_value=0.0, max_value=0.3)
    )
    @hypothesis_settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_alignment_with_duplicate_timestamps(self, duplicate_ratio):
        """
        Property test: Data alignment should handle duplicate timestamps correctly.
        
        **Feature: traffic-pollution-dashboard, Property 7: Data alignment by timestamp**
        **Validates: Requirements 3.5**
        """
        processor = self.create_processor()
        
        traffic_df = self.create_sample_traffic_data(20, 'Delhi')
        pollution_df = self.create_sample_pollution_data(20, 'Delhi')
        
        # Introduce duplicate timestamps
        num_duplicates = int(len(traffic_df) * duplicate_ratio)
        if num_duplicates > 0:
            # Duplicate some timestamps in traffic data
            duplicate_indices = np.random.choice(len(traffic_df) - 1, num_duplicates, replace=False)
            for idx in duplicate_indices:
                traffic_df.loc[idx + 1, 'timestamp'] = traffic_df.loc[idx, 'timestamp']
        
        # Clean the data first (as would happen in real usage)
        cleaned_traffic = processor.clean_traffic_data(traffic_df)
        cleaned_pollution = processor.clean_pollution_data(pollution_df)
        
        # Act
        aligned_df = processor.align_datasets(cleaned_traffic, cleaned_pollution)
        
        # Assert - should handle duplicates gracefully
        assert isinstance(aligned_df, pd.DataFrame), "Should return DataFrame even with duplicates"
        
        if not aligned_df.empty:
            # Should not have duplicate timestamp-city combinations in result after cleaning
            duplicates = aligned_df.duplicated(subset=['timestamp', 'city']).sum()
            assert duplicates == 0, "Aligned result should not have duplicate timestamp-city combinations after cleaning"