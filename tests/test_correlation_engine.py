"""
Property-based tests for correlation analysis engine.

**Feature: traffic-pollution-dashboard, Property 8: Correlation calculation validity**
"""

import pandas as pd
import pytest
import numpy as np
from datetime import datetime, timedelta
from hypothesis import given, strategies as st, settings as hypothesis_settings, HealthCheck

from traffic_pollution_dashboard.services.correlation_engine import CorrelationEngine, CorrelationResult


class TestCorrelationEngine:
    """Property-based tests for CorrelationEngine."""
    
    def create_engine(self):
        """Create a CorrelationEngine instance for testing."""
        return CorrelationEngine()
    
    @given(
        data_size=st.integers(min_value=10, max_value=100),
        correlation_strength=st.floats(min_value=-0.9, max_value=0.9)
    )
    @hypothesis_settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_correlation_calculation_validity(self, data_size, correlation_strength):
        """
        Property test: For any valid traffic and pollution dataset pair, the Pearson 
        correlation coefficient should be calculated and fall within the valid range [-1, 1].
        
        **Feature: traffic-pollution-dashboard, Property 8: Correlation calculation validity**
        **Validates: Requirements 4.1**
        """
        engine = self.create_engine()
        
        # Generate correlated data
        np.random.seed(42)  # For reproducible tests
        
        # Generate base traffic data
        traffic_data = np.random.normal(50, 15, data_size)
        
        # Generate pollution data with specified correlation
        noise = np.random.normal(0, 1, data_size)
        pollution_data = (correlation_strength * traffic_data + 
                         np.sqrt(1 - correlation_strength**2) * noise * 50 + 150)
        
        # Convert to pandas Series
        traffic_series = pd.Series(traffic_data)
        pollution_series = pd.Series(pollution_data)
        
        # Calculate correlation
        result = engine.calculate_pearson_correlation(traffic_series, pollution_series)
        
        # Assert correlation validity
        assert isinstance(result, CorrelationResult), "Should return CorrelationResult object"
        assert -1.0 <= result.correlation <= 1.0, f"Correlation should be in [-1, 1], got {result.correlation}"
        assert isinstance(result.p_value, float), "P-value should be a float"
        assert 0.0 <= result.p_value <= 1.0, f"P-value should be in [0, 1], got {result.p_value}"
        assert result.sample_size == data_size, f"Sample size should be {data_size}"
        assert len(result.confidence_interval) == 2, "Confidence interval should have 2 values"
        assert result.confidence_interval[0] <= result.correlation <= result.confidence_interval[1], \
            "Correlation should be within confidence interval"
        
        # Check correlation strength categorization
        abs_corr = abs(result.correlation)
        if abs_corr >= 0.7:
            assert result.strength == 'strong'
        elif abs_corr >= 0.3:
            assert result.strength == 'moderate'
        elif abs_corr >= 0.1:
            assert result.strength == 'weak'
        else:
            assert result.strength == 'negligible'
        
        # Check significance determination
        assert result.is_significant == (result.p_value < 0.05)
    
    @given(
        data_size=st.integers(min_value=5, max_value=50)
    )
    @hypothesis_settings(max_examples=15, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_spearman_correlation_validity(self, data_size):
        """
        Property test: Spearman correlation should also be valid for any dataset.
        
        **Feature: traffic-pollution-dashboard, Property 8: Correlation calculation validity**
        **Validates: Requirements 4.1**
        """
        engine = self.create_engine()
        
        # Generate monotonic relationship (good for Spearman)
        traffic_data = np.sort(np.random.uniform(0, 100, data_size))
        pollution_data = np.sort(np.random.uniform(50, 300, data_size))
        
        traffic_series = pd.Series(traffic_data)
        pollution_series = pd.Series(pollution_data)
        
        result = engine.calculate_spearman_correlation(traffic_series, pollution_series)
        
        # Assert Spearman correlation validity
        assert isinstance(result, CorrelationResult), "Should return CorrelationResult object"
        assert -1.0 <= result.correlation <= 1.0, f"Spearman correlation should be in [-1, 1]"
        assert result.method == 'spearman', "Method should be 'spearman'"
        assert result.sample_size == data_size, f"Sample size should be {data_size}"
    
    def test_perfect_correlations(self):
        """
        Test perfect positive and negative correlations.
        
        **Feature: traffic-pollution-dashboard, Property 8: Correlation calculation validity**
        **Validates: Requirements 4.1**
        """
        engine = self.create_engine()
        
        # Perfect positive correlation
        x = pd.Series([1, 2, 3, 4, 5])
        y_positive = pd.Series([2, 4, 6, 8, 10])  # y = 2x
        
        result_positive = engine.calculate_pearson_correlation(x, y_positive)
        assert abs(result_positive.correlation - 1.0) < 0.001, "Should be perfect positive correlation"
        
        # Perfect negative correlation
        y_negative = pd.Series([10, 8, 6, 4, 2])  # y = -2x + 12
        
        result_negative = engine.calculate_pearson_correlation(x, y_negative)
        assert abs(result_negative.correlation - (-1.0)) < 0.001, "Should be perfect negative correlation"
        
        # No correlation (constant)
        y_constant = pd.Series([5, 5, 5, 5, 5])
        
        result_none = engine.calculate_pearson_correlation(x, y_constant)
        assert abs(result_none.correlation) < 0.001, "Should be no correlation with constant"
    
    def test_correlation_with_missing_values(self):
        """
        Test correlation calculation with missing values.
        
        **Feature: traffic-pollution-dashboard, Property 8: Correlation calculation validity**
        **Validates: Requirements 4.1**
        """
        engine = self.create_engine()
        
        # Data with NaN values
        traffic_data = pd.Series([1, 2, np.nan, 4, 5, np.nan, 7, 8])
        pollution_data = pd.Series([10, 20, 30, np.nan, 50, 60, np.nan, 80])
        
        result = engine.calculate_pearson_correlation(traffic_data, pollution_data)
        
        # Should handle NaN values by removing them
        assert isinstance(result, CorrelationResult), "Should handle NaN values"
        assert result.sample_size < len(traffic_data), "Sample size should be reduced after NaN removal"
        assert -1.0 <= result.correlation <= 1.0, "Correlation should still be valid"
    
    def test_insufficient_data_error(self):
        """
        Test error handling with insufficient data points.
        
        **Feature: traffic-pollution-dashboard, Property 8: Correlation calculation validity**
        **Validates: Requirements 4.1**
        """
        engine = self.create_engine()
        
        # Too few data points
        traffic_data = pd.Series([1, 2])
        pollution_data = pd.Series([10, 20])
        
        with pytest.raises(ValueError, match="Insufficient data points"):
            engine.calculate_pearson_correlation(traffic_data, pollution_data)
        
        # Empty series
        empty_traffic = pd.Series([])
        empty_pollution = pd.Series([])
        
        with pytest.raises(ValueError, match="Series cannot be empty"):
            engine.calculate_pearson_correlation(empty_traffic, empty_pollution)
    
    def test_mismatched_series_length(self):
        """
        Test error handling with mismatched series lengths.
        
        **Feature: traffic-pollution-dashboard, Property 8: Correlation calculation validity**
        **Validates: Requirements 4.1**
        """
        engine = self.create_engine()
        
        traffic_data = pd.Series([1, 2, 3, 4, 5])
        pollution_data = pd.Series([10, 20, 30])  # Different length
        
        with pytest.raises(ValueError, match="Series must have the same length"):
            engine.calculate_pearson_correlation(traffic_data, pollution_data)
    
    @given(
        sample_size=st.integers(min_value=10, max_value=100)
    )
    @hypothesis_settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_confidence_interval_properties(self, sample_size):
        """
        Property test: Confidence intervals should have valid properties.
        
        **Feature: traffic-pollution-dashboard, Property 8: Correlation calculation validity**
        **Validates: Requirements 4.1**
        """
        engine = self.create_engine()
        
        # Generate random data
        np.random.seed(42)
        traffic_data = pd.Series(np.random.normal(50, 15, sample_size))
        pollution_data = pd.Series(np.random.normal(150, 30, sample_size))
        
        result = engine.calculate_pearson_correlation(traffic_data, pollution_data)
        
        lower, upper = result.confidence_interval
        
        # Confidence interval properties
        assert lower <= upper, "Lower bound should be <= upper bound"
        assert -1.0 <= lower <= 1.0, "Lower bound should be in [-1, 1]"
        assert -1.0 <= upper <= 1.0, "Upper bound should be in [-1, 1]"
        assert lower <= result.correlation <= upper, "Correlation should be within interval"
        
        # Interval should be narrower for larger samples
        if sample_size >= 50:
            interval_width = upper - lower
            assert interval_width < 1.0, "Confidence interval should be reasonably narrow for large samples"
    
    def test_peak_hour_identification(self):
        """
        Test peak hour identification functionality.
        
        **Feature: traffic-pollution-dashboard, Property 8: Correlation calculation validity**
        **Validates: Requirements 4.1**
        """
        engine = self.create_engine()
        
        # Create traffic data with clear peak hours
        timestamps = [datetime(2024, 1, 1, hour, 0, 0) for hour in range(24)]
        congestion_levels = [
            20, 25, 30, 35, 40,  # 0-4: Low
            50, 60, 70, 80, 70,  # 5-9: Morning peak
            60, 55, 50, 45, 50,  # 10-14: Midday
            55, 60, 75, 85, 80,  # 15-19: Evening peak
            70, 60, 50, 40       # 20-23: Night
        ]
        
        traffic_df = pd.DataFrame({
            'timestamp': timestamps,
            'congestion_level': congestion_levels
        })
        
        peak_hours = engine.identify_peak_hours(traffic_df)
        
        # Should identify morning and evening peaks
        assert isinstance(peak_hours, list), "Should return list of peak hours"
        assert all(0 <= hour <= 23 for hour in peak_hours), "Peak hours should be in valid range"
        assert len(peak_hours) > 0, "Should identify at least some peak hours"
        
        # High congestion hours should be in peak hours
        high_congestion_hours = [7, 8, 17, 18]  # Based on our test data
        overlap = set(peak_hours).intersection(set(high_congestion_hours))
        assert len(overlap) > 0, "Should identify some of the high congestion hours as peaks"
    
    def test_pollution_analysis_during_peaks(self):
        """
        Test pollution analysis during peak hours.
        
        **Feature: traffic-pollution-dashboard, Property 8: Correlation calculation validity**
        **Validates: Requirements 4.1**
        """
        engine = self.create_engine()
        
        # Create pollution data
        timestamps = [datetime(2024, 1, 1, hour, 0, 0) for hour in range(24)]
        aqi_values = [100 + hour * 5 if hour in [8, 9, 17, 18] else 100 + hour * 2 for hour in range(24)]
        
        pollution_df = pd.DataFrame({
            'timestamp': timestamps,
            'aqi': aqi_values,
            'pm25': [val * 0.4 for val in aqi_values],
            'pm10': [val * 0.8 for val in aqi_values]
        })
        
        peak_hours = [8, 9, 17, 18]  # Define peak hours
        
        analysis = engine.analyze_pollution_during_peaks(pollution_df, peak_hours)
        
        # Check analysis results
        assert isinstance(analysis, dict), "Should return dictionary"
        assert 'aqi_peak_avg' in analysis, "Should include AQI peak average"
        assert 'aqi_non_peak_avg' in analysis, "Should include AQI non-peak average"
        assert 'aqi_percent_increase' in analysis, "Should include percentage increase"
        assert 'peak_hours_count' in analysis, "Should include peak hours count"
        
        # Peak average should be higher than non-peak for our test data
        assert analysis['aqi_peak_avg'] > analysis['aqi_non_peak_avg'], \
            "Peak AQI should be higher than non-peak"
        assert analysis['aqi_percent_increase'] > 0, "Should show positive increase during peaks"
    
    def test_comprehensive_analysis_integration(self):
        """
        Test comprehensive analysis with integrated data.
        
        **Feature: traffic-pollution-dashboard, Property 8: Correlation calculation validity**
        **Validates: Requirements 4.1**
        """
        engine = self.create_engine()
        
        # Create aligned dataset
        timestamps = [datetime(2024, 1, 1, hour, 0, 0) for hour in range(24)]
        
        aligned_data = pd.DataFrame({
            'timestamp': timestamps,
            'congestion_level': [50 + 20 * np.sin(hour * np.pi / 12) for hour in range(24)],
            'aqi': [150 + 30 * np.sin(hour * np.pi / 12) + np.random.normal(0, 5) for hour in range(24)],
            'pm25': [60 + 15 * np.sin(hour * np.pi / 12) + np.random.normal(0, 3) for hour in range(24)],
            'pm10': [100 + 25 * np.sin(hour * np.pi / 12) + np.random.normal(0, 5) for hour in range(24)]
        })
        
        results = engine.perform_comprehensive_analysis(aligned_data)
        
        # Check comprehensive results structure
        assert isinstance(results, dict), "Should return dictionary"
        assert 'timestamp' in results, "Should include timestamp"
        assert 'sample_size' in results, "Should include sample size"
        assert 'correlations' in results, "Should include correlations"
        assert 'peak_analysis' in results, "Should include peak analysis"
        assert 'summary_statistics' in results, "Should include summary statistics"
        
        # Check correlations
        assert 'aqi' in results['correlations'], "Should include AQI correlation"
        assert 'pearson' in results['correlations']['aqi'], "Should include Pearson correlation"
        assert 'spearman' in results['correlations']['aqi'], "Should include Spearman correlation"
        
        # Check that correlations are valid
        pearson_corr = results['correlations']['aqi']['pearson']['correlation']
        assert -1.0 <= pearson_corr <= 1.0, "Pearson correlation should be valid"
        
        spearman_corr = results['correlations']['aqi']['spearman']['correlation']
        assert -1.0 <= spearman_corr <= 1.0, "Spearman correlation should be valid"
    
    def test_rolling_correlation_calculation(self):
        """
        Test rolling correlation calculation.
        
        **Feature: traffic-pollution-dashboard, Property 8: Correlation calculation validity**
        **Validates: Requirements 4.1**
        """
        engine = self.create_engine()
        
        # Create time series data
        dates = pd.date_range('2024-01-01', periods=48, freq='h')
        traffic_data = pd.Series(
            [50 + 20 * np.sin(i * np.pi / 12) + np.random.normal(0, 5) for i in range(48)],
            index=dates
        )
        pollution_data = pd.Series(
            [150 + 30 * np.sin(i * np.pi / 12) + np.random.normal(0, 8) for i in range(48)],
            index=dates
        )
        
        rolling_corr = engine.calculate_rolling_correlation(traffic_data, pollution_data, window=12)
        
        # Check rolling correlation properties
        assert isinstance(rolling_corr, pd.Series), "Should return pandas Series"
        assert len(rolling_corr) > 0, "Should have some rolling correlation values"
        assert all(-1.0 <= corr <= 1.0 for corr in rolling_corr.dropna()), \
            "All rolling correlations should be in [-1, 1]"