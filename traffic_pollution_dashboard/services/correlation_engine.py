"""
Correlation analysis engine for calculating statistical relationships between traffic and pollution metrics.

This module provides the CorrelationEngine class that performs statistical analysis
to identify relationships between traffic congestion and air pollution data.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, time
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings

from ..services.data_cache import cached_api_call

logger = logging.getLogger(__name__)


class CorrelationResult:
    """Container for correlation analysis results."""
    
    def __init__(self, correlation: float, p_value: float, sample_size: int, 
                 confidence_interval: Tuple[float, float], method: str = 'pearson'):
        self.correlation = correlation
        self.p_value = p_value
        self.sample_size = sample_size
        self.confidence_interval = confidence_interval
        self.method = method
        self.is_significant = p_value < 0.05
        self.strength = self._categorize_strength()
    
    def _categorize_strength(self) -> str:
        """Categorize correlation strength based on absolute value."""
        abs_corr = abs(self.correlation)
        if abs_corr >= 0.7:
            return 'strong'
        elif abs_corr >= 0.3:
            return 'moderate'
        elif abs_corr >= 0.1:
            return 'weak'
        else:
            return 'negligible'
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'correlation': self.correlation,
            'p_value': self.p_value,
            'sample_size': self.sample_size,
            'confidence_interval': self.confidence_interval,
            'method': self.method,
            'is_significant': self.is_significant,
            'strength': self.strength
        }


class CorrelationEngine:
    """
    Engine for calculating statistical correlations between traffic and pollution metrics.
    
    Provides methods for Pearson correlation, statistical significance testing,
    peak hour analysis, and confidence interval calculation.
    """
    
    def __init__(self):
        """Initialize the correlation engine."""
        self.logger = logging.getLogger(__name__)
    
    def calculate_pearson_correlation(self, traffic_data: pd.Series, 
                                    pollution_data: pd.Series) -> CorrelationResult:
        """
        Calculate Pearson correlation coefficient between traffic and pollution data.
        
        Args:
            traffic_data: Series of traffic congestion values
            pollution_data: Series of pollution values (AQI, PM2.5, etc.)
            
        Returns:
            CorrelationResult with correlation coefficient and statistics
            
        Raises:
            ValueError: If data series are invalid or incompatible
        """
        # Validate input data
        self._validate_series_data(traffic_data, pollution_data)
        
        # Remove any NaN values
        clean_data = self._clean_paired_data(traffic_data, pollution_data)
        if len(clean_data[0]) < 3:
            raise ValueError("Insufficient data points for correlation analysis (minimum 3 required)")
        
        traffic_clean, pollution_clean = clean_data
        
        # Calculate Pearson correlation
        try:
            correlation, p_value = pearsonr(traffic_clean, pollution_clean)
            
            # Handle edge cases
            if np.isnan(correlation):
                correlation = 0.0
                p_value = 1.0
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(
                correlation, len(traffic_clean)
            )
            
            result = CorrelationResult(
                correlation=correlation,
                p_value=p_value,
                sample_size=len(traffic_clean),
                confidence_interval=confidence_interval,
                method='pearson'
            )
            
            logger.info(f"Pearson correlation: {correlation:.3f} (p={p_value:.3f}, n={len(traffic_clean)})")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Pearson correlation: {e}")
            raise ValueError(f"Failed to calculate correlation: {e}")
    
    def calculate_spearman_correlation(self, traffic_data: pd.Series, 
                                     pollution_data: pd.Series) -> CorrelationResult:
        """
        Calculate Spearman rank correlation coefficient.
        
        Args:
            traffic_data: Series of traffic congestion values
            pollution_data: Series of pollution values
            
        Returns:
            CorrelationResult with Spearman correlation statistics
        """
        # Validate input data
        self._validate_series_data(traffic_data, pollution_data)
        
        # Remove any NaN values
        clean_data = self._clean_paired_data(traffic_data, pollution_data)
        if len(clean_data[0]) < 3:
            raise ValueError("Insufficient data points for correlation analysis")
        
        traffic_clean, pollution_clean = clean_data
        
        try:
            correlation, p_value = spearmanr(traffic_clean, pollution_clean)
            
            if np.isnan(correlation):
                correlation = 0.0
                p_value = 1.0
            
            # Confidence interval for Spearman (approximation)
            confidence_interval = self._calculate_confidence_interval(
                correlation, len(traffic_clean), method='spearman'
            )
            
            result = CorrelationResult(
                correlation=correlation,
                p_value=p_value,
                sample_size=len(traffic_clean),
                confidence_interval=confidence_interval,
                method='spearman'
            )
            
            logger.info(f"Spearman correlation: {correlation:.3f} (p={p_value:.3f}, n={len(traffic_clean)})")
            return result
            
        except Exception as e:
            logger.error(f"Error calculating Spearman correlation: {e}")
            raise ValueError(f"Failed to calculate correlation: {e}")
    
    def calculate_significance(self, correlation: float, sample_size: int) -> float:
        """
        Calculate statistical significance of correlation coefficient.
        
        Args:
            correlation: Correlation coefficient
            sample_size: Number of data points
            
        Returns:
            p-value for the correlation
        """
        if sample_size < 3:
            return 1.0  # Not significant with too few points
        
        # Calculate t-statistic
        degrees_freedom = sample_size - 2
        t_stat = correlation * np.sqrt(degrees_freedom / (1 - correlation**2 + 1e-10))
        
        # Calculate two-tailed p-value
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), degrees_freedom))
        
        return p_value
    
    def identify_peak_hours(self, traffic_data: pd.DataFrame) -> List[int]:
        """
        Identify peak traffic hours from traffic data.
        
        Args:
            traffic_data: DataFrame with timestamp and congestion_level columns
            
        Returns:
            List of hours (0-23) that are considered peak traffic hours
        """
        if 'timestamp' not in traffic_data.columns or 'congestion_level' not in traffic_data.columns:
            raise ValueError("Traffic data must contain 'timestamp' and 'congestion_level' columns")
        
        # Extract hour from timestamp
        traffic_data = traffic_data.copy()
        traffic_data['hour'] = traffic_data['timestamp'].dt.hour
        
        # Calculate average congestion by hour
        hourly_avg = traffic_data.groupby('hour')['congestion_level'].mean()
        
        # Define peak hours as those above 75th percentile
        threshold = hourly_avg.quantile(0.75)
        peak_hours = hourly_avg[hourly_avg >= threshold].index.tolist()
        
        logger.info(f"Identified peak hours: {peak_hours} (threshold: {threshold:.1f})")
        return peak_hours
    
    def analyze_pollution_during_peaks(self, pollution_data: pd.DataFrame, 
                                     peak_hours: List[int]) -> Dict[str, float]:
        """
        Analyze pollution levels during peak traffic hours.
        
        Args:
            pollution_data: DataFrame with timestamp and pollution metrics
            peak_hours: List of peak traffic hours
            
        Returns:
            Dictionary with pollution analysis during peak hours
        """
        if 'timestamp' not in pollution_data.columns:
            raise ValueError("Pollution data must contain 'timestamp' column")
        
        pollution_data = pollution_data.copy()
        pollution_data['hour'] = pollution_data['timestamp'].dt.hour
        
        # Separate peak and non-peak data
        peak_data = pollution_data[pollution_data['hour'].isin(peak_hours)]
        non_peak_data = pollution_data[~pollution_data['hour'].isin(peak_hours)]
        
        analysis = {}
        
        # Analyze each pollution metric
        pollution_columns = ['aqi', 'pm25', 'pm10', 'no2', 'co']
        for col in pollution_columns:
            if col in pollution_data.columns:
                peak_mean = peak_data[col].mean() if not peak_data.empty else 0
                non_peak_mean = non_peak_data[col].mean() if not non_peak_data.empty else 0
                
                # Calculate percentage increase during peak hours
                if non_peak_mean > 0:
                    percent_increase = ((peak_mean - non_peak_mean) / non_peak_mean) * 100
                else:
                    percent_increase = 0
                
                analysis[f'{col}_peak_avg'] = peak_mean
                analysis[f'{col}_non_peak_avg'] = non_peak_mean
                analysis[f'{col}_percent_increase'] = percent_increase
        
        # Overall statistics
        analysis['peak_hours_count'] = len(peak_hours)
        analysis['peak_data_points'] = len(peak_data)
        analysis['non_peak_data_points'] = len(non_peak_data)
        
        logger.info(f"Peak hour analysis completed for {len(peak_hours)} peak hours")
        return analysis
    
    def calculate_rolling_correlation(self, traffic_data: pd.Series, pollution_data: pd.Series,
                                    window: int = 24) -> pd.Series:
        """
        Calculate rolling correlation over time windows.
        
        Args:
            traffic_data: Series of traffic data with datetime index
            pollution_data: Series of pollution data with datetime index
            window: Rolling window size in hours
            
        Returns:
            Series of rolling correlation values
        """
        # Align the series by index
        aligned_data = pd.DataFrame({
            'traffic': traffic_data,
            'pollution': pollution_data
        }).dropna()
        
        if len(aligned_data) < window:
            raise ValueError(f"Insufficient data for rolling correlation (need {window} points)")
        
        # Calculate rolling correlation
        rolling_corr = aligned_data['traffic'].rolling(window=window).corr(
            aligned_data['pollution']
        )
        
        return rolling_corr.dropna()
    
    def perform_comprehensive_analysis(self, aligned_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive correlation analysis on aligned traffic-pollution data.
        
        Args:
            aligned_data: DataFrame with traffic and pollution columns
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        required_columns = ['congestion_level', 'aqi', 'pm25', 'pm10']
        missing_columns = [col for col in required_columns if col not in aligned_data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(aligned_data),
            'correlations': {},
            'peak_analysis': {},
            'summary_statistics': {}
        }
        
        # Calculate correlations between traffic and various pollution metrics
        traffic_series = aligned_data['congestion_level']
        
        for pollution_metric in ['aqi', 'pm25', 'pm10', 'no2', 'co']:
            if pollution_metric in aligned_data.columns:
                try:
                    # Pearson correlation
                    pearson_result = self.calculate_pearson_correlation(
                        traffic_series, aligned_data[pollution_metric]
                    )
                    
                    # Spearman correlation
                    spearman_result = self.calculate_spearman_correlation(
                        traffic_series, aligned_data[pollution_metric]
                    )
                    
                    results['correlations'][pollution_metric] = {
                        'pearson': pearson_result.to_dict(),
                        'spearman': spearman_result.to_dict()
                    }
                    
                except Exception as e:
                    logger.warning(f"Could not calculate correlation for {pollution_metric}: {e}")
                    results['correlations'][pollution_metric] = None
        
        # Peak hour analysis
        try:
            peak_hours = self.identify_peak_hours(aligned_data)
            peak_analysis = self.analyze_pollution_during_peaks(aligned_data, peak_hours)
            results['peak_analysis'] = peak_analysis
        except Exception as e:
            logger.warning(f"Could not perform peak hour analysis: {e}")
            results['peak_analysis'] = {}
        
        # Summary statistics
        results['summary_statistics'] = {
            'traffic_mean': float(traffic_series.mean()),
            'traffic_std': float(traffic_series.std()),
            'aqi_mean': float(aligned_data['aqi'].mean()) if 'aqi' in aligned_data.columns else None,
            'aqi_std': float(aligned_data['aqi'].std()) if 'aqi' in aligned_data.columns else None,
            'pm25_mean': float(aligned_data['pm25'].mean()) if 'pm25' in aligned_data.columns else None,
            'pm25_std': float(aligned_data['pm25'].std()) if 'pm25' in aligned_data.columns else None
        }
        
        logger.info(f"Comprehensive analysis completed for {len(aligned_data)} data points")
        return results
    
    def _validate_series_data(self, series1: pd.Series, series2: pd.Series) -> None:
        """Validate input series for correlation analysis."""
        if not isinstance(series1, pd.Series) or not isinstance(series2, pd.Series):
            raise ValueError("Input data must be pandas Series")
        
        if len(series1) != len(series2):
            raise ValueError("Series must have the same length")
        
        if len(series1) == 0:
            raise ValueError("Series cannot be empty")
    
    def _clean_paired_data(self, series1: pd.Series, series2: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Remove NaN values from paired series data."""
        # Create DataFrame to handle paired NaN removal
        df = pd.DataFrame({'x': series1, 'y': series2})
        df_clean = df.dropna()
        
        return df_clean['x'].values, df_clean['y'].values
    
    def _calculate_confidence_interval(self, correlation: float, sample_size: int, 
                                     confidence_level: float = 0.95, 
                                     method: str = 'pearson') -> Tuple[float, float]:
        """
        Calculate confidence interval for correlation coefficient.
        
        Args:
            correlation: Correlation coefficient
            sample_size: Sample size
            confidence_level: Confidence level (default 0.95)
            method: Correlation method ('pearson' or 'spearman')
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if sample_size < 3:
            return (-1.0, 1.0)  # Wide interval for small samples
        
        # Fisher's z-transformation for Pearson correlation
        if method == 'pearson':
            # Avoid division by zero
            r_clipped = np.clip(correlation, -0.999, 0.999)
            z = 0.5 * np.log((1 + r_clipped) / (1 - r_clipped))
            
            # Standard error
            se = 1 / np.sqrt(sample_size - 3)
            
            # Critical value for confidence level
            alpha = 1 - confidence_level
            z_critical = stats.norm.ppf(1 - alpha/2)
            
            # Confidence interval in z-space
            z_lower = z - z_critical * se
            z_upper = z + z_critical * se
            
            # Transform back to correlation space
            r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
            
            return (float(r_lower), float(r_upper))
        
        else:  # Spearman - use approximation
            se = 1 / np.sqrt(sample_size - 3)
            z_critical = stats.norm.ppf(1 - (1 - confidence_level)/2)
            
            lower = correlation - z_critical * se
            upper = correlation + z_critical * se
            
            # Clip to valid correlation range
            return (max(-1.0, lower), min(1.0, upper))