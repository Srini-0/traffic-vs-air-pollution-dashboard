"""
Data processing service for cleaning and aligning traffic and pollution datasets.

This module provides the DataProcessor class that handles data cleaning,
normalization, timestamp alignment, and dataset merging operations.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union
import pytz

from ..data.models import DateRange

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Service for processing, cleaning, and aligning traffic and pollution datasets.
    
    Handles data cleaning, normalization, timestamp alignment, timezone conversion,
    and merging of datasets from different sources.
    """
    
    def __init__(self, default_timezone: str = 'Asia/Kolkata'):
        """
        Initialize the data processor.
        
        Args:
            default_timezone: Default timezone for Indian cities
        """
        self.default_timezone = pytz.timezone(default_timezone)
        self.logger = logging.getLogger(__name__)
    
    def clean_traffic_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize traffic data.
        
        Args:
            raw_data: Raw traffic DataFrame
            
        Returns:
            Cleaned DataFrame with standardized columns and data types
            
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        if raw_data.empty:
            logger.warning("Empty traffic data provided for cleaning")
            return self._create_empty_traffic_dataframe()
        
        logger.info(f"Cleaning traffic data with {len(raw_data)} records")
        
        # Create a copy to avoid modifying original data
        df = raw_data.copy()
        
        # Validate required columns
        required_columns = ['timestamp', 'city', 'congestion_level']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in traffic data: {missing_columns}")
        
        # Clean timestamp column
        df = self._clean_timestamps(df)
        
        # Clean numeric columns
        df = self._clean_traffic_numeric_columns(df)
        
        # Remove duplicates based on timestamp and city
        initial_count = len(df)
        df = df.drop_duplicates(subset=['timestamp', 'city'], keep='first')
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} duplicate records")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Validate data ranges
        df = self._validate_traffic_data_ranges(df)
        
        logger.info(f"Cleaned traffic data: {len(df)} records remaining")
        return df
    
    def clean_pollution_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize pollution data.
        
        Args:
            raw_data: Raw pollution DataFrame
            
        Returns:
            Cleaned DataFrame with standardized columns and data types
            
        Raises:
            ValueError: If required columns are missing or data is invalid
        """
        if raw_data.empty:
            logger.warning("Empty pollution data provided for cleaning")
            return self._create_empty_pollution_dataframe()
        
        logger.info(f"Cleaning pollution data with {len(raw_data)} records")
        
        # Create a copy to avoid modifying original data
        df = raw_data.copy()
        
        # Validate required columns
        required_columns = ['timestamp', 'city', 'aqi', 'pm25', 'pm10']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in pollution data: {missing_columns}")
        
        # Clean timestamp column
        df = self._clean_timestamps(df)
        
        # Clean numeric columns
        df = self._clean_pollution_numeric_columns(df)
        
        # Remove duplicates based on timestamp and city
        initial_count = len(df)
        df = df.drop_duplicates(subset=['timestamp', 'city'], keep='first')
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} duplicate records")
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Validate data ranges
        df = self._validate_pollution_data_ranges(df)
        
        logger.info(f"Cleaned pollution data: {len(df)} records remaining")
        return df
    
    def align_datasets(self, traffic_df: pd.DataFrame, pollution_df: pd.DataFrame) -> pd.DataFrame:
        """
        Align traffic and pollution datasets by timestamp and handle timezone differences.
        
        Args:
            traffic_df: Cleaned traffic DataFrame
            pollution_df: Cleaned pollution DataFrame
            
        Returns:
            Merged DataFrame with aligned timestamps
            
        Raises:
            ValueError: If datasets cannot be aligned
        """
        if traffic_df.empty or pollution_df.empty:
            logger.warning("One or both datasets are empty, returning empty aligned dataset")
            return self._create_empty_aligned_dataframe()
        
        logger.info(f"Aligning datasets: {len(traffic_df)} traffic records, {len(pollution_df)} pollution records")
        
        # Ensure both datasets have timezone-aware timestamps
        traffic_df = self._ensure_timezone_aware(traffic_df.copy())
        pollution_df = self._ensure_timezone_aware(pollution_df.copy())
        
        # Round timestamps to nearest hour for alignment
        traffic_df['timestamp_hour'] = traffic_df['timestamp'].dt.floor('h')
        pollution_df['timestamp_hour'] = pollution_df['timestamp'].dt.floor('h')
        
        # Merge on timestamp and city
        merged_df = pd.merge(
            traffic_df,
            pollution_df,
            on=['timestamp_hour', 'city'],
            how='inner',
            suffixes=('_traffic', '_pollution')
        )
        
        if merged_df.empty:
            logger.warning("No overlapping data found between traffic and pollution datasets")
            return self._create_empty_aligned_dataframe()
        
        # Use traffic timestamp as primary timestamp (more frequent updates)
        merged_df['timestamp'] = merged_df['timestamp_traffic']
        
        # Select and rename columns
        aligned_df = merged_df[[
            'timestamp', 'city',
            'congestion_level', 'average_speed', 'delay_index',
            'aqi', 'pm25', 'pm10', 'no2', 'co'
        ]].copy()
        
        # Sort by timestamp
        aligned_df = aligned_df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info(f"Successfully aligned datasets: {len(aligned_df)} records")
        return aligned_df
    
    def normalize_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize metrics to standard scales for comparison and analysis.
        
        Args:
            df: DataFrame with traffic and pollution metrics
            
        Returns:
            DataFrame with additional normalized columns
        """
        if df.empty:
            return df
        
        logger.info(f"Normalizing metrics for {len(df)} records")
        
        normalized_df = df.copy()
        
        # Normalize congestion level (0-100) to 0-1 scale
        if 'congestion_level' in df.columns:
            normalized_df['congestion_level_norm'] = df['congestion_level'] / 100.0
        
        # Normalize AQI (0-500) to 0-1 scale
        if 'aqi' in df.columns:
            normalized_df['aqi_norm'] = df['aqi'] / 500.0
        
        # Normalize PM2.5 (typical range 0-200) to 0-1 scale
        if 'pm25' in df.columns:
            normalized_df['pm25_norm'] = np.clip(df['pm25'] / 200.0, 0, 1)
        
        # Normalize PM10 (typical range 0-400) to 0-1 scale
        if 'pm10' in df.columns:
            normalized_df['pm10_norm'] = np.clip(df['pm10'] / 400.0, 0, 1)
        
        # Calculate composite pollution index (weighted average)
        if all(col in df.columns for col in ['aqi', 'pm25', 'pm10']):
            normalized_df['pollution_index'] = (
                0.5 * normalized_df['aqi_norm'] +
                0.3 * normalized_df['pm25_norm'] +
                0.2 * normalized_df['pm10_norm']
            )
        
        logger.info("Metrics normalization completed")
        return normalized_df
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'interpolate') -> pd.DataFrame:
        """
        Handle missing values in the dataset using specified method.
        
        Args:
            df: DataFrame with potential missing values
            method: Method to handle missing values ('interpolate', 'forward_fill', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        if df.empty:
            return df
        
        logger.info(f"Handling missing values using method: {method}")
        
        result_df = df.copy()
        numeric_columns = result_df.select_dtypes(include=[np.number]).columns
        
        if method == 'interpolate':
            # Use linear interpolation for numeric columns
            for col in numeric_columns:
                if result_df[col].isna().any():
                    result_df[col] = result_df[col].interpolate(method='linear')
        
        elif method == 'forward_fill':
            # Forward fill missing values
            result_df[numeric_columns] = result_df[numeric_columns].fillna(method='ffill')
        
        elif method == 'drop':
            # Drop rows with any missing values
            initial_count = len(result_df)
            result_df = result_df.dropna()
            logger.info(f"Dropped {initial_count - len(result_df)} rows with missing values")
        
        else:
            raise ValueError(f"Unknown missing value handling method: {method}")
        
        # Fill any remaining missing values with column medians
        for col in numeric_columns:
            if result_df[col].isna().any():
                median_value = result_df[col].median()
                result_df[col] = result_df[col].fillna(median_value)
                logger.info(f"Filled remaining missing values in {col} with median: {median_value}")
        
        return result_df
    
    def _clean_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize timestamp column."""
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        
        # Remove rows with invalid timestamps
        initial_count = len(df)
        df = df.dropna(subset=['timestamp'])
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} rows with invalid timestamps")
        
        return df
    
    def _clean_traffic_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric columns specific to traffic data."""
        # Clean congestion level
        if 'congestion_level' in df.columns:
            df['congestion_level'] = pd.to_numeric(df['congestion_level'], errors='coerce')
            df['congestion_level'] = df['congestion_level'].clip(0, 100)
        
        # Clean average speed
        if 'average_speed' in df.columns:
            df['average_speed'] = pd.to_numeric(df['average_speed'], errors='coerce')
            df['average_speed'] = df['average_speed'].clip(0, 200)  # Reasonable speed limit
        
        # Clean delay index
        if 'delay_index' in df.columns:
            df['delay_index'] = pd.to_numeric(df['delay_index'], errors='coerce')
            df['delay_index'] = df['delay_index'].clip(1.0, 10.0)  # Reasonable delay range
        
        return df
    
    def _clean_pollution_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean numeric columns specific to pollution data."""
        # Clean AQI
        if 'aqi' in df.columns:
            df['aqi'] = pd.to_numeric(df['aqi'], errors='coerce')
            df['aqi'] = df['aqi'].clip(0, 500)
        
        # Clean PM2.5
        if 'pm25' in df.columns:
            df['pm25'] = pd.to_numeric(df['pm25'], errors='coerce')
            df['pm25'] = df['pm25'].clip(0, 1000)  # Extreme upper limit
        
        # Clean PM10
        if 'pm10' in df.columns:
            df['pm10'] = pd.to_numeric(df['pm10'], errors='coerce')
            df['pm10'] = df['pm10'].clip(0, 2000)  # Extreme upper limit
        
        # Clean NO2
        if 'no2' in df.columns:
            df['no2'] = pd.to_numeric(df['no2'], errors='coerce')
            df['no2'] = df['no2'].clip(0, 500)
        
        # Clean CO
        if 'co' in df.columns:
            df['co'] = pd.to_numeric(df['co'], errors='coerce')
            df['co'] = df['co'].clip(0, 50)
        
        return df
    
    def _validate_traffic_data_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate traffic data is within reasonable ranges."""
        initial_count = len(df)
        
        # Remove rows with invalid congestion levels
        if 'congestion_level' in df.columns:
            df = df[df['congestion_level'].between(0, 100)]
        
        # Remove rows with invalid speeds
        if 'average_speed' in df.columns:
            df = df[df['average_speed'] > 0]
        
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} rows with invalid traffic data ranges")
        
        return df
    
    def _validate_pollution_data_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate pollution data is within reasonable ranges."""
        initial_count = len(df)
        
        # Remove rows with invalid AQI
        if 'aqi' in df.columns:
            df = df[df['aqi'].between(0, 500)]
        
        # Remove rows with invalid PM values
        if 'pm25' in df.columns:
            df = df[df['pm25'] > 0]
        
        if 'pm10' in df.columns:
            df = df[df['pm10'] > 0]
        
        if len(df) < initial_count:
            logger.info(f"Removed {initial_count - len(df)} rows with invalid pollution data ranges")
        
        return df
    
    def _ensure_timezone_aware(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure timestamps are timezone-aware."""
        if df['timestamp'].dt.tz is None:
            # Assume local timezone if not specified
            df['timestamp'] = df['timestamp'].dt.tz_localize(self.default_timezone)
        else:
            # Convert to default timezone
            df['timestamp'] = df['timestamp'].dt.tz_convert(self.default_timezone)
        
        return df
    
    def _create_empty_traffic_dataframe(self) -> pd.DataFrame:
        """Create empty DataFrame with traffic data structure."""
        return pd.DataFrame(columns=[
            'timestamp', 'city', 'congestion_level', 'average_speed', 'delay_index', 'road_segment'
        ])
    
    def _create_empty_pollution_dataframe(self) -> pd.DataFrame:
        """Create empty DataFrame with pollution data structure."""
        return pd.DataFrame(columns=[
            'timestamp', 'city', 'aqi', 'pm25', 'pm10', 'no2', 'co', 'station_id'
        ])
    
    def _create_empty_aligned_dataframe(self) -> pd.DataFrame:
        """Create empty DataFrame with aligned data structure."""
        return pd.DataFrame(columns=[
            'timestamp', 'city', 'congestion_level', 'average_speed', 'delay_index',
            'aqi', 'pm25', 'pm10', 'no2', 'co'
        ])
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict:
        """
        Generate a data quality report for the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data quality metrics
        """
        if df.empty:
            return {'status': 'empty', 'record_count': 0}
        
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        report = {
            'record_count': len(df),
            'date_range': {
                'start': df['timestamp'].min().isoformat() if 'timestamp' in df.columns else None,
                'end': df['timestamp'].max().isoformat() if 'timestamp' in df.columns else None
            },
            'missing_values': df.isnull().sum().to_dict(),
            'duplicate_count': df.duplicated().sum(),
            'numeric_stats': {}
        }
        
        # Add statistics for numeric columns
        for col in numeric_columns:
            report['numeric_stats'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max())
            }
        
        return report