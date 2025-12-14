"""
Data models for traffic and pollution data.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Tuple


@dataclass
class TrafficData:
    """Traffic congestion data model."""
    timestamp: datetime
    city: str
    congestion_level: float  # 0-100 scale
    average_speed: float     # km/h
    delay_index: float       # relative to free-flow
    road_segment: Optional[str] = None  # optional


@dataclass
class PollutionData:
    """Air pollution data model."""
    timestamp: datetime
    city: str
    aqi: int                 # Air Quality Index
    pm25: float             # PM2.5 μg/m³
    pm10: float             # PM10 μg/m³
    no2: float              # Nitrogen Dioxide
    co: float               # Carbon Monoxide
    station_id: Optional[str] = None  # monitoring station


@dataclass
class CorrelationResult:
    """Correlation analysis result model."""
    traffic_pollution_correlation: float
    statistical_significance: float
    sample_size: int
    confidence_interval: Tuple[float, float]
    peak_hour_analysis: Dict[str, float]


@dataclass
class DateRange:
    """Date range for data queries."""
    start_date: datetime
    end_date: datetime
    
    def __post_init__(self):
        """Validate date range."""
        if self.start_date >= self.end_date:
            raise ValueError("Start date must be before end date")
    
    @property
    def duration_hours(self) -> float:
        """Get duration in hours."""
        return (self.end_date - self.start_date).total_seconds() / 3600