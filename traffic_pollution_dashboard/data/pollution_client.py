"""
Air pollution data API client for fetching AQI, PM2.5, and PM10 data.

This module provides a PollutionDataClient class that integrates with air quality APIs
to fetch real-time and historical pollution data for Indian cities.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
import pandas as pd
import numpy as np
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config.settings import settings
from .models import PollutionData, DateRange
from .exceptions import PollutionAPIError, RateLimitError, AuthenticationError, NetworkError, DataValidationError
from ..services.data_cache import cache_manager, cached_api_call

logger = logging.getLogger(__name__)


class PollutionDataClient:
    """
    Client for fetching air pollution data from external APIs.
    
    Supports multiple pollution data providers with automatic fallback,
    rate limiting, authentication, and comprehensive error handling.
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize the pollution data client.
        
        Args:
            api_key: API key for authentication (defaults to settings)
            base_url: Base URL for the pollution API (defaults to settings)
        """
        self.api_key = api_key or settings.POLLUTION_API_KEY
        self.base_url = base_url or settings.POLLUTION_API_BASE_URL
        self.session = self._create_session()
        self.rate_limit_calls = 0
        self.rate_limit_window_start = time.time()
        
        if not self.api_key:
            logger.warning("No pollution API key provided. Using mock data.")
    
    def _create_session(self) -> requests.Session:
        """Create a requests session with retry strategy and connection pooling."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=20)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Set default headers
        session.headers.update({
            'User-Agent': 'TrafficPollutionDashboard/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        return session
    
    @cached_api_call('pollution_data', ttl=900)
    def fetch_pollution_data(self, city: str, date_range: DateRange) -> pd.DataFrame:
        """
        Fetch air pollution data for a specific city and date range.
        
        Args:
            city: City name (Delhi, Bengaluru, Chennai)
            date_range: Date range for data retrieval
            
        Returns:
            DataFrame with columns: timestamp, city, aqi, pm25, pm10, no2, co, station_id
            
        Raises:
            PollutionAPIError: When API request fails
            DataValidationError: When response data is invalid
        """
        logger.info(f"Fetching pollution data for {city} from {date_range.start_date} to {date_range.end_date}")
        
        # Validate inputs
        self._validate_city(city)
        self._validate_date_range(date_range)
        
        try:
            # Check rate limiting
            self._check_rate_limit()
            
            # For now, use mock data since we don't have real API keys
            # In production, this would make actual API calls
            if not self.api_key or self.api_key == "your_pollution_api_key_here":
                logger.info("Using mock pollution data")
                return self._generate_mock_pollution_data(city, date_range)
            
            # Real API implementation would go here
            response_data = self._make_api_request(city, date_range)
            
            # Validate response
            if not self.validate_response(response_data):
                raise DataValidationError("Invalid response format from pollution API")
            
            # Convert to DataFrame
            df = self._convert_to_dataframe(response_data, city)
            
            logger.info(f"Successfully fetched {len(df)} pollution data points for {city}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching pollution data: {e}")
            raise NetworkError(f"Failed to fetch pollution data: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching pollution data: {e}")
            raise PollutionAPIError(f"Failed to fetch pollution data for {city}: {e}")
    
    def _generate_mock_pollution_data(self, city: str, date_range: DateRange) -> pd.DataFrame:
        """Generate realistic mock pollution data."""
        # Generate hourly timestamps
        current = date_range.start_date
        timestamps = []
        while current <= date_range.end_date:
            timestamps.append(current)
            current += timedelta(hours=1)
        
        data = []
        for timestamp in timestamps:
            hour = timestamp.hour
            is_weekend = timestamp.weekday() >= 5
            
            # Base pollution levels vary by city (Delhi > Bengaluru > Chennai)
            city_base_aqi = {
                'Delhi': 180,
                'Bengaluru': 120, 
                'Chennai': 90
            }.get(city, 120)
            
            # Peak pollution during traffic hours and winter months
            if 7 <= hour <= 10 or 17 <= hour <= 21:  # Traffic peak hours
                pollution_multiplier = 1.4 if not is_weekend else 1.1
            elif 22 <= hour <= 6:  # Night hours - lower pollution
                pollution_multiplier = 0.7
            else:  # Day hours
                pollution_multiplier = 1.0
            
            # Add seasonal and random variation
            seasonal_factor = 1.2 if timestamp.month in [11, 12, 1, 2] else 0.9  # Winter pollution
            noise = np.random.normal(0, 15)
            
            aqi = max(10, min(500, city_base_aqi * pollution_multiplier * seasonal_factor + noise))
            
            # Calculate PM2.5 and PM10 based on AQI
            pm25 = max(5, aqi * 0.4 + np.random.normal(0, 5))
            pm10 = max(10, pm25 * 1.8 + np.random.normal(0, 8))
            
            # Other pollutants
            no2 = max(5, aqi * 0.3 + np.random.normal(0, 3))
            co = max(0.1, aqi * 0.02 + np.random.normal(0, 0.5))
            
            data.append(PollutionData(
                timestamp=timestamp,
                city=city,
                aqi=int(round(aqi)),
                pm25=round(pm25, 1),
                pm10=round(pm10, 1),
                no2=round(no2, 1),
                co=round(co, 2),
                station_id=f"{city}_central_station"
            ))
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': d.timestamp,
                'city': d.city,
                'aqi': d.aqi,
                'pm25': d.pm25,
                'pm10': d.pm10,
                'no2': d.no2,
                'co': d.co,
                'station_id': d.station_id
            }
            for d in data
        ])
        
        return df
    
    def _make_api_request(self, city: str, date_range: DateRange) -> Dict:
        """Make the actual API request (placeholder for real implementation)."""
        # Get city coordinates
        coords = settings.CITY_COORDINATES.get(city)
        if not coords:
            raise PollutionAPIError(f"Coordinates not found for city: {city}")
        
        # Prepare request parameters
        params = {
            'lat': coords['lat'],
            'lon': coords['lon'],
            'start': int(date_range.start_date.timestamp()),
            'end': int(date_range.end_date.timestamp()),
            'appid': self.api_key
        }
        
        # Make API request
        response = self.session.get(
            f"{self.base_url}/air_pollution/history",
            params=params,
            timeout=settings.API_TIMEOUT_SECONDS
        )
        
        # Handle HTTP errors
        if response.status_code == 401:
            raise AuthenticationError("Invalid API key for pollution service")
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            raise RateLimitError("Pollution API rate limit exceeded", retry_after)
        elif response.status_code != 200:
            raise PollutionAPIError(f"Pollution API returned status {response.status_code}: {response.text}")
        
        return response.json()
    
    def _convert_to_dataframe(self, response_data: Dict, city: str) -> pd.DataFrame:
        """Convert API response to standardized DataFrame format."""
        # This would parse the actual API response format
        # For now, return empty DataFrame as placeholder
        return pd.DataFrame(columns=['timestamp', 'city', 'aqi', 'pm25', 'pm10', 'no2', 'co', 'station_id'])
    
    def get_aqi_metrics(self, city: str) -> Dict[str, float]:
        """Get current AQI metrics for a city."""
        try:
            # Use last 1 hour of data
            end_date = datetime.now()
            start_date = end_date - timedelta(hours=1)
            date_range = DateRange(start_date=start_date, end_date=end_date)
            
            df = self.fetch_pollution_data(city, date_range)
            if df.empty:
                return {}
            
            latest = df.iloc[-1]
            return {
                'aqi': float(latest['aqi']),
                'pm25': float(latest['pm25']),
                'pm10': float(latest['pm10']),
                'no2': float(latest['no2']),
                'co': float(latest['co'])
            }
        except Exception as e:
            logger.error(f"Error getting AQI metrics: {e}")
            return {}
    
    def validate_response(self, response: Dict) -> bool:
        """
        Validate API response format and content.
        
        Args:
            response: API response dictionary
            
        Returns:
            True if response is valid, False otherwise
        """
        if not isinstance(response, dict):
            return False
        
        # Check for required fields (this would be API-specific)
        required_fields = ['list']  # OpenWeatherMap format
        for field in required_fields:
            if field not in response:
                logger.warning(f"Missing required field in response: {field}")
                return False
        
        return True
    
    def _check_rate_limit(self) -> None:
        """Check and enforce API rate limiting."""
        current_time = time.time()
        
        # Reset counter if window has passed
        if current_time - self.rate_limit_window_start >= 60:
            self.rate_limit_calls = 0
            self.rate_limit_window_start = current_time
        
        # Check if we've exceeded the limit
        if self.rate_limit_calls >= settings.API_RATE_LIMIT_PER_MINUTE:
            wait_time = 60 - (current_time - self.rate_limit_window_start)
            raise RateLimitError(f"Rate limit exceeded. Wait {wait_time:.1f} seconds.", int(wait_time))
        
        self.rate_limit_calls += 1
    
    def _validate_city(self, city: str) -> None:
        """Validate that the city is supported."""
        if city not in settings.SUPPORTED_CITIES:
            raise PollutionAPIError(f"Unsupported city: {city}. Supported cities: {settings.SUPPORTED_CITIES}")
    
    def _validate_date_range(self, date_range: DateRange) -> None:
        """Validate date range parameters."""
        # Check if date range is too large
        if date_range.duration_hours > 24 * 30:  # 30 days max
            raise PollutionAPIError("Date range too large. Maximum 30 days allowed.")
        
        # Check if dates are in the future
        if date_range.start_date > datetime.now():
            raise PollutionAPIError("Start date cannot be in the future")
    
    def handle_api_errors(self, error: Exception) -> None:
        """
        Handle API errors with appropriate logging and recovery strategies.
        
        Args:
            error: The exception that occurred
        """
        if isinstance(error, AuthenticationError):
            logger.error("Authentication failed. Check API key.")
        elif isinstance(error, RateLimitError):
            logger.warning(f"Rate limit exceeded: {error}")
            if hasattr(error, 'retry_after') and error.retry_after:
                logger.info(f"Waiting {error.retry_after} seconds before retry...")
                time.sleep(error.retry_after)
        elif isinstance(error, NetworkError):
            logger.error(f"Network error: {error}")
        else:
            logger.error(f"Unexpected API error: {error}")
    
    def get_supported_cities(self) -> List[str]:
        """Get list of supported cities."""
        return settings.SUPPORTED_CITIES.copy()
    
    def health_check(self) -> bool:
        """Check if the pollution API is accessible."""
        try:
            # Simple health check - try to make a minimal request
            response = self.session.get(f"{self.base_url}/weather", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Pollution API health check failed: {e}")
            return False