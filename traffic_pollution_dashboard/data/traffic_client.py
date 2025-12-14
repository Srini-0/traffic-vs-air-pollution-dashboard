"""
Traffic data API client for fetching congestion data.

This module provides a TrafficDataClient class that integrates with traffic APIs
to fetch real-time and historical traffic congestion data for Indian cities.
"""

import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import requests
import pandas as pd
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config.settings import settings
from .models import TrafficData, DateRange
from .exceptions import TrafficAPIError, RateLimitError, AuthenticationError, NetworkError, DataValidationError
from ..services.data_cache import cache_manager, cached_api_call

logger = logging.getLogger(__name__)


class TrafficDataClient:
    """
    Client for fetching traffic congestion data from external APIs.
    
    Supports multiple traffic data providers with automatic fallback,
    rate limiting, authentication, and comprehensive error handling.
    """
    
    def __init__(self, api_key: str = None, base_url: str = None):
        """
        Initialize the traffic data client.
        
        Args:
            api_key: API key for authentication (defaults to settings)
            base_url: Base URL for the traffic API (defaults to settings)
        """
        self.api_key = api_key or settings.TRAFFIC_API_KEY
        self.base_url = base_url or settings.TRAFFIC_API_BASE_URL
        self.session = self._create_session()
        self.rate_limit_calls = 0
        self.rate_limit_window_start = time.time()
        
        if not self.api_key:
            logger.warning("No traffic API key provided. Using mock data.")
    
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
    
    @cached_api_call('traffic_data', ttl=900)
    def fetch_traffic_data(self, city: str, date_range: DateRange) -> pd.DataFrame:
        """
        Fetch traffic congestion data for a specific city and date range.
        
        Args:
            city: City name (Delhi, Bengaluru, Chennai)
            date_range: Date range for data retrieval
            
        Returns:
            DataFrame with columns: timestamp, city, congestion_level, average_speed, delay_index
            
        Raises:
            TrafficAPIError: When API request fails
            DataValidationError: When response data is invalid
        """
        logger.info(f"Fetching traffic data for {city} from {date_range.start_date} to {date_range.end_date}")
        
        # Validate inputs
        self._validate_city(city)
        self._validate_date_range(date_range)
        
        try:
            # Check rate limiting
            self._check_rate_limit()
            
            # For now, use mock data since we don't have real API keys
            # In production, this would make actual API calls
            if not self.api_key or self.api_key == "your_traffic_api_key_here":
                logger.info("Using mock traffic data")
                return self._generate_mock_traffic_data(city, date_range)
            
            # Real API implementation would go here
            response_data = self._make_api_request(city, date_range)
            
            # Validate response
            if not self.validate_response(response_data):
                raise DataValidationError("Invalid response format from traffic API")
            
            # Convert to DataFrame
            df = self._convert_to_dataframe(response_data, city)
            
            logger.info(f"Successfully fetched {len(df)} traffic data points for {city}")
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching traffic data: {e}")
            raise NetworkError(f"Failed to fetch traffic data: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching traffic data: {e}")
            raise TrafficAPIError(f"Failed to fetch traffic data for {city}: {e}")
    
    def _make_api_request(self, city: str, date_range: DateRange) -> Dict:
        """Make the actual API request (placeholder for real implementation)."""
        # Get city coordinates
        coords = settings.CITY_COORDINATES.get(city)
        if not coords:
            raise TrafficAPIError(f"Coordinates not found for city: {city}")
        
        # Prepare request parameters
        params = {
            'lat': coords['lat'],
            'lon': coords['lon'],
            'start_time': int(date_range.start_date.timestamp()),
            'end_time': int(date_range.end_date.timestamp()),
            'api_key': self.api_key
        }
        
        # Make API request
        response = self.session.get(
            f"{self.base_url}/traffic/historical",
            params=params,
            timeout=settings.API_TIMEOUT_SECONDS
        )
        
        # Handle HTTP errors
        if response.status_code == 401:
            raise AuthenticationError("Invalid API key for traffic service")
        elif response.status_code == 429:
            retry_after = int(response.headers.get('Retry-After', 60))
            raise RateLimitError("Traffic API rate limit exceeded", retry_after)
        elif response.status_code != 200:
            raise TrafficAPIError(f"Traffic API returned status {response.status_code}: {response.text}")
        
        return response.json()
    
    def _generate_mock_traffic_data(self, city: str, date_range: DateRange) -> pd.DataFrame:
        """
        Generate realistic mock traffic data for testing and development.
        
        This creates synthetic but realistic traffic patterns with:
        - Peak hours (8-10 AM, 6-8 PM) with higher congestion
        - Weekend vs weekday patterns
        - City-specific variations
        """
        import numpy as np
        
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
            
            # Base congestion level varies by city
            city_base_congestion = {
                'Delhi': 45,
                'Bengaluru': 40, 
                'Chennai': 35
            }.get(city, 40)
            
            # Peak hour multipliers
            if 8 <= hour <= 10 or 18 <= hour <= 20:  # Peak hours
                congestion_multiplier = 1.8 if not is_weekend else 1.3
            elif 6 <= hour <= 22:  # Day hours
                congestion_multiplier = 1.2 if not is_weekend else 0.8
            else:  # Night hours
                congestion_multiplier = 0.3
            
            # Add some randomness
            noise = np.random.normal(0, 5)
            congestion_level = max(0, min(100, city_base_congestion * congestion_multiplier + noise))
            
            # Calculate derived metrics
            # Higher congestion = lower average speed
            base_speed = 45  # km/h free flow speed
            average_speed = max(5, base_speed * (1 - congestion_level / 150))
            
            # Delay index relative to free flow
            delay_index = max(1.0, congestion_level / 30)
            
            data.append(TrafficData(
                timestamp=timestamp,
                city=city,
                congestion_level=round(congestion_level, 1),
                average_speed=round(average_speed, 1),
                delay_index=round(delay_index, 2),
                road_segment=f"{city}_main_roads"
            ))
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                'timestamp': d.timestamp,
                'city': d.city,
                'congestion_level': d.congestion_level,
                'average_speed': d.average_speed,
                'delay_index': d.delay_index,
                'road_segment': d.road_segment
            }
            for d in data
        ])
        
        return df
    
    def _convert_to_dataframe(self, response_data: Dict, city: str) -> pd.DataFrame:
        """Convert API response to standardized DataFrame format."""
        # This would parse the actual API response format
        # For now, return empty DataFrame as placeholder
        return pd.DataFrame(columns=['timestamp', 'city', 'congestion_level', 'average_speed', 'delay_index'])
    
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
        required_fields = ['data', 'status']
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
            raise TrafficAPIError(f"Unsupported city: {city}. Supported cities: {settings.SUPPORTED_CITIES}")
    
    def _validate_date_range(self, date_range: DateRange) -> None:
        """Validate date range parameters."""
        # Check if date range is too large
        if date_range.duration_hours > 24 * 30:  # 30 days max
            raise TrafficAPIError("Date range too large. Maximum 30 days allowed.")
        
        # Check if dates are in the future
        if date_range.start_date > datetime.now():
            raise TrafficAPIError("Start date cannot be in the future")
    
    def handle_rate_limiting(self) -> None:
        """Handle rate limiting by implementing exponential backoff."""
        try:
            self._check_rate_limit()
        except RateLimitError as e:
            if e.retry_after:
                logger.info(f"Rate limited. Waiting {e.retry_after} seconds...")
                time.sleep(e.retry_after)
            else:
                # Default backoff
                time.sleep(60)
    
    def get_supported_cities(self) -> List[str]:
        """Get list of supported cities."""
        return settings.SUPPORTED_CITIES.copy()
    
    def health_check(self) -> bool:
        """Check if the traffic API is accessible."""
        try:
            # Simple health check - try to make a minimal request
            response = self.session.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Traffic API health check failed: {e}")
            return False