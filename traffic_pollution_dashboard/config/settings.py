"""
Configuration settings for the traffic pollution dashboard.
"""

import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application configuration settings."""
    
    # API Configuration
    TRAFFIC_API_KEY: str = os.getenv("TRAFFIC_API_KEY", "")
    TRAFFIC_API_BASE_URL: str = os.getenv(
        "TRAFFIC_API_BASE_URL", 
        "https://api.example.com/traffic"
    )
    
    POLLUTION_API_KEY: str = os.getenv("POLLUTION_API_KEY", "")
    POLLUTION_API_BASE_URL: str = os.getenv(
        "POLLUTION_API_BASE_URL",
        "https://api.openweathermap.org/data/2.5/air_pollution"
    )
    
    # Cache Configuration
    REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    CACHE_TTL_SECONDS: int = int(os.getenv("CACHE_TTL_SECONDS", "900"))  # 15 minutes
    
    # Application Settings
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Supported Cities
    SUPPORTED_CITIES: List[str] = os.getenv(
        "SUPPORTED_CITIES", 
        "Delhi,Bengaluru,Chennai"
    ).split(",")
    
    # City Coordinates (for API calls)
    CITY_COORDINATES = {
        "Delhi": {"lat": 28.6139, "lon": 77.2090},
        "Bengaluru": {"lat": 12.9716, "lon": 77.5946},
        "Chennai": {"lat": 13.0827, "lon": 80.2707}
    }
    
    # API Rate Limiting
    API_RATE_LIMIT_PER_MINUTE: int = 60
    API_TIMEOUT_SECONDS: int = 30
    
    # Data Processing
    MAX_DATA_POINTS: int = 1000
    CORRELATION_MIN_SAMPLES: int = 10
    
    @classmethod
    def validate_config(cls) -> bool:
        """Validate that required configuration is present."""
        required_keys = [
            cls.TRAFFIC_API_KEY,
            cls.POLLUTION_API_KEY
        ]
        
        missing_keys = [key for key in required_keys if not key]
        
        if missing_keys:
            print(f"Warning: Missing required configuration: {missing_keys}")
            return False
        
        return True


# Global settings instance
settings = Settings()