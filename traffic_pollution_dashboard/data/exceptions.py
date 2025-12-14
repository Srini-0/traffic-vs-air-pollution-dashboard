"""
Custom exceptions for data layer operations.
"""


class APIError(Exception):
    """Base exception for API-related errors."""
    pass


class TrafficAPIError(APIError):
    """Exception for traffic API specific errors."""
    pass


class PollutionAPIError(APIError):
    """Exception for pollution API specific errors."""
    pass


class RateLimitError(APIError):
    """Exception for API rate limiting errors."""
    def __init__(self, message: str, retry_after: int = None):
        super().__init__(message)
        self.retry_after = retry_after


class AuthenticationError(APIError):
    """Exception for API authentication errors."""
    pass


class DataValidationError(Exception):
    """Exception for data validation errors."""
    pass


class NetworkError(APIError):
    """Exception for network-related errors."""
    pass