"""
Exception classes for the data module.
"""


class DataError(Exception):
    """Base exception for all data-related errors."""

    pass


class ProviderError(DataError):
    """Exception raised when a data provider encounters an error."""

    pass


class ValidationError(DataError):
    """Exception raised when data validation fails."""

    pass


class NetworkError(DataError):
    """Exception raised when network operations fail."""

    pass


class RateLimitError(DataError):
    """Exception raised when API rate limits are exceeded."""

    pass


class AuthenticationError(DataError):
    """Exception raised when API authentication fails."""

    pass
