"""Middleware module for StreamHub.

This module provides middleware components for:
- Error handling and standardized error responses
- Request logging and tracing
- Rate limiting
- Request validation
"""

from app.middleware.error_handler import (
    StreamHubException,
    NotFoundException,
    UnauthorizedException,
    ForbiddenException,
    BadRequestException,
    ConflictException,
    ValidationException,
    RateLimitExceededException,
    ServiceUnavailableException,
    http_exception_handler,
    validation_exception_handler,
    general_exception_handler,
)

__all__ = [
    "StreamHubException",
    "NotFoundException",
    "UnauthorizedException",
    "ForbiddenException",
    "BadRequestException",
    "ConflictException",
    "ValidationException",
    "RateLimitExceededException",
    "ServiceUnavailableException",
    "http_exception_handler",
    "validation_exception_handler",
    "general_exception_handler",
]
