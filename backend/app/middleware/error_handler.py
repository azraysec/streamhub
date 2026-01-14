"""Error Handling Middleware for StreamHub.

This module provides custom exception classes and exception handlers
for standardized error responses across the API.
"""

import logging
import traceback
from datetime import datetime, timezone
from typing import Any, Optional, Dict, List

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class ErrorResponse(BaseModel):
    """Standardized error response schema.
    
    Attributes:
        error: Error type identifier
        message: Human-readable error message
        details: Additional error details (optional)
        path: Request path that caused the error
        timestamp: ISO 8601 timestamp of the error
        request_id: Unique request identifier (optional)
    """
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    path: Optional[str] = None
    timestamp: str = None
    request_id: Optional[str] = None
    
    def __init__(self, **kwargs):
        if "timestamp" not in kwargs or kwargs["timestamp"] is None:
            kwargs["timestamp"] = datetime.now(timezone.utc).isoformat()
        super().__init__(**kwargs)


class StreamHubException(Exception):
    """Base exception for all StreamHub API errors.
    
    Attributes:
        status_code: HTTP status code
        error: Error type identifier
        message: Human-readable error message
        details: Additional error details
        headers: Optional response headers
    """
    
    def __init__(
        self,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error: str = "internal_error",
        message: str = "An internal error occurred",
        details: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        self.status_code = status_code
        self.error = error
        self.message = message
        self.details = details
        self.headers = headers
        super().__init__(message)


class NotFoundException(StreamHubException):
    """Exception for resource not found errors (404)."""
    
    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
    ):
        details = {}
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id
        
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            error="not_found",
            message=message,
            details=details if details else None,
        )


class UnauthorizedException(StreamHubException):
    """Exception for authentication errors (401)."""
    
    def __init__(
        self,
        message: str = "Authentication required",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            error="unauthorized",
            message=message,
            details=details,
            headers={"WWW-Authenticate": "Bearer"},
        )


class ForbiddenException(StreamHubException):
    """Exception for authorization errors (403)."""
    
    def __init__(
        self,
        message: str = "Access denied",
        required_role: Optional[str] = None,
    ):
        details = {"required_role": required_role} if required_role else None
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            error="forbidden",
            message=message,
            details=details,
        )


class BadRequestException(StreamHubException):
    """Exception for invalid request errors (400)."""
    
    def __init__(
        self,
        message: str = "Bad request",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            error="bad_request",
            message=message,
            details=details,
        )


class ConflictException(StreamHubException):
    """Exception for resource conflict errors (409)."""
    
    def __init__(
        self,
        message: str = "Resource already exists",
        conflicting_field: Optional[str] = None,
    ):
        details = {"conflicting_field": conflicting_field} if conflicting_field else None
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            error="conflict",
            message=message,
            details=details,
        )


class ValidationException(StreamHubException):
    """Exception for validation errors (422)."""
    
    def __init__(
        self,
        message: str = "Validation error",
        errors: Optional[List[Dict[str, Any]]] = None,
    ):
        details = {"validation_errors": errors} if errors else None
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error="validation_error",
            message=message,
            details=details,
        )


class RateLimitExceededException(StreamHubException):
    """Exception for rate limit exceeded errors (429)."""
    
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: Optional[int] = None,
    ):
        details = {"retry_after_seconds": retry_after} if retry_after else None
        headers = {"Retry-After": str(retry_after)} if retry_after else None
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error="rate_limit_exceeded",
            message=message,
            details=details,
            headers=headers,
        )


class ServiceUnavailableException(StreamHubException):
    """Exception for service unavailable errors (503)."""
    
    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        service: Optional[str] = None,
    ):
        details = {"service": service} if service else None
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error="service_unavailable",
            message=message,
            details=details,
        )


async def http_exception_handler(
    request: Request,
    exc: StreamHubException,
) -> JSONResponse:
    """Handle StreamHub custom exceptions.
    
    Args:
        request: The incoming request.
        exc: The StreamHubException that was raised.
    
    Returns:
        Standardized JSON error response.
    """
    request_id = request.headers.get("X-Request-ID", None)
    
    error_response = ErrorResponse(
        error=exc.error,
        message=exc.message,
        details=exc.details,
        path=str(request.url.path),
        request_id=request_id,
    )
    
    logger.warning(
        f"HTTP {exc.status_code}: {exc.error} - {exc.message}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "request_id": request_id,
        },
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.model_dump(exclude_none=True),
        headers=exc.headers,
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handle Pydantic validation errors.
    
    Args:
        request: The incoming request.
        exc: The validation error.
    
    Returns:
        Standardized JSON error response with validation details.
    """
    request_id = request.headers.get("X-Request-ID", None)
    
    # Format validation errors for clearer output
    formatted_errors = []
    for error in exc.errors():
        formatted_errors.append({
            "field": ".".join(str(loc) for loc in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        })
    
    error_response = ErrorResponse(
        error="validation_error",
        message="Request validation failed",
        details={"validation_errors": formatted_errors},
        path=str(request.url.path),
        request_id=request_id,
    )
    
    logger.warning(
        f"Validation error on {request.url.path}: {len(formatted_errors)} errors",
        extra={"errors": formatted_errors, "request_id": request_id},
    )
    
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=error_response.model_dump(exclude_none=True),
    )


async def general_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handle unexpected exceptions.
    
    This is a catch-all handler for any exceptions not handled by
    other exception handlers. It logs the full stack trace and
    returns a generic error message.
    
    Args:
        request: The incoming request.
        exc: The unexpected exception.
    
    Returns:
        Generic 500 error response.
    """
    request_id = request.headers.get("X-RK]UYest-ID", None)
    
    # Log the full exception with stack trace
    logger.error(
        f"Unhandled exception on {request.url.path}: {str(exc)}",
        exc_info=True,
        extra={
            "path": request.url.path,
            "method": request.method,
            "request_id": request_id,
            "exception_type": type(exc).__name__,
        },
    )
    
    # Don't expose internal error details in production
    error_response = ErrorResponse(
        error="internal_error",
        message="An internal server error occurred",
        path=str(request.url.path),
        request_id=request_id,
    )
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.model_dump(exclude_none=True),
    )
