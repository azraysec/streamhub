"""JWT Authentication Utilities for StreamHub.

This module provides JWT token creation, verification, and password
hashing functionality using industry-standard security practices.
"""

from datetime import datetime, timedelta, timezone
from typing import Any, Optional, Union

import bcrypt
from jose import JWTError, jwt
from pydantic import ValidationError

from app.config import settings


# Token types
ACCESS_TOKEN_TYPE = "access"
REFRESH_TOKEN_TYPE = "refresh"


class TokenError(Exception):
    """Base exception for token-related errors."""
    pass


class TokenExpiredError(TokenError):
    """Raised when a token has expired."""
    pass


class TokenInvalidError(TokenError):
    """Raised when a token is invalid."""
    pass


def create_access_token(
    subject: Union[str, int],
    additional_claims: Optional[dict[str, Any]] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a JWT access token.
    
    Args:
        subject: The subject of the token (typically user ID).
        additional_claims: Additional claims to include in the token.
        expires_delta: Custom expiration time. Defaults to settings value.
    
    Returns:
        Encoded JWT access token string.
    
    Example:
        >>> token = create_access_token(subject="user123", additional_claims={"role": "admin"})
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.jwt_access_token_expire_minutes)
    
    now = datetime.now(timezone.utc)
    expire = now + expires_delta
    
    to_encode = {
        "sub": str(subject),
        "type": ACCESS_TOKEN_TYPE,
        "iat": now,
        "exp": expire,
        "nbf": now,  # Not valid before
    }
    
    if additional_claims:
        to_encode.update(additional_claims)
    
    return jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def create_refresh_token(
    subject: Union[str, int],
    additional_claims: Optional[dict[str, Any]] = None,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a JWT refresh token.
    
    Refresh tokens have longer expiration and are used to obtain new access tokens.
    
    Args:
        subject: The subject of the token (typically user ID).
        additional_claims: Additional claims to include in the token.
        expires_delta: Custom expiration time. Defaults to settings value.
    
    Returns:
        Encoded JWT refresh token string.
    """
    if expires_delta is None:
        expires_delta = timedelta(days=settings.jwt_refresh_token_expire_days)
    
    now = datetime.now(timezone.utc)
    expire = now + expires_delta
    
    to_encode = {
        "sub": str(subject),
        "type": REFRESH_TOKEN_TYPE,
        "iat": now,
        "exp": expire,
        "nbf": now,
    }
    
    if additional_claims:
        to_encode.update(additional_claims)
    
    return jwt.encode(
        to_encode,
        settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
    )


def create_token_pair(
    subject: Union[str, int],
    additional_claims: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """Create both access and refresh tokens.
    
    Args:
        subject: The subject of the tokens (typically user ID).
        additional_claims: Additional claims to include in both tokens.
    
    Returns:
        Dictionary with 'access_token' and 'refresh_token' keys.
    """
    return {
        "access_token": create_access_token(subject, additional_claims),
        "refresh_token": create_refresh_token(subject, additional_claims),
    }


def decode_token(token: str, verify_exp: bool = True) -> dict[str, Any]:
    """Decode and validate a JWT token.
    
    Args:
        token: The JWT token string to decode.
        verify_exp: Whether to verify token expiration.
    
    Returns:
        Dictionary containing the token claims.
    
    Raises:
        TokenExpiredError: If the token has expired.
        TokenInvalidError: If the token is invalid.
    """
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            options={"verify_exp": verify_exp},
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise TokenExpiredError("Token has expired")
    except JWTError as e:
        raise TokenInvalidError(f"Invalid token: {str(e)}")


def verify_token(
    token: str,
    token_type: Optional[str] = None,
) -> dict[str, Any]:
    """Verify a JWT token and optionally check its type.
    
    Args:
        token: The JWT token string to verify.
        token_type: Expected token type ('access' or 'refresh').
    
    Returns:
        Dictionary containing the token claims.
    
    Raises:
        TokenExpiredError: If the token has expired.
        TokenInvalidError: If the token is invalid or wrong type.
    """
    payload = decode_token(token)
    
    if token_type and payload.get("type") != token_type:
        raise TokenInvalidError(
            f"Invalid token type. Expected {token_type}, got {payload.get('type')}"
        )
    
    return payload


def get_token_subject(token: str) -> str:
    """Extract the subject (user ID) from a token.
    
    Args:
        token: The JWT token string.
    
    Returns:
        The subject claim from the token.
    
    Raises:
        TokenInvalidError: If subject claim is missing.
    """
    payload = decode_token(token)
    subject = payload.get("sub")
    
    if not subject:
        raise TokenInvalidError("Token missing subject claim")
    
    return subject


def hash_password(password: str) -> str:
    """Hash a password using bcrypt.
    
    Uses bcrypt with automatic salt generation for secure password hashing.
    
    Args:
        password: Plain text password to hash.
    
    Returns:
        Hashed password string.
    
    Example:
        >>> hashed = hash_password("my_secure_password")
        >>> verify_password("my_secure_password", hashed)
        True
    """
    password_bytes = password.encode("utf-8")
    salt = bcrypt.gensalt(rounds=12)  # Cost factor of 12
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash.
    
    Args:
        plain_password: Plain text password to verify.
        hashed_password: Hashed password to compare against.
    
    Returns:
        True if password matches, False otherwise.
    """
    try:
        password_bytes = plain_password.encode("utf-8")
        hashed_bytes = hashed_password.encode("utf-8")
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception:
        return False


def is_token_expired(token: str) -> bool:
    """Check if a token has expired without raising an exception.
    
    Args:
        token: The JWT token string to check.
    
    Returns:
        True if token is expired, False otherwise.
    """
    try:
        decode_token(token, verify_exp=True)
        return False
    except TokenExpiredError:
        return True
    except TokenInvalidError:
        return True
