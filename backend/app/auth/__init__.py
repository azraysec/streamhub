"""Authentication module for StreamHub.

This module provides JWT-based authentication, password hashing,
and user authorization utilities.
"""

from app.auth.jwt import (
    create_access_token,
    create_refresh_token,
    verify_token,
    decode_token,
    hash_password,
    verify_password,
)
from app.auth.dependencies import (
    get_current_user,
    get_current_active_user,
    get_optional_user,
    require_role,
)

__all__ = [
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "decode_token",
    "hash_password",
    "verify_password",
    "get_current_user",
    "get_current_active_user",
    "get_optional_user",
    "require_role",
]
