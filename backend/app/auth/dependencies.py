"""Authentication Dependencies for FastAPI.

This module provides dependency injection functions for authentication
and authorization in FastAPI endpoints.
"""

from typing import Annotated, List, Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from app.auth.jwt import (
    verify_token,
    ACCESS_TOKEN_TYPE,
    TokenError,
    TokenExpiredError,
    TokenInvalidError,
)
from app.schemas.auth import TokenPayload, UserInDB


# OAuth2 scheme for token extraction from Authorization header
oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login",
    scheme_name="JWT",
    auto_error=True,
)

# Optional OAuth2 scheme that doesn't raise error if token is missing
oauth2_scheme_optional = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login",
    scheme_name="JWT",
    auto_error=False,
)


async def get_token_payload(
    token: Annotated[str, Depends(oauth2_scheme)],
) -> TokenPayload:
    """Extract and validate token payload from the Authorization header.
    
    Args:
        token: JWT token from Authorization header.
    
    Returns:
        Validated token payload.
    
    Raises:
        HTTPException: If token is invalid or expired.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = verify_token(token, token_type=ACCESS_TOKEN_TYPE)
        return TokenPayload(**payload)
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except TokenInvalidError as e:
        raise credentials_exception
    except Exception:
        raise credentials_exception


async def get_current_user(
    token_payload: Annotated[TokenPayload, Depends(get_token_payload)],
    # db: Annotated[AsyncSession, Depends(get_db)],  # Uncomment when DB is ready
) -> UserInDB:
    """Get the current authenticated user from the token.
    
    Args:
        token_payload: Validated token payload.
        db: Database session (inject when available).
    
    Returns:
        User object from database.
    
    Raises:
        HTTPException: If user not found or inactive.
    """
    user_id = token_payload.sub
    
    # TODO: Fetch user from database
    # user = await user_repository.get_by_id(db, user_id)
    # if not user:
    #     raise HTTPException(
    #         status_code=status.HTTP_404_NOT_FOUND,
    #         detail="User not found",
    #     )
    
    # Placeholder user for development
    user = UserInDB(
        id=user_id,
        email="user@example.com",
        username="testuser",
        hashed_password="placeholder",
        is_active=True,
        is_verified=True,
        role=token_payload.role or "user",
    )
    
    return user


async def get_current_active_user(
    current_user: Annotated[UserInDB, Depends(get_current_user)],
) -> UserInDB:
    """Get current user and verify they are active.
    
    Args:
        current_user: User from get_current_user dependency.
    
    Returns:
        Active user object.
    
    Raises:
        HTTPException: If user is inactive.
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is disabled",
        )
    return current_user


async def get_optional_user(
    token: Annotated[Optional[str], Depends(oauth2_scheme_optional)],
) -> Optional[UserInDB]:
    """Get user if authenticated, otherwise return None.
    
    Use this for endpoints that work for both authenticated and
    unauthenticated users but provide additional features for authenticated users.
    
    Args:
        token: Optional JWT token from Authorization header.
    
    Returns:
        User object if authenticated, None otherwise.
    """
    if not token:
        return None
    
    try:
        payload = verify_token(token, token_type=ACCESS_TOKEN_TYPE)
        token_payload = TokenPayload(**payload)
        
        # TODO: Fetch user from database
        user = UserInDB(
            id=token_payload.sub,
            email="user@example.com",
            username="testuser",
            hashed_password="placeholder",
            is_active=True,
            is_verified=True,
            role=token_payload.role or "user",
        )
        return user
    except TokenError:
        return None


def require_role(allowed_roles: List[str]):
    """Dependency factory for role-based access control.
    
    Creates a dependency that checks if the current user has one of the
    allowed roles.
    
    Args:
        allowed_roles: List of role names that are allowed access.
    
    Returns:
        Dependency function that validates user role.
    
    Example:
        @router.get("/admin", dependencies=[Depends(require_role(["admin"]))])
        async def admin_endpoint():
            return {"message": "Admin only"}
    """
    async def role_checker(
        current_user: Annotated[UserInDB, Depends(get_current_active_user)],
    ) -> UserInDB:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {allowed_roles}",
            )
        return current_user
    
    return role_checker


def require_verified_user():
    """Dependency that requires a verified user account.
    
    Returns:
        Dependency function that validates user verification status.
    """
    async def verified_checker(
        current_user: Annotated[UserInDB, Depends(get_current_active_user)],
    ) -> UserInDB:
        if not current_user.is_verified:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Email verification required",
            )
        return current_user
    
    return verified_checker


# Pre-configured role dependencies for convenience
RequireAdmin = Depends(require_role(["admin"]))
RequireModerator = Depends(require_role(["admin", "moderator"]))
RequireUser = Depends(require_role(["admin", "moderator", "user"]))
