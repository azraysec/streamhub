"""SQLAlchemy models for StreamHub content aggregation platform.

This package contains all database models and Pydantic schemas
for the StreamHub application.

Models:
    - Source: Content source configurations (Telegram, RSS, etc.)
    - Content: Unified content storage with vector embeddings
    - Category: Hierarchical content categories
    - Tag: User-defined content tags
    - User: Dashboard users with authentication
    - UserPreferences: User personalization settings

Association Tables:
    - ContentCategory: Links content to categories with confidence scores
    - ContentTag: Links content to tags

Usage:
    from app.models import Source, Content, Category, Tag, User
    from app.models import SourceType, ContentType, UserRole
    from app.models.schemas import SourceCreate, ContentResponse, etc.
"""

# Base and utilities
from .base import Base, TimestampMixin, metadata

# Enums
from .enums import ContentType, SourceType, UserRole

# Models
from .source import Source
from .content import Content
from .category import Category, ContentCategory
from .tag import Tag, ContentTag
from .user import User, UserPreferences

# Pydantic schemas
from .schemas import (
    # Source schemas
    SourceBase,
    SourceCreate,
    SourceUpdate,
    SourceResponse,
    SourceWithStats,
    # Content schemas
    ContentBase,
    ContentCreate,
    ContentUpdate,
    ContentResponse,
    ContentDetailResponse,
    ContentListResponse,
    SimilarContentRequest,
    SimilarContentResult,
    # Category schemas
    CategoryBase,
    CategoryCreate,
    CategoryUpdate,
    CategoryResponse,
    CategoryTreeResponse,
    ContentCategoryAssignment,
    # Tag schemas
    TagBase,
    TagCreate,
    TagUpdate,
    TagResponse,
    TagWithCount,
    # User schemas
    UserBase,
    UserCreate,
    UserUpdate,
    UserPasswordChange,
    UserResponse,
    UserDetailResponse,
    # User preferences schemas
    UserPreferencesBase,
    UserPreferencesCreate,
    UserPreferencesUpdate,
    UserPreferencesResponse,
    # Auth schemas
    LoginRequest,
    TokenResponse,
    TokenRefreshRequest,
    # Common schemas
    PaginationParams,
    FilterParams,
    SortParams,
    HealthResponse,
    ErrorResponse,
)

__all__ = [
    # Base
    "Base",
    "TimestampMixin",
    "metadata",
    # Enums
    "SourceType",
    "ContentType",
    "UserRole",
    # Models
    "Source",
    "Content",
    "Category",
    "ContentCategory",
    "Tag",
    "ContentTag",
    "User",
    "UserPreferences",
    # Source schemas
    "SourceBase",
    "SourceCreate",
    "SourceUpdate",
    "SourceResponse",
    "SourceWithStats",
    # Content schemas
    "ContentBase",
    "ContentCreate",
    "ContentUpdate",
    "ContentResponse",
    "ContentDetailResponse",
    "ContentListResponse",
    "SimilarContentRequest",
    "SimilarContentResult",
    # Category schemas
    "CategoryBase",
    "CategoryCreate",
    "CategoryUpdate",
    "CategoryResponse",
    "CategoryTreeResponse",
    "ContentCategoryAssignment",
    # Tag schemas
    "TagBase",
    "TagCreate",
    "TagUpdate",
    "TagResponse",
    "TagWithCount",
    # User schemas
    "UserBase",
    "UserCreate",
    "UserUpdate",
    "UserPasswordChange",
    "UserResponse",
    "UserDetailResponse",
    # User preferences schemas
    "UserPreferencesBase",
    "UserPreferencesCreate",
    "UserPreferencesUpdate",
    "UserPreferencesResponse",
    # Auth schemas
    "LoginRequest",
    "TokenResponse",
    "TokenRefreshRequest",
    # Common schemas
    "PaginationParams",
    "FilterParams",
    "SortParams",
    "HealthResponse",
    "ErrorResponse",
]
