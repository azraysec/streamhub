"""Pydantic schemas for API request/response validation."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

from .enums import ContentType, SourceType, UserRole


# =============================================================================
# BASE SCHEMAS
# =============================================================================

class BaseSchema(BaseModel):
    """Base schema with common configuration."""
    
    model_config = ConfigDict(
        from_attributes=True,
        str_strip_whitespace=True,
    )


class TimestampSchema(BaseSchema):
    """Schema mixin for timestamp fields."""
    
    created_at: datetime
    updated_at: datetime


# =============================================================================
# SOURCE SCHEMAS
# =============================================================================

class SourceBase(BaseSchema):
    """Base schema for source data."""
    
    name: str = Field(..., min_length=1, max_length=255, description="Human-readable name")
    source_type: SourceType = Field(..., description="Type of source platform")
    config: dict[str, Any] = Field(default_factory=dict, description="Source-specific configuration")
    enabled: bool = Field(default=True, description="Whether source is actively synced")


class SourceCreate(SourceBase):
    """Schema for creating a new source."""
    pass


class SourceUpdate(BaseSchema):
    """Schema for updating an existing source."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    config: Optional[dict[str, Any]] = None
    enabled: Optional[bool] = None


class SourceResponse(SourceBase, TimestampSchema):
    """Schema for source response."""
    
    id: int
    last_sync: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class SourceWithStats(SourceResponse):
    """Source response with content statistics."""
    
    content_count: int = 0
    latest_content_at: Optional[datetime] = None


# =============================================================================
# CONTENT SCHEMAS
# =============================================================================

class ContentBase(BaseSchema):
    """Base schema for content data."""
    
    content_type: ContentType = Field(..., description="Type of content")
    title: Optional[str] = Field(None, description="Title or headline")
    body: Optional[str] = Field(None, description="Main content text")
    author: Optional[str] = Field(None, max_length=255, description="Author name")
    url: Optional[str] = Field(None, description="Original URL")
    media_urls: list[str] = Field(default_factory=list, description="Media URLs")
    importance_score: int = Field(default=50, ge=0, le=100, description="Importance score (0-100)")
    published_at: Optional[datetime] = Field(None, description="Original publish date")


class ContentCreate(ContentBase):
    """Schema for creating new content."""
    
    source_id: int = Field(..., description="Source ID")
    external_id: str = Field(..., max_length=512, description="External platform ID")
    raw_data: dict[str, Any] = Field(default_factory=dict, description="Raw data from source")
    category_ids: list[int] = Field(default_factory=list, description="Category IDs to assign")
    tag_ids: list[int] = Field(default_factory=list, description="Tag IDs to assign")


class ContentUpdate(BaseSchema):
    """Schema for updating existing content."""
    
    title: Optional[str] = None
    body: Optional[str] = None
    author: Optional[str] = Field(None, max_length=255)
    url: Optional[str] = None
    media_urls: Optional[list[str]] = None
    importance_score: Optional[int] = Field(None, ge=0, le=100)
    is_duplicate: Optional[bool] = None
    category_ids: Optional[list[int]] = None
    tag_ids: Optional[list[int]] = None


class ContentResponse(ContentBase, TimestampSchema):
    """Schema for content response."""
    
    id: uuid.UUID
    source_id: int
    external_id: str
    is_duplicate: bool = False
    raw_data: dict[str, Any] = Field(default_factory=dict)
    
    model_config = ConfigDict(from_attributes=True)


class ContentDetailResponse(ContentResponse):
    """Detailed content response with relationships."""
    
    source: SourceResponse
    categories: list["CategoryResponse"] = []
    tags: list["TagResponse"] = []


class ContentListResponse(BaseSchema):
    """Paginated list of content."""
    
    items: list[ContentResponse]
    total: int
    page: int
    page_size: int
    pages: int


class SimilarContentRequest(BaseSchema):
    """Request for finding similar content."""
    
    text: str = Field(..., min_length=1, description="Text to find similar content for")
    threshold: float = Field(default=0.85, ge=0, le=1, description="Similarity threshold")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")


class SimilarContentResult(BaseSchema):
    """Similar content search result."""
    
    content: ContentResponse
    similarity: float = Field(..., ge=0, le=1)


# =============================================================================
# CATEGORY SCHEMAS
# =============================================================================

class CategoryBase(BaseSchema):
    """Base schema for category data."""
    
    name: str = Field(..., min_length=1, max_length=255, description="Category name")
    slug: str = Field(..., min_length=1, max_length=255, pattern=r"^[a-z0-9-]+$", description="URL-friendly slug")
    description: Optional[str] = Field(None, description="Category description")
    sort_order: int = Field(default=0, description="Display order")


class CategoryCreate(CategoryBase):
    """Schema for creating a new category."""
    
    parent_id: Optional[int] = Field(None, description="Parent category ID")


class CategoryUpdate(BaseSchema):
    """Schema for updating an existing category."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    slug: Optional[str] = Field(None, min_length=1, max_length=255, pattern=r"^[a-z0-9-]+$")
    description: Optional[str] = None
    parent_id: Optional[int] = None
    sort_order: Optional[int] = None


class CategoryResponse(CategoryBase, TimestampSchema):
    """Schema for category response."""
    
    id: int
    parent_id: Optional[int] = None
    
    model_config = ConfigDict(from_attributes=True)


class CategoryTreeResponse(CategoryResponse):
    """Category response with children for tree view."""
    
    children: list["CategoryTreeResponse"] = []
    content_count: int = 0


class ContentCategoryAssignment(BaseSchema):
    """Schema for assigning content to a category."""
    
    category_id: int
    confidence: Decimal = Field(default=Decimal("1.0"), ge=0, le=1)


# =============================================================================
# TAG SCHEMAS
# =============================================================================

class TagBase(BaseSchema):
    """Base schema for tag data."""
    
    name: str = Field(..., min_length=1, max_length=100, description="Tag name")
    slug: str = Field(..., min_length=1, max_length=100, pattern=r"^[a-z0-9-]+$", description="URL-friendly slug")
    color: str = Field(default="#6B7280", pattern=r"^#[0-9A-Fa-f]{6}$", description="Hex color code")


class TagCreate(TagBase):
    """Schema for creating a new tag."""
    pass


class TagUpdate(BaseSchema):
    """Schema for updating an existing tag."""
    
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    slug: Optional[str] = Field(None, min_length=1, max_length=100, pattern=r"^[a-z0-9-]+$")
    color: Optional[str] = Field(None, pattern=r"^#[0-9A-Fa-f]{6}$")


class TagResponse(TagBase):
    """Schema for tag response."""
    
    id: int
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class TagWithCount(TagResponse):
    """Tag response with content count."""
    
    content_count: int = 0


# =============================================================================
# USER SCHEMAS
# =============================================================================

class UserBase(BaseSchema):
    """Base schema for user data."""
    
    email: EmailStr = Field(..., description="User email address")
    full_name: Optional[str] = Field(None, max_length=255, description="Display name")
    role: UserRole = Field(default=UserRole.VIEWER, description="User role")


class UserCreate(UserBase):
    """Schema for creating a new user."""
    
    password: str = Field(..., min_length=8, max_length=128, description="Password (min 8 chars)")
    
    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserUpdate(BaseSchema):
    """Schema for updating an existing user."""
    
    email: Optional[EmailStr] = None
    full_name: Optional[str] = Field(None, max_length=255)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None


class UserPasswordChange(BaseSchema):
    """Schema for changing user password."""
    
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")


class UserResponse(UserBase, TimestampSchema):
    """Schema for user response (excludes sensitive data)."""
    
    id: int
    is_active: bool = True
    last_login: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class UserDetailResponse(UserResponse):
    """Detailed user response with preferences."""
    
    preferences: Optional["UserPreferencesResponse"] = None


# =============================================================================
# USER PREFERENCES SCHEMAS
# =============================================================================

class UserPreferencesBase(BaseSchema):
    """Base schema for user preferences."""
    
    preferences: dict[str, Any] = Field(default_factory=dict, description="User preference settings")


class UserPreferencesCreate(UserPreferencesBase):
    """Schema for creating user preferences."""
    pass


class UserPreferencesUpdate(BaseSchema):
    """Schema for updating user preferences."""
    
    preferences: dict[str, Any] = Field(..., description="Updated preference settings")


class UserPreferencesResponse(UserPreferencesBase, TimestampSchema):
    """Schema for user preferences response."""
    
    id: int
    user_id: int
    
    model_config = ConfigDict(from_attributes=True)


# =============================================================================
# AUTH SCHEMAS
# =============================================================================

class LoginRequest(BaseSchema):
    """Schema for login request."""
    
    email: EmailStr
    password: str


class TokenResponse(BaseSchema):
    """Schema for authentication token response."""
    
    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Token expiration in seconds")
    user: UserResponse


class TokenRefreshRequest(BaseSchema):
    """Schema for token refresh request."""
    
    refresh_token: str


# =============================================================================
# COMMON SCHEMAS
# =============================================================================

class PaginationParams(BaseSchema):
    """Common pagination parameters."""
    
    page: int = Field(default=1, ge=1, description="Page number")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")


class FilterParams(BaseSchema):
    """Common filter parameters for content listing."""
    
    source_ids: Optional[list[int]] = Field(None, description="Filter by source IDs")
    content_types: Optional[list[ContentType]] = Field(None, description="Filter by content types")
    category_ids: Optional[list[int]] = Field(None, description="Filter by category IDs")
    tag_ids: Optional[list[int]] = Field(None, description="Filter by tag IDs")
    author: Optional[str] = Field(None, description="Filter by author")
    is_duplicate: Optional[bool] = Field(None, description="Filter duplicates")
    min_importance: Optional[int] = Field(None, ge=0, le=100, description="Minimum importance score")
    date_from: Optional[datetime] = Field(None, description="Published after date")
    date_to: Optional[datetime] = Field(None, description="Published before date")
    search: Optional[str] = Field(None, description="Full-text search query")


class SortParams(BaseSchema):
    """Common sort parameters."""
    
    sort_by: str = Field(default="created_at", description="Field to sort by")
    sort_order: str = Field(default="desc", pattern=r"^(asc|desc)$", description="Sort direction")


class HealthResponse(BaseSchema):
    """API health check response."""
    
    status: str = "healthy"
    version: str
    timestamp: datetime


class ErrorResponse(BaseSchema):
    """Standard error response."""
    
    error: str
    message: str
    details: Optional[dict[str, Any]] = None


# Rebuild models for forward references
CategoryTreeResponse.model_rebuild()
ContentDetailResponse.model_rebuild()
UserDetailResponse.model_rebuild()
