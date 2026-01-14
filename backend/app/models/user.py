"""User models for authentication and preferences."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import Boolean, ForeignKey, Index, String
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin
from .enums import UserRole


class User(Base, TimestampMixin):
    """Dashboard users with authentication credentials and roles.
    
    Users can access the StreamHub dashboard with role-based
    permissions. Passwords are stored as bcrypt hashes.
    
    Attributes:
        id: Primary key identifier.
        email: User email address (unique, used for login).
        password_hash: Bcrypt hashed password.
        full_name: User's display name.
        role: Permission level (admin, editor, viewer).
        is_active: Whether the user account is active.
        last_login: Timestamp of last successful login.
        created_at: When the user was created.
        updated_at: When the user was last modified.
        preferences: User preferences relationship.
    """
    
    __tablename__ = "users"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        doc="User email address (used for login)"
    )
    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Bcrypt hashed password"
    )
    full_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="User's display name"
    )
    role: Mapped[UserRole] = mapped_column(
        nullable=False,
        default=UserRole.VIEWER,
        doc="User permission level (admin, editor, viewer)"
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        doc="Whether the user account is active"
    )
    last_login: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        doc="Timestamp of last successful login"
    )
    
    # Relationships
    preferences: Mapped[Optional["UserPreferences"]] = relationship(
        "UserPreferences",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_users_email", "email"),
        Index("idx_users_role", "role"),
        Index("idx_users_active", "is_active", postgresql_where=(is_active == True)),
    )
    
    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}', role={self.role.value})>"
    
    def is_admin(self) -> bool:
        """Check if user has admin role."""
        return self.role == UserRole.ADMIN
    
    def is_editor(self) -> bool:
        """Check if user has editor or higher role."""
        return self.role in (UserRole.ADMIN, UserRole.EDITOR)
    
    def can_edit_content(self) -> bool:
        """Check if user can edit content."""
        return self.is_active and self.is_editor()
    
    def can_manage_users(self) -> bool:
        """Check if user can manage other users."""
        return self.is_active and self.is_admin()


class UserPreferences(Base, TimestampMixin):
    """User-specific personalization settings.
    
    Stores user preferences as JSON including theme settings,
    notification preferences, dashboard layout, and favorites.
    
    Attributes:
        id: Primary key identifier.
        user_id: Foreign key to user (unique - one preferences per user).
        preferences: JSON object with all preference settings.
        created_at: When preferences were created.
        updated_at: When preferences were last modified.
        user: User relationship.
    """
    
    __tablename__ = "user_preferences"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        doc="Reference to the user"
    )
    preferences: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="JSON object containing theme, notifications, dashboard layout, favorites, etc."
    )
    
    # Relationships
    user: Mapped["User"] = relationship(
        "User",
        back_populates="preferences"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_user_preferences_user", "user_id"),
    )
    
    def __repr__(self) -> str:
        return f"<UserPreferences(id={self.id}, user_id={self.user_id})>"
    
    def get_preference(self, key: str, default: Any = None) -> Any:
        """Get a specific preference value.
        
        Args:
            key: Preference key to retrieve.
            default: Default value if key not found.
            
        Returns:
            Preference value or default.
        """
        return self.preferences.get(key, default)
    
    def set_preference(self, key: str, value: Any) -> None:
        """Set a specific preference value.
        
        Args:
            key: Preference key to set.
            value: Value to store.
        """
        self.preferences[key] = value
