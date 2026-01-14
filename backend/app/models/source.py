"""Source model for tracking configured content sources."""

from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import String, Boolean, Index
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin
from .enums import SourceType

if TYPE_CHECKING:
    from .content import Content


class Source(Base, TimestampMixin):
    """Model representing a configured content source for aggregation.
    
    Sources can be Telegram channels, RSS feeds, Twitter accounts, etc.
    Each source has its own configuration and sync state.
    
    Attributes:
        id: Primary key identifier.
        name: Human-readable name for the source (unique).
        source_type: Type of source platform.
        config: Source-specific configuration as JSON.
        enabled: Whether the source is actively being synced.
        last_sync: Timestamp of the last successful sync.
        created_at: When the source was created.
        updated_at: When the source was last modified.
        contents: Relationship to Content items from this source.
    """
    
    __tablename__ = "sources"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        doc="Human-readable name for the source"
    )
    source_type: Mapped[SourceType] = mapped_column(
        nullable=False,
        doc="Type of source platform"
    )
    config: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        nullable=False,
        default=dict,
        doc="Source-specific configuration (API keys, channel IDs, etc.)"
    )
    enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        doc="Whether the source is actively being synced"
    )
    last_sync: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        doc="Timestamp of the last successful content sync"
    )
    
    # Relationships
    contents: Mapped[list["Content"]] = relationship(
        "Content",
        back_populates="source",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_sources_type", "source_type"),
        Index("idx_sources_enabled", "enabled", postgresql_where=(enabled == True)),
        Index("idx_sources_last_sync", "last_sync"),
    )
    
    def __repr__(self) -> str:
        return f"<Source(id={self.id}, name='{self.name}', type={self.source_type.value})>"