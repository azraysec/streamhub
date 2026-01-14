"""Tag model for flexible content labeling."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Index, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base

if TYPE_CHECKING:
    from .content import Content


class Tag(Base):
    """User-defined tags for flexible content labeling and filtering.
    
    Tags provide a lightweight way to organize content beyond
    the hierarchical category system. Each tag has a name,
    URL-friendly slug, and color for UI display.
    
    Attributes:
        id: Primary key identifier.
        name: Tag name (unique).
        slug: URL-friendly identifier (unique).
        color: Hex color code for visual display.
        created_at: When the tag was created.
        contents: Content items with this tag.
    """
    
    __tablename__ = "tags"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        doc="Tag name"
    )
    slug: Mapped[str] = mapped_column(
        String(100),
        unique=True,
        nullable=False,
        doc="URL-friendly identifier for the tag"
    )
    color: Mapped[str] = mapped_column(
        String(7),
        default="#6B7280",
        doc="Hex color code for visual display in the UI"
    )
    created_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        nullable=False,
        doc="When the tag was created"
    )
    
    # Many-to-many relationship with content
    contents: Mapped[list["Content"]] = relationship(
        "Content",
        secondary="content_tags",
        back_populates="tags",
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_tags_name", "name"),
        Index("idx_tags_slug", "slug"),
    )
    
    def __repr__(self) -> str:
        return f"<Tag(id={self.id}, name='{self.name}', color='{self.color}')>"


class ContentTag(Base):
    """Association table linking content to tags.
    
    This junction table tracks which tags are assigned to content.
    
    Attributes:
        content_id: Foreign key to content.
        tag_id: Foreign key to tag.
        assigned_at: When the tag was assigned.
    """
    
    __tablename__ = "content_tags"
    
    content_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("content.id", ondelete="CASCADE"),
        primary_key=True,
        doc="Reference to the content item"
    )
    tag_id: Mapped[int] = mapped_column(
        ForeignKey("tags.id", ondelete="CASCADE"),
        primary_key=True,
        doc="Reference to the tag"
    )
    assigned_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        nullable=False,
        doc="When the tag was assigned to the content"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_content_tags_content", "content_id"),
        Index("idx_content_tags_tag", "tag_id"),
    )
    
    def __repr__(self) -> str:
        return f"<ContentTag(content_id={self.content_id}, tag_id={self.tag_id})>"