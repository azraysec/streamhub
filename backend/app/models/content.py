"""Content model for unified storage of aggregated content."""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean, CheckConstraint, ForeignKey, Index, 
    SmallInteger, String, Text, UniqueConstraint
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin
from .enums import ContentType

if TYPE_CHECKING:
    from .category import Category
    from .source import Source
    from .tag import Tag


class Content(Base, TimestampMixin):
    """Unified storage for all aggregated content from various sources.
    
    Content items are collected from multiple platforms and normalized
    into this unified schema. Vector embeddings are used for semantic
    similarity search and deduplication.
    
    Attributes:
        id: UUID primary key.
        source_id: Foreign key to the source.
        external_id: Unique identifier from the source platform.
        content_type: Type of content (text, image, video, link).
        title: Optional title or headline.
        body: Main content text.
        author: Author name or username.
        url: Original URL of the content.
        media_urls: JSON array of media URLs.
        embedding: Vector embedding for similarity search (1536 dimensions).
        importance_score: Calculated importance score (0-100).
        is_duplicate: Whether this is a duplicate of existing content.
        raw_data: Original raw data from the source.
        published_at: When the content was originally published.
        created_at: When this record was created.
        updated_at: When this record was last modified.
    """
    
    __tablename__ = "content"
    
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        doc="Unique identifier for the content item"
    )
    source_id: Mapped[int] = mapped_column(
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        doc="Reference to the source this content came from"
    )
    external_id: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        doc="Unique identifier from the source platform (tweet ID, message ID, etc.)"
    )
    content_type: Mapped[ContentType] = mapped_column(
        nullable=False,
        doc="Type of content"
    )
    title: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Title or headline of the content"
    )
    body: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Main content text or description"
    )
    author: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        doc="Author name or username"
    )
    url: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Original URL of the content"
    )
    media_urls: Mapped[list[str]] = mapped_column(
        JSONB,
        default=list,
        doc="Array of media URLs (images, videos) associated with this content"
    )
    embedding: Mapped[Optional[list[float]]] = mapped_column(
        Vector(1536),
        nullable=True,
        doc="Vector embedding (1536 dimensions for OpenAI ada-002) for semantic similarity"
    )
    importance_score: Mapped[int] = mapped_column(
        SmallInteger,
        default=50,
        doc="Calculated importance score (0-100) based on engagement, relevance, etc."
    )
    is_duplicate: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        doc="Flag indicating if this content is a duplicate of existing content"
    )
    raw_data: Mapped[dict[str, Any]] = mapped_column(
        JSONB,
        default=dict,
        doc="Original raw data from the source platform for reference"
    )
    published_at: Mapped[Optional[datetime]] = mapped_column(
        nullable=True,
        doc="When the content was originally published on the source platform"
    )
    
    # Relationships
    source: Mapped["Source"] = relationship(
        "Source",
        back_populates="contents"
    )
    categories: Mapped[list["Category"]] = relationship(
        "Category",
        secondary="content_categories",
        back_populates="contents",
    )
    tags: Mapped[list["Tag"]] = relationship(
        "Tag",
        secondary="content_tags",
        back_populates="contents",
    )
    
    # Constraints and indexes
    __table_args__ = (
        UniqueConstraint("source_id", "external_id", name="content_source_external_unique"),
        CheckConstraint(
            "importance_score >= 0 AND importance_score <= 100",
            name="content_importance_score_range"
        ),
        Index("idx_content_source_id", "source_id"),
        Index("idx_content_type", "content_type"),
        Index("idx_content_published_at", "published_at", postgresql_using="btree"),
        Index("idx_content_created_at", "created_at", postgresql_using="btree"),
        Index("idx_content_importance", "importance_score", postgresql_using="btree"),
        Index("idx_content_is_duplicate", "is_duplicate", postgresql_where=(is_duplicate == False)),
        Index("idx_content_author", "author", postgresql_where=(author != None)),
        # Vector similarity search index using IVFFlat
        Index(
            "idx_content_embedding",
            "embedding",
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"}
        ),
    )
    
    def __repr__(self) -> str:
        title_preview = self.title[:30] + "..." if self.title and len(self.title) > 30 else self.title
        return f"<Content(id={self.id}, type={self.content_type.value}, title='{title_preview}')>"
