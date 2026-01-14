"""Category model for hierarchical content organization."""

import uuid
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base, TimestampMixin

if TYPE_CHECKING:
    from .content import Content


class Category(Base, TimestampMixin):
    """Hierarchical categories for organizing and filtering content.
    
    Categories support a tree structure through self-referential
    parent_id relationship. Content can be assigned to multiple
    categories with confidence scores from LLM classification.
    
    Attributes:
        id: Primary key identifier.
        name: Human-readable category name.
        slug: URL-friendly identifier (unique).
        description: Optional description of the category.
        parent_id: Reference to parent category for hierarchy.
        sort_order: Display order within the same hierarchy level.
        created_at: When the category was created.
        updated_at: When the category was last modified.
        parent: Parent category relationship.
        children: Child categories relationship.
        contents: Content items in this category.
    """
    
    __tablename__ = "categories"
    
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        doc="Human-readable category name"
    )
    slug: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        doc="URL-friendly identifier for the category"
    )
    description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        doc="Optional description of the category"
    )
    parent_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("categories.id", ondelete="SET NULL"),
        nullable=True,
        doc="Reference to parent category for hierarchical structure"
    )
    sort_order: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        doc="Display order within the same hierarchy level"
    )
    
    # Self-referential relationships for hierarchy
    parent: Mapped[Optional["Category"]] = relationship(
        "Category",
        back_populates="children",
        remote_side="Category.id",
    )
    children: Mapped[list["Category"]] = relationship(
        "Category",
        back_populates="parent",
        cascade="all, delete-orphan",
    )
    
    # Many-to-many relationship with content
    contents: Mapped[list["Content"]] = relationship(
        "Content",
        secondary="content_categories",
        back_populates="categories",
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_categories_parent", "parent_id"),
        Index("idx_categories_slug", "slug"),
        Index("idx_categories_sort", "sort_order"),
    )
    
    def __repr__(self) -> str:
        return f"<Category(id={self.id}, name='{self.name}', slug='{self.slug}')>"
    
    def get_ancestors(self) -> list["Category"]:
        """Get all ancestor categories from this category to root.
        
        Returns:
            List of ancestor categories, starting from immediate parent.
        """
        ancestors = []
        current = self.parent
        while current is not None:
            ancestors.append(current)
            current = current.parent
        return ancestors
    
    def get_descendants(self) -> list["Category"]:
        """Get all descendant categories (children, grandchildren, etc.).
        
        Returns:
            Flat list of all descendant categories.
        """
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants


class ContentCategory(Base):
    """Association table linking content to categories with confidence scores.
    
    This junction table tracks which categories are assigned to content,
    along with the LLM classification confidence score.
    
    Attributes:
        content_id: Foreign key to content.
        category_id: Foreign key to category.
        confidence: LLM classification confidence (0-1).
        assigned_at: When the category was assigned.
    """
    
    __tablename__ = "content_categories"
    
    content_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("content.id", ondelete="CASCADE"),
        primary_key=True,
        doc="Reference to the content item"
    )
    category_id: Mapped[int] = mapped_column(
        ForeignKey("categories.id", ondelete="CASCADE"),
        primary_key=True,
        doc="Reference to the category"
    )
    confidence: Mapped[Decimal] = mapped_column(
        default=Decimal("1.0"),
        doc="LLM classification confidence score (0-1)"
    )
    assigned_at: Mapped[datetime] = mapped_column(
        default=datetime.utcnow,
        nullable=False,
        doc="When the category was assigned to the content"
    )
    
    # Indexes
    __table_args__ = (
        Index("idx_content_categories_content", "content_id"),
        Index("idx_content_categories_category", "category_id"),
        Index("idx_content_categories_confidence", "confidence", postgresql_using="btree"),
    )
    
    def __repr__(self) -> str:
        return f"<ContentCategory(content_id={self.content_id}, category_id={self.category_id}, confidence={self.confidence})>"
