"""Base model class with common fields and utilities for StreamHub."""

from datetime import datetime
from typing import Any

from sqlalchemy import MetaData, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Naming convention for database constraints
convention = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

metadata = MetaData(naming_convention=convention)


class Base(DeclarativeBase):
    """Base class for all SQLAlchemy models.
    
    Provides common functionality and metadata configuration
    for all database models in the StreamHub application.
    """
    
    metadata = metadata
    
    def to_dict(self) -> dict[str, Any]:
        """Convert model instance to dictionary.
        
        Returns:
            Dictionary representation of the model.
        """
        return {
            column.name: getattr(self, column.name)
            for column in self.__table__.columns
        }


class TimestampMixin:
    """Mixin providing created_at and updated_at timestamp fields.
    
    Automatically sets created_at on insert and updated_at on update.
    """
    
    created_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        nullable=False,
        doc="Timestamp when the record was created"
    )
    updated_at: Mapped[datetime] = mapped_column(
        default=func.now(),
        onupdate=func.now(),
        nullable=False,
        doc="Timestamp when the record was last updated"
    )
