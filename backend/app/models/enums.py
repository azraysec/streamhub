"""Enum definitions matching PostgreSQL enum types for StreamHub."""

import enum


class SourceType(str, enum.Enum):
    """Supported content source platforms."""
    
    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"
    INSTAGRAM = "instagram"
    TWITTER = "twitter"
    RSS = "rss"
    YOUTUBE = "youtube"
    REDDIT = "reddit"


class ContentType(str, enum.Enum):
    """Types of content that can be aggregated."""
    
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    LINK = "link"


class UserRole(str, enum.Enum):
    """User permission levels for dashboard access."""
    
    ADMIN = "admin"
    EDITOR = "editor"
    VIEWER = "viewer"
