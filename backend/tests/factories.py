"""
StreamHub Test Data Factories
==============================

Factory classes for generating test data using factory_boy.
These factories create realistic test data for all StreamHub models.

Usage:
    from tests.factories import ContentItemFactory, UserFactory
    
    # Create a single instance
    content = ContentItemFactory()
    
    # Create with specific attributes
    content = ContentItemFactory(source_type="telegram", category="technology")
    
    # Create a batch
    contents = ContentItemFactory.create_batch(10)
    
    # Build without saving to database
    content = ContentItemFactory.build()

Dependencies:
    pip install factory-boy faker
"""

import random
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

import factory
from factory import fuzzy
from faker import Faker

# Initialize Faker with consistent seed for reproducibility
fake = Faker()
Faker.seed(42)

# Uncomment when models are available:
# from app.models import User, ContentItem, Source, Category


# =============================================================================
# Base Factory Configuration
# =============================================================================

class BaseFactory(factory.Factory):
    """
    Base factory class with common configuration.
    
    All factories should inherit from this class to ensure
    consistent behavior and easy database integration.
    """
    
    class Meta:
        abstract = True
    
    # Uncomment for SQLAlchemy integration:
    # class Meta:
    #     sqlalchemy_session = None  # Set in conftest.py
    #     sqlalchemy_session_persistence = "commit"


class AsyncBaseFactory(factory.Factory):
    """
    Base factory for async SQLAlchemy models.
    
    Use this when working with async database sessions.
    """
    
    class Meta:
        abstract = True
    
    @classmethod
    async def _create(cls, model_class, *args, **kwargs):
        """Override create to support async sessions."""
        # Uncomment for async SQLAlchemy:
        # instance = model_class(*args, **kwargs)
        # session = cls._meta.sqlalchemy_session
        # session.add(instance)
        # await session.flush()
        # return instance
        return super()._create(model_class, *args, **kwargs)


# =============================================================================
# User Factory
# =============================================================================

class UserFactory(BaseFactory):
    """
    Factory for creating User instances.
    
    Generates realistic user data for testing authentication,
    authorization, and user-related features.
    
    Examples:
        # Regular user
        user = UserFactory()
        
        # Admin user
        admin = UserFactory(is_superuser=True)
        
        # Inactive user
        inactive = UserFactory(is_active=False)
    """
    
    class Meta:
        model = dict  # Replace with User model when available
    
    id = factory.LazyFunction(lambda: str(uuid4()))
    email = factory.LazyAttribute(lambda o: f"{o.username}@streamhub.test")
    username = factory.LazyFunction(lambda: fake.user_name())
    full_name = factory.LazyFunction(lambda: fake.name())
    hashed_password = factory.LazyFunction(
        lambda: "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4.VTtYF0K1K1K1K1"  # "password"
    )
    is_active = True
    is_superuser = False
    is_verified = True
    created_at = factory.LazyFunction(datetime.utcnow)
    updated_at = factory.LazyFunction(datetime.utcnow)
    last_login = factory.LazyFunction(
        lambda: datetime.utcnow() - timedelta(hours=random.randint(1, 24))
    )
    preferences = factory.LazyFunction(lambda: {
        "theme": "dark",
        "notifications_enabled": True,
        "email_digest": "daily",
    })
    
    class Params:
        """Traits for common user variations."""
        
        # Admin trait
        admin = factory.Trait(
            is_superuser=True,
            email=factory.LazyAttribute(lambda o: f"admin_{o.username}@streamhub.test"),
        )
        
        # New user trait (just registered)
        new = factory.Trait(
            is_verified=False,
            last_login=None,
            created_at=factory.LazyFunction(datetime.utcnow),
        )
        
        # Inactive trait
        inactive = factory.Trait(
            is_active=False,
            last_login=factory.LazyFunction(
                lambda: datetime.utcnow() - timedelta(days=90)
            ),
        )


# =============================================================================
# Content Item Factory
# =============================================================================

class ContentItemFactory(BaseFactory):
    """
    Factory for creating ContentItem instances.
    
    Generates normalized content items with realistic data
    for testing content aggregation, deduplication, and categorization.
    
    Examples:
        # Random content
        content = ContentItemFactory()
        
        # Telegram content
        content = ContentItemFactory(source_type="telegram")
        
        # Content with specific category
        content = ContentItemFactory(category="technology")
        
        # Content batch from same source
        contents = ContentItemFactory.create_batch(5, source_type="twitter")
    """
    
    class Meta:
        model = dict  # Replace with ContentItem model when available
    
    # Core fields
    id = factory.LazyFunction(lambda: str(uuid4()))
    source_type = fuzzy.FuzzyChoice([
        "telegram", "twitter", "instagram", "rss", "youtube", "reddit", "whatsapp"
    ])
    source_id = factory.LazyAttribute(
        lambda o: f"{o.source_type}_{fake.slug()}"
    )
    original_id = factory.LazyFunction(
        lambda: str(random.randint(100000, 999999))
    )
    
    # Content fields
    title = factory.LazyFunction(lambda: fake.sentence(nb_words=8))
    content = factory.LazyFunction(lambda: fake.paragraph(nb_sentences=5))
    content_type = fuzzy.FuzzyChoice(["text", "image", "video", "article", "post"])
    
    # Author and source
    author = factory.LazyFunction(lambda: fake.user_name())
    url = factory.LazyAttribute(
        lambda o: f"https://{o.source_type}.com/{o.source_id}/{o.original_id}"
    )
    
    # Media
    media_urls = factory.LazyFunction(
        lambda: [fake.image_url() for _ in range(random.randint(0, 3))]
    )
    thumbnail_url = factory.LazyFunction(
        lambda: fake.image_url() if random.random() > 0.5 else None
    )
    
    # Metadata
    metadata = factory.LazyFunction(lambda: {
        "views": random.randint(100, 100000),
        "likes": random.randint(10, 10000),
        "shares": random.randint(0, 1000),
        "comments": random.randint(0, 500),
    })
    
    # Timestamps
    published_at = factory.LazyFunction(
        lambda: datetime.utcnow() - timedelta(
            hours=random.randint(1, 72),
            minutes=random.randint(0, 59)
        )
    )
    collected_at = factory.LazyFunction(datetime.utcnow)
    processed_at = factory.LazyFunction(
        lambda: datetime.utcnow() if random.random() > 0.2 else None
    )
    
    # Classification
    category = fuzzy.FuzzyChoice([
        "technology", "cryptocurrency", "ai", "startups",
        "programming", "science", "business", "general"
    ])
    tags = factory.LazyFunction(
        lambda: [fake.word() for _ in range(random.randint(1, 5))]
    )
    language = fuzzy.FuzzyChoice(["en", "es", "fr", "de", "zh", "ja"])
    
    # Analysis
    sentiment_score = factory.LazyFunction(
        lambda: round(random.uniform(-1.0, 1.0), 2)
    )
    relevance_score = factory.LazyFunction(
        lambda: round(random.uniform(0.0, 1.0), 2)
    )
    
    # Deduplication
    content_hash = factory.LazyFunction(
        lambda: fake.sha256()[:64]
    )
    is_duplicate = False
    duplicate_of_id = None
    
    # Vector embedding (stored as list, will be pgvector in DB)
    embedding = factory.LazyFunction(
        lambda: [random.gauss(0, 1) for _ in range(1536)]
    )
    
    class Params:
        """Traits for common content variations."""
        
        # Telegram content
        telegram = factory.Trait(
            source_type="telegram",
            content_type="text",
            url=factory.LazyAttribute(
                lambda o: f"https://t.me/{o.source_id}/{o.original_id}"
            ),
        )
        
        # Twitter content
        twitter = factory.Trait(
            source_type="twitter",
            content_type="text",
            title=None,  # Tweets don't have titles
            content=factory.LazyFunction(
                lambda: fake.text(max_nb_chars=280)
            ),
        )
        
        # YouTube content
        youtube = factory.Trait(
            source_type="youtube",
            content_type="video",
            thumbnail_url=factory.LazyFunction(lambda: fake.image_url()),
            metadata=factory.LazyFunction(lambda: {
                "views": random.randint(1000, 10000000),
                "likes": random.randint(100, 100000),
                "duration_seconds": random.randint(60, 3600),
            }),
        )
        
        # RSS/Article content
        article = factory.Trait(
            source_type="rss",
            content_type="article",
            content=factory.LazyFunction(
                lambda: "\n\n".join(fake.paragraphs(nb=5))
            ),
        )
        
        # Duplicate content trait
        duplicate = factory.Trait(
            is_duplicate=True,
            duplicate_of_id=factory.LazyFunction(lambda: str(uuid4())),
        )
        
        # High engagement content
        viral = factory.Trait(
            metadata=factory.LazyFunction(lambda: {
                "views": random.randint(1000000, 10000000),
                "likes": random.randint(100000, 1000000),
                "shares": random.randint(10000, 100000),
                "comments": random.randint(5000, 50000),
            }),
            relevance_score=factory.LazyFunction(
                lambda: round(random.uniform(0.8, 1.0), 2)
            ),
        )


# =============================================================================
# Source Configuration Factory
# =============================================================================

class SourceConfigFactory(BaseFactory):
    """
    Factory for creating Source configuration instances.
    
    Generates source connector configurations for testing
    the content collection system.
    
    Examples:
        # Random source
        source = SourceConfigFactory()
        
        # Telegram source
        source = SourceConfigFactory(source_type="telegram")
        
        # Disabled source
        source = SourceConfigFactory(enabled=False)
    """
    
    class Meta:
        model = dict  # Replace with Source model when available
    
    id = factory.LazyFunction(lambda: str(uuid4()))
    name = factory.LazyFunction(lambda: f"{fake.company()} Feed")
    description = factory.LazyFunction(lambda: fake.sentence())
    source_type = fuzzy.FuzzyChoice([
        "telegram", "twitter", "instagram", "rss", "youtube", "reddit"
    ])
    enabled = True
    priority = fuzzy.FuzzyInteger(1, 10)
    
    # Configuration (varies by source type)
    config = factory.LazyAttribute(lambda o: _generate_source_config(o.source_type))
    
    # Scheduling
    poll_interval_seconds = fuzzy.FuzzyChoice([60, 120, 300, 600, 900])
    last_poll_at = factory.LazyFunction(
        lambda: datetime.utcnow() - timedelta(minutes=random.randint(1, 60))
    )
    next_poll_at = factory.LazyFunction(
        lambda: datetime.utcnow() + timedelta(minutes=random.randint(1, 15))
    )
    
    # Statistics
    total_items_collected = fuzzy.FuzzyInteger(0, 10000)
    last_error = None
    error_count = 0
    
    # Timestamps
    created_at = factory.LazyFunction(
        lambda: datetime.utcnow() - timedelta(days=random.randint(1, 365))
    )
    updated_at = factory.LazyFunction(datetime.utcnow)
    
    # Owner
    created_by_id = factory.LazyFunction(lambda: str(uuid4()))
    
    class Params:
        """Traits for specific source types."""
        
        telegram = factory.Trait(
            source_type="telegram",
            config=factory.LazyFunction(lambda: {
                "channel_id": f"@{fake.slug()}",
                "api_id": str(random.randint(100000, 999999)),
                "api_hash": fake.sha256()[:32],
                "max_messages_per_poll": 100,
            }),
        )
        
        twitter = factory.Trait(
            source_type="twitter",
            config=factory.LazyFunction(lambda: {
                "usernames": [fake.user_name() for _ in range(3)],
                "search_queries": [fake.word() for _ in range(2)],
                "bearer_token": f"AAAA{fake.sha256()[:50]}",
                "include_retweets": False,
            }),
        )
        
        rss = factory.Trait(
            source_type="rss",
            config=factory.LazyFunction(lambda: {
                "feed_urls": [
                    fake.url() + "/feed/" for _ in range(random.randint(1, 5))
                ],
                "max_items_per_feed": 50,
            }),
        )
        
        youtube = factory.Trait(
            source_type="youtube",
            config=factory.LazyFunction(lambda: {
                "channel_ids": [f"UC{fake.sha256()[:22]}" for _ in range(3)],
                "api_key": f"AIza{fake.sha256()[:35]}",
                "include_shorts": True,
            }),
        )
        
        # Error state trait
        errored = factory.Trait(
            last_error=factory.LazyFunction(lambda: fake.sentence()),
            error_count=fuzzy.FuzzyInteger(1, 10),
            enabled=False,
        )


def _generate_source_config(source_type: str) -> dict:
    """Generate appropriate config based on source type."""
    configs = {
        "telegram": {
            "channel_id": f"@{fake.slug()}",
            "api_id": str(random.randint(100000, 999999)),
            "api_hash": fake.sha256()[:32],
        },
        "twitter": {
            "usernames": [fake.user_name() for _ in range(3)],
            "bearer_token": f"AAAA{fake.sha256()[:50]}",
        },
        "instagram": {
            "usernames": [fake.user_name() for _ in range(3)],
            "session_id": fake.sha256()[:64],
        },
        "rss": {
            "feed_urls": [fake.url() + "/feed/" for _ in range(3)],
        },
        "youtube": {
            "channel_ids": [f"UC{fake.sha256()[:22]}" for _ in range(2)],
            "api_key": f"AIza{fake.sha256()[:35]}",
        },
        "reddit": {
            "subreddits": [fake.word() for _ in range(3)],
            "client_id": fake.sha256()[:24],
            "client_secret": fake.sha256()[:32],
        },
    }
    return configs.get(source_type, {})


# =============================================================================
# Category Factory
# =============================================================================

class CategoryFactory(BaseFactory):
    """
    Factory for creating Category instances.
    
    Generates content categories for testing the categorization system.
    
    Examples:
        # Random category
        category = CategoryFactory()
        
        # Category with parent
        parent = CategoryFactory()
        child = CategoryFactory(parent_id=parent["id"])
    """
    
    class Meta:
        model = dict  # Replace with Category model when available
    
    id = factory.LazyFunction(lambda: fake.slug())
    name = factory.LazyFunction(lambda: fake.word().title())
    description = factory.LazyFunction(lambda: fake.sentence())
    slug = factory.LazyAttribute(lambda o: o.id)
    
    # Hierarchy
    parent_id = None
    level = 0
    path = factory.LazyAttribute(lambda o: o.id)
    
    # Display
    icon = factory.LazyFunction(
        lambda: random.choice(["📱", "💻", "🚀", "💡", "📊", "🔬", "💰", "🌐"])
    )
    color = factory.LazyFunction(lambda: fake.hex_color())
    display_order = factory.Sequence(lambda n: n)
    
    # Settings
    is_active = True
    is_system = False
    auto_assign = True
    
    # Keywords for auto-categorization
    keywords = factory.LazyFunction(
        lambda: [fake.word() for _ in range(random.randint(3, 10))]
    )
    
    # Statistics
    content_count = fuzzy.FuzzyInteger(0, 10000)
    
    # Timestamps
    created_at = factory.LazyFunction(datetime.utcnow)
    updated_at = factory.LazyFunction(datetime.utcnow)
    
    class Params:
        """Traits for common category variations."""
        
        # System category (cannot be deleted)
        system = factory.Trait(
            is_system=True,
            auto_assign=False,
        )
        
        # Child category
        child = factory.Trait(
            level=1,
            parent_id=factory.LazyFunction(lambda: str(uuid4())),
        )


# =============================================================================
# Embedding Factory
# =============================================================================

class EmbeddingFactory(BaseFactory):
    """
    Factory for creating vector embeddings.
    
    Generates normalized 1536-dimensional vectors (OpenAI ada-002 format)
    for testing similarity search and deduplication.
    
    Examples:
        # Random embedding
        emb = EmbeddingFactory()
        
        # Similar embeddings
        emb1 = EmbeddingFactory()
        emb2 = EmbeddingFactory.create_similar(emb1["vector"])
    """
    
    class Meta:
        model = dict
    
    id = factory.LazyFunction(lambda: str(uuid4()))
    content_id = factory.LazyFunction(lambda: str(uuid4()))
    vector = factory.LazyFunction(lambda: _generate_normalized_vector(1536))
    model = "text-embedding-ada-002"
    dimensions = 1536
    created_at = factory.LazyFunction(datetime.utcnow)
    
    @classmethod
    def create_similar(cls, base_vector: list[float], similarity: float = 0.95) -> dict:
        """
        Create an embedding similar to the base vector.
        
        Args:
            base_vector: The reference vector
            similarity: Target cosine similarity (0.0 to 1.0)
        
        Returns:
            A new embedding with the specified similarity
        """
        import numpy as np
        
        base = np.array(base_vector)
        noise_scale = (1 - similarity) * 2  # Approximate
        noise = np.random.randn(len(base)) * noise_scale
        similar = base + noise
        similar = similar / np.linalg.norm(similar)
        
        return cls(vector=similar.tolist())
    
    @classmethod
    def create_dissimilar(cls, base_vector: list[float]) -> dict:
        """
        Create an embedding dissimilar to the base vector.
        
        Returns an embedding with low cosine similarity (< 0.3).
        """
        import numpy as np
        
        # Create orthogonal vector
        base = np.array(base_vector)
        random_vec = np.random.randn(len(base))
        orthogonal = random_vec - np.dot(random_vec, base) * base
        orthogonal = orthogonal / np.linalg.norm(orthogonal)
        
        return cls(vector=orthogonal.tolist())


def _generate_normalized_vector(dimensions: int) -> list[float]:
    """Generate a normalized random vector."""
    import numpy as np
    vector = np.random.randn(dimensions)
    vector = vector / np.linalg.norm(vector)
    return vector.tolist()


# =============================================================================
# Queue Message Factory
# =============================================================================

class QueueMessageFactory(BaseFactory):
    """
    Factory for creating Redis Stream queue messages.
    
    Generates messages for testing the async processing pipeline.
    """
    
    class Meta:
        model = dict
    
    id = factory.LazyFunction(
        lambda: f"{int(datetime.utcnow().timestamp() * 1000)}-{random.randint(0, 999)}"
    )
    stream_name = fuzzy.FuzzyChoice([
        "content:raw",
        "content:process",
        "content:categorize",
        "content:embed",
        "content:notify",
    ])
    payload = factory.LazyFunction(lambda: {
        "content_id": str(uuid4()),
        "action": random.choice(["process", "categorize", "embed", "notify"]),
        "priority": random.randint(1, 10),
        "retry_count": 0,
    })
    created_at = factory.LazyFunction(datetime.utcnow)
    
    class Params:
        """Traits for different message types."""
        
        # High priority message
        urgent = factory.Trait(
            payload=factory.LazyFunction(lambda: {
                "content_id": str(uuid4()),
                "action": "process",
                "priority": 10,
                "retry_count": 0,
            }),
        )
        
        # Retry message
        retry = factory.Trait(
            payload=factory.LazyFunction(lambda: {
                "content_id": str(uuid4()),
                "action": "process",
                "priority": 5,
                "retry_count": random.randint(1, 3),
                "last_error": fake.sentence(),
            }),
        )


# =============================================================================
# API Response Factories
# =============================================================================

class PaginatedResponseFactory(BaseFactory):
    """
    Factory for creating paginated API responses.
    
    Useful for testing pagination handling in API tests.
    """
    
    class Meta:
        model = dict
    
    items = factory.LazyFunction(
        lambda: ContentItemFactory.create_batch(10)
    )
    total = fuzzy.FuzzyInteger(10, 1000)
    page = 1
    page_size = 10
    pages = factory.LazyAttribute(
        lambda o: (o.total + o.page_size - 1) // o.page_size
    )
    has_next = factory.LazyAttribute(lambda o: o.page < o.pages)
    has_prev = factory.LazyAttribute(lambda o: o.page > 1)


# =============================================================================
# Helper Functions
# =============================================================================

def create_content_batch(
    count: int = 10,
    source_type: str | None = None,
    category: str | None = None,
    **kwargs: Any,
) -> list[dict]:
    """
    Create a batch of content items with optional filters.
    
    Args:
        count: Number of items to create
        source_type: Filter by source type
        category: Filter by category
        **kwargs: Additional attributes to set
    
    Returns:
        List of content item dictionaries
    """
    factory_kwargs = {**kwargs}
    if source_type:
        factory_kwargs["source_type"] = source_type
    if category:
        factory_kwargs["category"] = category
    
    return ContentItemFactory.create_batch(count, **factory_kwargs)


def create_duplicate_pair(
    similarity: float = 0.95,
) -> tuple[dict, dict]:
    """
    Create a pair of content items that are duplicates.
    
    Args:
        similarity: How similar the content should be (0.0 to 1.0)
    
    Returns:
        Tuple of (original, duplicate) content items
    """
    original = ContentItemFactory()
    
    # Create duplicate with similar content
    duplicate_content = original["content"]
    if similarity < 1.0:
        # Add some variation
        words = duplicate_content.split()
        variation_count = int(len(words) * (1 - similarity))
        for _ in range(variation_count):
            if words:
                idx = random.randint(0, len(words) - 1)
                words[idx] = fake.word()
        duplicate_content = " ".join(words)
    
    duplicate = ContentItemFactory(
        content=duplicate_content,
        is_duplicate=True,
        duplicate_of_id=original["id"],
    )
    
    return original, duplicate


def create_source_with_content(
    source_type: str = "telegram",
    content_count: int = 10,
) -> tuple[dict, list[dict]]:
    """
    Create a source configuration with associated content items.
    
    Args:
        source_type: Type of source to create
        content_count: Number of content items to generate
    
    Returns:
        Tuple of (source_config, content_items)
    """
    source = SourceConfigFactory(source_type=source_type)
    contents = ContentItemFactory.create_batch(
        content_count,
        source_type=source_type,
        source_id=source["id"],
    )
    return source, contents
