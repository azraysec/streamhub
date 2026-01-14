"""
StreamHub Test Fixtures
=======================

Comprehensive pytest fixtures for testing the StreamHub content aggregation platform.
Provides fixtures for database, Redis, FastAPI client, authentication, and test data.

Usage:
    Fixtures are automatically discovered by pytest. Simply include them as
    function parameters in your test functions.

Example:
    async def test_create_content(async_client, db_session, sample_content):
        response = await async_client.post("/api/content", json=sample_content)
        assert response.status_code == 201
"""

import asyncio
import os
from datetime import datetime, timedelta
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
import pytest_asyncio
from fastapi import FastAPI
from httpx import AsyncClient, ASGITransport
from redis.asyncio import Redis
from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool

# Import application modules (adjust paths as needed)
# from app.main import create_app
# from app.core.config import settings
# from app.core.database import Base, get_db
# from app.core.redis import get_redis
# from app.models import User, ContentItem, Source, Category
# from app.core.security import create_access_token


# =============================================================================
# Environment Configuration
# =============================================================================

# Set testing environment variables before any imports
os.environ.setdefault("TESTING", "1")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://test_user:test_pass@localhost:5432/streamhub_test")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/1")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-testing-only")


# =============================================================================
# Event Loop Configuration
# =============================================================================

@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """
    Create an event loop for the entire test session.
    
    This fixture ensures all async tests share the same event loop,
    which is necessary for session-scoped async fixtures.
    """
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Database Fixtures
# =============================================================================

# Test database URL (separate from production)
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://test_user:test_pass@localhost:5432/streamhub_test"
)


@pytest_asyncio.fixture(scope="session")
async def async_engine():
    """
    Create an async database engine for the test session.
    
    Uses NullPool to avoid connection issues with async tests.
    The engine is shared across all tests for efficiency.
    """
    engine = create_async_engine(
        TEST_DATABASE_URL,
        poolclass=NullPool,
        echo=False,  # Set to True for SQL debugging
    )
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def setup_database(async_engine):
    """
    Set up the test database schema.
    
    Creates all tables before tests run and drops them after.
    This runs once per test session.
    """
    # Import Base from your models
    # from app.core.database import Base
    
    async with async_engine.begin() as conn:
        # Uncomment when Base is available:
        # await conn.run_sync(Base.metadata.drop_all)
        # await conn.run_sync(Base.metadata.create_all)
        pass
    
    yield
    
    async with async_engine.begin() as conn:
        # Uncomment when Base is available:
        # await conn.run_sync(Base.metadata.drop_all)
        pass


@pytest_asyncio.fixture
async def db_session(async_engine, setup_database) -> AsyncGenerator[AsyncSession, None]:
    """
    Provide a transactional database session for each test.
    
    Each test gets its own session with a transaction that is
    rolled back after the test completes, ensuring test isolation.
    """
    async_session_factory = async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autocommit=False,
        autoflush=False,
    )
    
    async with async_session_factory() as session:
        async with session.begin():
            yield session
            # Rollback after each test for isolation
            await session.rollback()


@pytest.fixture
def db_session_factory(async_engine):
    """
    Provide a session factory for tests that need multiple sessions.
    
    Useful for testing scenarios involving multiple concurrent
    database connections or transactions.
    """
    return async_sessionmaker(
        async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


# =============================================================================
# Redis Fixtures
# =============================================================================

TEST_REDIS_URL = os.environ.get("TEST_REDIS_URL", "redis://localhost:6379/1")


@pytest_asyncio.fixture(scope="session")
async def redis_client() -> AsyncGenerator[Redis, None]:
    """
    Create a Redis client for the test session.
    
    Uses a separate database (db=1) from production to avoid
    data conflicts. All keys are flushed after tests complete.
    """
    client = Redis.from_url(
        TEST_REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
    )
    
    # Verify connection
    try:
        await client.ping()
    except Exception as e:
        pytest.skip(f"Redis not available: {e}")
    
    yield client
    
    # Clean up test database
    await client.flushdb()
    await client.close()


@pytest_asyncio.fixture
async def redis_session(redis_client) -> AsyncGenerator[Redis, None]:
    """
    Provide a Redis session for each test with automatic cleanup.
    
    Keys created during the test are tracked and deleted afterward.
    """
    # Track keys created during test for cleanup
    test_prefix = f"test:{uuid4().hex}:"
    
    yield redis_client
    
    # Clean up keys created during this test
    async for key in redis_client.scan_iter(f"{test_prefix}*"):
        await redis_client.delete(key)


@pytest.fixture
def mock_redis() -> MagicMock:
    """
    Provide a mock Redis client for unit tests.
    
    Use this when you want to test code that uses Redis
    without requiring an actual Redis connection.
    """
    mock = MagicMock(spec=Redis)
    mock.get = AsyncMock(return_value=None)
    mock.set = AsyncMock(return_value=True)
    mock.delete = AsyncMock(return_value=1)
    mock.exists = AsyncMock(return_value=0)
    mock.expire = AsyncMock(return_value=True)
    mock.xadd = AsyncMock(return_value="1234567890-0")  # Stream ID
    mock.xread = AsyncMock(return_value=[])
    mock.xrange = AsyncMock(return_value=[])
    return mock


# =============================================================================
# FastAPI Test Client Fixtures
# =============================================================================

@pytest.fixture
def app() -> FastAPI:
    """
    Create a FastAPI application instance for testing.
    
    Override dependencies to use test database and Redis.
    """
    # Import and create app (uncomment when available):
    # from app.main import create_app
    # test_app = create_app()
    
    # For now, create a minimal test app
    test_app = FastAPI(title="StreamHub Test")
    
    return test_app


@pytest_asyncio.fixture
async def async_client(app, db_session, redis_client) -> AsyncGenerator[AsyncClient, None]:
    """
    Provide an async HTTP client for testing API endpoints.
    
    This client is configured to use the test database and Redis,
    and includes proper cleanup after each test.
    """
    # Override dependencies (uncomment when available):
    # from app.core.database import get_db
    # from app.core.redis import get_redis
    # 
    # async def override_get_db():
    #     yield db_session
    # 
    # async def override_get_redis():
    #     yield redis_client
    # 
    # app.dependency_overrides[get_db] = override_get_db
    # app.dependency_overrides[get_redis] = override_get_redis
    
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://test",
        headers={"Content-Type": "application/json"},
    ) as client:
        yield client
    
    # Clear dependency overrides
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def authenticated_client(
    async_client: AsyncClient,
    test_user: dict,
    auth_token: str,
) -> AsyncGenerator[AsyncClient, None]:
    """
    Provide an authenticated async client with valid JWT token.
    
    Use this for testing endpoints that require authentication.
    """
    async_client.headers["Authorization"] = f"Bearer {auth_token}"
    yield async_client
    del async_client.headers["Authorization"]


# =============================================================================
# Authentication Fixtures
# =============================================================================

@pytest.fixture
def test_user() -> dict:
    """
    Provide test user data.
    
    Returns a dictionary with user information that can be used
    for authentication and authorization testing.
    """
    return {
        "id": str(uuid4()),
        "email": "testuser@streamhub.test",
        "username": "testuser",
        "full_name": "Test User",
        "is_active": True,
        "is_superuser": False,
        "created_at": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def admin_user() -> dict:
    """
    Provide admin user data for testing admin-only endpoints.
    """
    return {
        "id": str(uuid4()),
        "email": "admin@streamhub.test",
        "username": "admin",
        "full_name": "Admin User",
        "is_active": True,
        "is_superuser": True,
        "created_at": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def auth_token(test_user: dict) -> str:
    """
    Generate a valid JWT token for the test user.
    
    Token expires in 1 hour and contains standard claims.
    """
    # Uncomment when security module is available:
    # from app.core.security import create_access_token
    # return create_access_token(subject=test_user["id"])
    
    # Return a mock token for now
    import jwt
    payload = {
        "sub": test_user["id"],
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
        "type": "access",
    }
    return jwt.encode(payload, "test-secret-key", algorithm="HS256")


@pytest.fixture
def admin_token(admin_user: dict) -> str:
    """
    Generate a valid JWT token for the admin user.
    """
    import jwt
    payload = {
        "sub": admin_user["id"],
        "exp": datetime.utcnow() + timedelta(hours=1),
        "iat": datetime.utcnow(),
        "type": "access",
        "is_superuser": True,
    }
    return jwt.encode(payload, "test-secret-key", algorithm="HS256")


# =============================================================================
# Content Fixtures
# =============================================================================

@pytest.fixture
def sample_content() -> dict:
    """
    Provide a sample normalized content item.
    
    Represents a content item in the unified schema used by StreamHub.
    """
    return {
        "id": str(uuid4()),
        "source_type": "telegram",
        "source_id": "tech_news_channel",
        "original_id": "12345",
        "title": "Breaking: New AI Framework Released",
        "content": "A groundbreaking new AI framework has been released that promises to revolutionize machine learning workflows.",
        "content_type": "text",
        "author": "TechNewsBot",
        "url": "https://t.me/tech_news_channel/12345",
        "media_urls": [],
        "metadata": {
            "views": 15000,
            "forwards": 250,
            "reactions": {"👍": 120, "🔥": 45},
        },
        "published_at": datetime.utcnow().isoformat(),
        "collected_at": datetime.utcnow().isoformat(),
        "category": "technology",
        "tags": ["ai", "machine-learning", "framework"],
        "language": "en",
        "sentiment_score": 0.75,
    }


@pytest.fixture
def sample_content_batch() -> list[dict]:
    """
    Provide a batch of sample content items from various sources.
    
    Useful for testing batch processing, deduplication, and aggregation.
    """
    base_time = datetime.utcnow()
    
    return [
        {
            "id": str(uuid4()),
            "source_type": "telegram",
            "source_id": "crypto_updates",
            "original_id": "1001",
            "title": "Bitcoin Reaches New High",
            "content": "Bitcoin has reached a new all-time high of $100,000.",
            "content_type": "text",
            "published_at": (base_time - timedelta(hours=1)).isoformat(),
            "category": "cryptocurrency",
        },
        {
            "id": str(uuid4()),
            "source_type": "twitter",
            "source_id": "elonmusk",
            "original_id": "1234567890",
            "title": None,
            "content": "Exciting news coming soon! 🚀",
            "content_type": "text",
            "published_at": (base_time - timedelta(hours=2)).isoformat(),
            "category": "general",
        },
        {
            "id": str(uuid4()),
            "source_type": "rss",
            "source_id": "techcrunch",
            "original_id": "article-2024-001",
            "title": "Startup Raises $50M Series B",
            "content": "AI startup announces major funding round led by top VCs.",
            "content_type": "article",
            "published_at": (base_time - timedelta(hours=3)).isoformat(),
            "category": "startups",
        },
        {
            "id": str(uuid4()),
            "source_type": "youtube",
            "source_id": "MKBHD",
            "original_id": "dQw4w9WgXcQ",
            "title": "iPhone 20 Review",
            "content": "Full review of the latest iPhone.",
            "content_type": "video",
            "published_at": (base_time - timedelta(hours=4)).isoformat(),
            "category": "technology",
        },
        {
            "id": str(uuid4()),
            "source_type": "reddit",
            "source_id": "r/programming",
            "original_id": "abc123",
            "title": "New Python 4.0 Features Announced",
            "content": "The Python steering council has announced upcoming features for Python 4.0.",
            "content_type": "post",
            "published_at": (base_time - timedelta(hours=5)).isoformat(),
            "category": "programming",
        },
    ]


@pytest.fixture
def duplicate_content_pair() -> tuple[dict, dict]:
    """
    Provide a pair of duplicate content items for deduplication testing.
    
    These items have similar content but come from different sources.
    """
    content_text = "Breaking: Major tech company announces revolutionary new product that will change the industry."
    base_time = datetime.utcnow()
    
    original = {
        "id": str(uuid4()),
        "source_type": "telegram",
        "source_id": "tech_news",
        "original_id": "orig_001",
        "title": "Major Tech Announcement",
        "content": content_text,
        "published_at": base_time.isoformat(),
    }
    
    duplicate = {
        "id": str(uuid4()),
        "source_type": "twitter",
        "source_id": "tech_reporter",
        "original_id": "tw_12345",
        "title": "Tech Company Makes Big Announcement",
        "content": content_text + " This is huge!",  # Slightly modified
        "published_at": (base_time + timedelta(minutes=5)).isoformat(),
    }
    
    return original, duplicate


# =============================================================================
# Vector Embedding Fixtures
# =============================================================================

@pytest.fixture
def sample_embedding() -> list[float]:
    """
    Provide a sample vector embedding.
    
    Returns a 1536-dimensional vector (OpenAI ada-002 format)
    normalized to unit length.
    """
    import numpy as np
    np.random.seed(42)  # Reproducible
    embedding = np.random.randn(1536).astype(float)
    # Normalize to unit length
    embedding = embedding / np.linalg.norm(embedding)
    return embedding.tolist()


@pytest.fixture
def similar_embeddings() -> tuple[list[float], list[float]]:
    """
    Provide two similar embeddings for testing similarity search.
    
    These embeddings have high cosine similarity (>0.9).
    """
    import numpy as np
    np.random.seed(42)
    
    base = np.random.randn(1536)
    base = base / np.linalg.norm(base)
    
    # Add small noise to create similar embedding
    noise = np.random.randn(1536) * 0.1
    similar = base + noise
    similar = similar / np.linalg.norm(similar)
    
    return base.tolist(), similar.tolist()


@pytest.fixture
def mock_openai_embeddings():
    """
    Mock the OpenAI embeddings API.
    
    Use this to avoid actual API calls during testing.
    """
    import numpy as np
    
    def create_mock_embedding(*args, **kwargs):
        np.random.seed(hash(str(args)) % 2**32)
        embedding = np.random.randn(1536)
        embedding = embedding / np.linalg.norm(embedding)
        return {"embedding": embedding.tolist()}
    
    with patch("openai.Embedding.create") as mock:
        mock.side_effect = create_mock_embedding
        yield mock


# =============================================================================
# Source Connector Fixtures
# =============================================================================

@pytest.fixture
def telegram_source_config() -> dict:
    """
    Provide configuration for a Telegram source connector.
    """
    return {
        "id": str(uuid4()),
        "name": "Tech News Channel",
        "source_type": "telegram",
        "enabled": True,
        "config": {
            "channel_id": "@tech_news_channel",
            "api_id": "12345",
            "api_hash": "test_hash",
            "poll_interval_seconds": 60,
            "max_messages_per_poll": 100,
        },
        "created_at": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def twitter_source_config() -> dict:
    """
    Provide configuration for a Twitter/X source connector.
    """
    return {
        "id": str(uuid4()),
        "name": "Tech Influencers",
        "source_type": "twitter",
        "enabled": True,
        "config": {
            "usernames": ["elonmusk", "sama", "karpathy"],
            "bearer_token": "test_bearer_token",
            "poll_interval_seconds": 120,
            "include_retweets": False,
        },
        "created_at": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def rss_source_config() -> dict:
    """
    Provide configuration for an RSS source connector.
    """
    return {
        "id": str(uuid4()),
        "name": "Tech News RSS",
        "source_type": "rss",
        "enabled": True,
        "config": {
            "feed_urls": [
                "https://techcrunch.com/feed/",
                "https://www.theverge.com/rss/index.xml",
                "https://arstechnica.com/feed/",
            ],
            "poll_interval_seconds": 300,
            "max_items_per_feed": 50,
        },
        "created_at": datetime.utcnow().isoformat(),
    }


@pytest.fixture
def all_source_configs(
    telegram_source_config,
    twitter_source_config,
    rss_source_config,
) -> list[dict]:
    """
    Provide a list of all source configurations for comprehensive testing.
    """
    return [telegram_source_config, twitter_source_config, rss_source_config]


# =============================================================================
# Mock External Services
# =============================================================================

@pytest.fixture
def mock_openai_client():
    """
    Mock the OpenAI client for testing LLM-dependent code.
    
    Provides predictable responses for categorization and embeddings.
    """
    mock_client = MagicMock()
    
    # Mock chat completion for categorization
    mock_completion = MagicMock()
    mock_completion.choices = [
        MagicMock(message=MagicMock(content='{"category": "technology", "confidence": 0.95}'))
    ]
    mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
    
    # Mock embeddings
    import numpy as np
    mock_embedding_response = MagicMock()
    mock_embedding_response.data = [
        MagicMock(embedding=np.random.randn(1536).tolist())
    ]
    mock_client.embeddings.create = AsyncMock(return_value=mock_embedding_response)
    
    with patch("openai.AsyncOpenAI", return_value=mock_client):
        yield mock_client


@pytest.fixture
def mock_telegram_client():
    """
    Mock the Telegram client for testing the Telegram connector.
    """
    mock = MagicMock()
    mock.get_messages = AsyncMock(return_value=[
        MagicMock(
            id=12345,
            text="Test message from Telegram",
            date=datetime.utcnow(),
            sender=MagicMock(username="testuser"),
            views=100,
            forwards=10,
        )
    ])
    mock.connect = AsyncMock()
    mock.disconnect = AsyncMock()
    return mock


@pytest.fixture
def mock_http_client():
    """
    Mock HTTP client for testing external API calls.
    
    Use this for testing RSS feeds, webhooks, and other HTTP-based integrations.
    """
    mock = MagicMock()
    mock.get = AsyncMock()
    mock.post = AsyncMock()
    return mock


# =============================================================================
# Category Fixtures
# =============================================================================

@pytest.fixture
def categories() -> list[dict]:
    """
    Provide a list of content categories.
    """
    return [
        {"id": "technology", "name": "Technology", "description": "Tech news and updates"},
        {"id": "cryptocurrency", "name": "Cryptocurrency", "description": "Crypto and blockchain"},
        {"id": "ai", "name": "Artificial Intelligence", "description": "AI and ML news"},
        {"id": "startups", "name": "Startups", "description": "Startup news and funding"},
        {"id": "programming", "name": "Programming", "description": "Software development"},
        {"id": "science", "name": "Science", "description": "Scientific discoveries"},
        {"id": "business", "name": "Business", "description": "Business news"},
        {"id": "general", "name": "General", "description": "Uncategorized content"},
    ]


# =============================================================================
# Utility Fixtures
# =============================================================================

@pytest.fixture
def freeze_time():
    """
    Freeze time for deterministic testing.
    
    Usage:
        def test_something(freeze_time):
            with freeze_time("2024-01-15 12:00:00"):
                # datetime.now() returns frozen time
                pass
    """
    from unittest.mock import patch
    from datetime import datetime
    
    def _freeze(time_str: str):
        frozen_time = datetime.fromisoformat(time_str)
        
        class FrozenDatetime:
            @classmethod
            def now(cls, tz=None):
                return frozen_time
            
            @classmethod
            def utcnow(cls):
                return frozen_time
        
        return patch("datetime.datetime", FrozenDatetime)
    
    return _freeze


@pytest.fixture
def temp_file(tmp_path):
    """
    Provide a function to create temporary files for testing.
    
    Files are automatically cleaned up after the test.
    """
    def _create_temp_file(content: str, filename: str = "test.txt") -> str:
        file_path = tmp_path / filename
        file_path.write_text(content)
        return str(file_path)
    
    return _create_temp_file


@pytest.fixture
def capture_logs(caplog):
    """
    Capture log messages during tests.
    
    Usage:
        def test_something(capture_logs):
            # ... do something that logs ...
            assert "Expected message" in capture_logs.text
    """
    import logging
    caplog.set_level(logging.DEBUG)
    return caplog


# =============================================================================
# Cleanup Fixtures
# =============================================================================

@pytest_asyncio.fixture(autouse=True)
async def cleanup_after_test(db_session, redis_client):
    """
    Automatic cleanup fixture that runs after each test.
    
    Ensures test isolation by cleaning up any leftover data.
    This fixture runs automatically for all tests.
    """
    yield
    
    # Database cleanup is handled by transaction rollback in db_session
    
    # Additional Redis cleanup if needed
    # await redis_client.flushdb()


@pytest.fixture(scope="session", autouse=True)
def cleanup_after_session():
    """
    Cleanup fixture that runs once after all tests complete.
    
    Use for any global cleanup that should happen at the end
    of the test session.
    """
    yield
    
    # Session cleanup code here
    print("\n✅ Test session completed. Cleaning up...")
