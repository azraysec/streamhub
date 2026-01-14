"""
StreamHub Test Suite
====================

Comprehensive test suite for the StreamHub content aggregation platform.

Test Organization:
-----------------
- conftest.py: Pytest fixtures (database, Redis, FastAPI client, etc.)
- factories.py: Test data factories using factory_boy
- pytest.ini: Pytest configuration and markers

Test Directories (to be created):
---------------------------------
- tests/unit/: Unit tests for individual components
- tests/integration/: Integration tests with database/Redis
- tests/e2e/: End-to-end API tests
- tests/connectors/: Source connector tests

Running Tests:
--------------
    # Run all tests
    pytest

    # Run with coverage
    pytest --cov=app --cov-report=html

    # Run specific markers
    pytest -m unit           # Unit tests only
    pytest -m integration    # Integration tests only
    pytest -m "not slow"     # Skip slow tests

    # Run specific test file
    pytest tests/unit/test_content.py

    # Run with verbose output
    pytest -v

Dependencies:
-------------
    pip install pytest pytest-asyncio pytest-cov pytest-timeout
    pip install factory-boy faker
    pip install httpx  # For async test client
"""

from tests.factories import (
    CategoryFactory,
    ContentItemFactory,
    EmbeddingFactory,
    QueueMessageFactory,
    SourceConfigFactory,
    UserFactory,
    create_content_batch,
    create_duplicate_pair,
    create_source_with_content,
)

__all__ = [
    # Factories
    "UserFactory",
    "ContentItemFactory",
    "SourceConfigFactory",
    "CategoryFactory",
    "EmbeddingFactory",
    "QueueMessageFactory",
    # Helper functions
    "create_content_batch",
    "create_duplicate_pair",
    "create_source_with_content",
]
