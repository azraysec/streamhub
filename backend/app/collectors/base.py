"""Base Collector Module

Provides the abstract base class for all content collectors.
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Generic, TypeVar, Optional, List
from uuid import uuid4

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.content import Content
from app.models.source import Source

logger = logging.getLogger(__name__)

RawDataT = TypeVar("RawDataT")


class CollectorStatus(str, Enum):
    """Status of a collector run."""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"


@dataclass
class CollectorConfig:
    """Configuration for a collector."""
    source_id: str
    source_type: str
    name: str
    enabled: bool = True
    interval_seconds: int = 300  # 5 minutes default
    rate_limit_requests: int = 10
    rate_limit_period: int = 60
    batch_size: int = 100
    max_retries: int = 5
    retry_base_delay: float = 1.0
    retry_max_delay: float = 300.0
    timeout_seconds: int = 30
    extra_config: dict = field(default_factory=dict)


@dataclass
class CollectorMetrics:
    """Metrics tracked for a collector."""
    total_collected: int = 0
    total_stored: int = 0
    total_duplicates: int = 0
    total_errors: int = 0
    last_run_at: Optional[datetime] = None
    last_success_at: Optional[datetime] = None
    last_error_at: Optional[datetime] = None
    last_error_message: Optional[str] = None
    average_duration_ms: float = 0.0
    run_count: int = 0

    def record_run(self, duration_ms: float, collected: int, stored: int, duplicates: int):
        """Record metrics from a collection run."""
        self.run_count += 1
        self.total_collected += collected
        self.total_stored += stored
        self.total_duplicates += duplicates
        self.last_run_at = datetime.utcnow()
        self.last_success_at = datetime.utcnow()
        # Running average
        self.average_duration_ms = (
            (self.average_duration_ms * (self.run_count - 1) + duration_ms) / self.run_count
        )

    def record_error(self, error: str):
        """Record an error."""
        self.total_errors += 1
        self.last_error_at = datetime.utcnow()
        self.last_error_message = error


class BaseCollector(ABC, Generic[RawDataT]):
    """Abstract base class for content collectors.
    
    Subclasses must implement:
    - collect(): Fetch raw data from the source
    - transform(raw_data): Convert raw data to Content objects
    """

    def __init__(
        self,
        config: CollectorConfig,
        db_session: AsyncSession,
        redis_client: redis.Redis,
    ):
        self.config = config
        self.db = db_session
        self.redis = redis_client
        self.status = CollectorStatus.IDLE
        self.metrics = CollectorMetrics()
        self._current_run_id: Optional[str] = None

    @property
    def collector_id(self) -> str:
        """Unique identifier for this collector instance."""
        return f"{self.config.source_type}:{self.config.source_id}"

    @abstractmethod
    async def collect(self) -> List[RawDataT]:
        """Fetch raw data from the source.
        
        Returns:
            List of raw data items to be transformed.
        """
        pass

    @abstractmethod
    async def transform(self, raw_data: RawDataT) -> Optional[Content]:
        """Transform a raw data item into a Content object.
        
        Args:
            raw_data: Raw data item from collect()
            
        Returns:
            Content object or None if item should be skipped.
        """
        pass

    async def store(self, contents: List[Content]) -> tuple[int, int]:
        """Store content items with deduplication.
        
        Args:
            contents: List of Content objects to store.
            
        Returns:
            Tuple of (stored_count, duplicate_count)
        """
        stored = 0
        duplicates = 0
        
        for content in contents:
            # Compute content hash for deduplication
            content_hash = self.compute_content_hash(content)
            
            # Check for duplicate
            if await self.check_duplicate(content_hash):
                duplicates += 1
                logger.debug(f"Duplicate content: {content.title[:50]}...")
                continue
            
            # Store in database
            content.content_hash = content_hash
            self.db.add(content)
            stored += 1
            
            # Cache hash for future dedup checks
            await self.redis.setex(
                f"content_hash:{content_hash}",
                86400 * 7,  # 7 days
                "1"
            )
        
        if stored > 0:
            await self.db.commit()
        
        return stored, duplicates

    def compute_content_hash(self, content: Content) -> str:
        """Compute a hash for content deduplication."""
        hash_input = f"{content.source_id}:{content.original_url}:{content.title}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:32]

    async def check_duplicate(self, content_hash: str) -> bool:
        """Check if content with this hash already exists."""
        # Check Redis cache first
        if await self.redis.exists(f"content_hash:{content_hash}"):
            return True
        
        # Fallback to database
        result = await self.db.execute(
            select(Content.id).where(Content.content_hash == content_hash).limit(1)
        )
        return result.scalar_one_or_none() is not None

    async def run(self) -> dict:
        """Execute a full collection cycle.
        
        Returns:
            Dict with run statistics.
        """
        self._current_run_id = str(uuid4())
        self.status = CollectorStatus.RUNNING
        start_time = datetime.utcnow()
        
        logger.info(
            f"Starting collection run",
            extra={
                "collector_id": self.collector_id,
                "run_id": self._current_run_id,
            }
        )
        
        retry_count = 0
        last_error = None
        
        while retry_count <= self.config.max_retries:
            try:
                # Collect raw data
                raw_items = await asyncio.wait_for(
                    self.collect(),
                    timeout=self.config.timeout_seconds
                )
                
                # Transform to Content objects
                contents = []
                for item in raw_items:
                    try:
                        content = await self.transform(item)
                        if content:
                            contents.append(content)
                    except Exception as e:
                        logger.warning(f"Transform error: {e}")
                        continue
                
                # Store with deduplication
                stored, duplicates = await self.store(contents)
                
                # Calculate duration
                duration = (datetime.utcnow() - start_time).total_seconds() * 1000
                
                # Update metrics
                self.metrics.record_run(
                    duration_ms=duration,
                    collected=len(raw_items),
                    stored=stored,
                    duplicates=duplicates,
                )
                
                self.status = CollectorStatus.SUCCESS
                
                result = {
                    "run_id": self._current_run_id,
                    "collector_id": self.collector_id,
                    "status": "success",
                    "collected": len(raw_items),
                    "transformed": len(contents),
                    "stored": stored,
                    "duplicates": duplicates,
                    "duration_ms": duration,
                }
                
                logger.info("Collection run completed", extra=result)
                return result
                
            except asyncio.TimeoutError:
                last_error = "Collection timeout"
                logger.warning(f"Collection timeout, attempt {retry_count + 1}")
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Collection error: {e}, attempt {retry_count + 1}")
            
            # Exponential backoff
            retry_count += 1
            if retry_count <= self.config.max_retries:
                delay = min(
                    self.config.retry_base_delay * (2 ** (retry_count - 1)),
                    self.config.retry_max_delay
                )
                await asyncio.sleep(delay)
        
        # All retries exhausted
        self.status = CollectorStatus.FAILED
        self.metrics.record_error(last_error or "Unknown error")
        
        duration = (datetime.utcnow() - start_time).total_seconds() * 1000
        result = {
            "run_id": self._current_run_id,
            "collector_id": self.collector_id,
            "status": "failed",
            "error": last_error,
            "duration_ms": duration,
            "retries": retry_count,
        }
        
        logger.error("Collection run failed", extra=result)
        return result

    async def get_source(self) -> Optional[Source]:
        """Get the Source model for this collector."""
        result = await self.db.execute(
            select(Source).where(Source.id == self.config.source_id)
        )
        return result.scalar_one_or_none()

    def to_dict(self) -> dict:
        """Export collector state as dictionary."""
        return {
            "collector_id": self.collector_id,
            "config": {
                "source_id": self.config.source_id,
                "source_type": self.config.source_type,
                "name": self.config.name,
                "enabled": self.config.enabled,
                "interval_seconds": self.config.interval_seconds,
            },
            "status": self.status.value,
            "metrics": {
                "total_collected": self.metrics.total_collected,
                "total_stored": self.metrics.total_stored,
                "total_duplicates": self.metrics.total_duplicates,
                "total_errors": self.metrics.total_errors,
                "last_run_at": self.metrics.last_run_at.isoformat() if self.metrics.last_run_at else None,
                "last_success_at": self.metrics.last_success_at.isoformat() if self.metrics.last_success_at else None,
                "average_duration_ms": self.metrics.average_duration_ms,
                "run_count": self.metrics.run_count,
            },
        }
