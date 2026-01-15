"""Collector Scheduler

Manages periodic execution of content collectors using APScheduler.
"""

import asyncio
import logging
import signal
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type

import redis.asyncio as redis
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger

from app.collectors.base import BaseCollector, CollectorConfig, CollectorStatus

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    UNKNOWN = "unknown"
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DISABLED = "disabled"
    RUNNING = "running"


@dataclass
class CollectorHealth:
    """Health information for a collector."""
    collector_id: str
    status: HealthStatus = HealthStatus.UNKNOWN
    last_run: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_error: Optional[str] = None
    success_count: int = 0
    error_count: int = 0
    consecutive_errors: int = 0


class CollectorScheduler:
    """Manages scheduled execution of collectors."""

    _instance: Optional["CollectorScheduler"] = None
    LOCK_PREFIX = "streamhub:collector_lock"
    LOCK_TTL = 300  # 5 minutes

    def __new__(cls, *args, **kwargs):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        redis_client: redis.Redis,
        max_concurrent: int = 5,
    ):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.redis = redis_client
        self.max_concurrent = max_concurrent
        
        self._scheduler = AsyncIOScheduler()
        self._collectors: Dict[str, BaseCollector] = {}
        self._health: Dict[str, CollectorHealth] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._running = False
        self._initialized = True

    async def start(self):
        """Start the scheduler."""
        if self._running:
            return
        
        self._scheduler.start()
        self._running = True
        
        # Setup signal handlers for graceful shutdown
        for sig in (signal.SIGTERM, signal.SIGINT):
            asyncio.get_event_loop().add_signal_handler(
                sig, lambda: asyncio.create_task(self.shutdown())
            )
        
        logger.info("Collector scheduler started")

    async def shutdown(self):
        """Gracefully shutdown the scheduler."""
        if not self._running:
            return
        
        logger.info("Shutting down collector scheduler...")
        self._scheduler.shutdown(wait=True)
        self._running = False
        
        # Close all collectors
        for collector in self._collectors.values():
            if hasattr(collector, 'close'):
                await collector.close()
        
        logger.info("Collector scheduler stopped")

    async def _acquire_lock(self, collector_id: str) -> bool:
        """Try to acquire distributed lock for collector."""
        lock_key = f"{self.LOCK_PREFIX}:{collector_id}"
        acquired = await self.redis.set(
            lock_key, "1", nx=True, ex=self.LOCK_TTL
        )
        return bool(acquired)

    async def _release_lock(self, collector_id: str):
        """Release distributed lock."""
        lock_key = f"{self.LOCK_PREFIX}:{collector_id}"
        await self.redis.delete(lock_key)

    async def _run_collector(self, collector_id: str):
        """Execute a single collector run."""
        collector = self._collectors.get(collector_id)
        if not collector:
            logger.warning(f"Collector not found: {collector_id}")
            return
        
        health = self._health.get(collector_id)
        if not health:
            health = CollectorHealth(collector_id=collector_id)
            self._health[collector_id] = health
        
        # Check if disabled
        if health.status == HealthStatus.DISABLED:
            return
        
        # Try to acquire distributed lock
        if not await self._acquire_lock(collector_id):
            logger.debug(f"Collector already running elsewhere: {collector_id}")
            return
        
        async with self._semaphore:
            health.status = HealthStatus.RUNNING
            health.last_run = datetime.utcnow()
            
            try:
                result = await collector.run()
                
                if result.get("status") == "success":
                    health.status = HealthStatus.HEALTHY
                    health.last_success = datetime.utcnow()
                    health.success_count += 1
                    health.consecutive_errors = 0
                    health.last_error = None
                else:
                    raise Exception(result.get("error", "Unknown error"))
                    
            except Exception as e:
                health.error_count += 1
                health.consecutive_errors += 1
                health.last_error = str(e)
                
                # Mark unhealthy after 3 consecutive errors
                if health.consecutive_errors >= 3:
                    health.status = HealthStatus.UNHEALTHY
                else:
                    health.status = HealthStatus.HEALTHY
                
                logger.error(
                    f"Collector error",
                    extra={"collector_id": collector_id, "error": str(e)}
                )
            finally:
                await self._release_lock(collector_id)

    def add_collector(
        self,
        collector: BaseCollector,
        interval_seconds: Optional[int] = None,
        cron_expression: Optional[str] = None,
    ):
        """Add a collector to the scheduler."""
        collector_id = collector.collector_id
        self._collectors[collector_id] = collector
        self._health[collector_id] = CollectorHealth(collector_id=collector_id)
        
        # Determine trigger
        if cron_expression:
            trigger = CronTrigger.from_crontab(cron_expression)
        else:
            interval = interval_seconds or collector.config.interval_seconds
            trigger = IntervalTrigger(seconds=interval)
        
        # Add job
        self._scheduler.add_job(
            self._run_collector,
            trigger=trigger,
            args=[collector_id],
            id=collector_id,
            replace_existing=True,
            max_instances=1,
        )
        
        logger.info(f"Added collector to scheduler: {collector_id}")

    def remove_collector(self, collector_id: str) -> bool:
        """Remove a collector from the scheduler."""
        if collector_id not in self._collectors:
            return False
        
        try:
            self._scheduler.remove_job(collector_id)
        except Exception:
            pass
        
        del self._collectors[collector_id]
        if collector_id in self._health:
            del self._health[collector_id]
        
        logger.info(f"Removed collector: {collector_id}")
        return True

    def enable_collector(self, collector_id: str) -> bool:
        """Enable a disabled collector."""
        if collector_id in self._health:
            self._health[collector_id].status = HealthStatus.UNKNOWN
            try:
                self._scheduler.resume_job(collector_id)
            except Exception:
                pass
            logger.info(f"Enabled collector: {collector_id}")
            return True
        return False

    def disable_collector(self, collector_id: str) -> bool:
        """Disable a collector."""
        if collector_id in self._health:
            self._health[collector_id].status = HealthStatus.DISABLED
            try:
                self._scheduler.pause_job(collector_id)
            except Exception:
                pass
            logger.info(f"Disabled collector: {collector_id}")
            return True
        return False

    async def trigger_collector(self, collector_id: str) -> bool:
        """Manually trigger a collector run."""
        if collector_id not in self._collectors:
            return False
        
        asyncio.create_task(self._run_collector(collector_id))
        return True

    def get_health(self, collector_id: str) -> Optional[CollectorHealth]:
        """Get health info for a collector."""
        return self._health.get(collector_id)

    def get_all_health(self) -> Dict[str, CollectorHealth]:
        """Get health info for all collectors."""
        return self._health.copy()

    def list_collectors(self) -> List[Dict[str, Any]]:
        """List all registered collectors."""
        result = []
        for collector_id, collector in self._collectors.items():
            health = self._health.get(collector_id, CollectorHealth(collector_id=collector_id))
            result.append({
                **collector.to_dict(),
                "health": {
                    "status": health.status.value,
                    "last_run": health.last_run.isoformat() if health.last_run else None,
                    "last_success": health.last_success.isoformat() if health.last_success else None,
                    "last_error": health.last_error,
                    "success_count": health.success_count,
                    "error_count": health.error_count,
                },
            })
        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get overall scheduler statistics."""
        total = len(self._collectors)
        healthy = sum(1 for h in self._health.values() if h.status == HealthStatus.HEALTHY)
        unhealthy = sum(1 for h in self._health.values() if h.status == HealthStatus.UNHEALTHY)
        disabled = sum(1 for h in self._health.values() if h.status == HealthStatus.DISABLED)
        running = sum(1 for h in self._health.values() if h.status == HealthStatus.RUNNING)
        
        return {
            "total_collectors": total,
            "healthy": healthy,
            "unhealthy": unhealthy,
            "disabled": disabled,
            "running": running,
            "scheduler_running": self._running,
        }


# Global scheduler instance
_scheduler: Optional[CollectorScheduler] = None


def get_scheduler() -> CollectorScheduler:
    """Get the global scheduler instance."""
    global _scheduler
    if _scheduler is None:
        raise RuntimeError("Scheduler not initialized. Call init_scheduler() first.")
    return _scheduler


async def init_scheduler(redis_client: redis.Redis, max_concurrent: int = 5) -> CollectorScheduler:
    """Initialize the global scheduler."""
    global _scheduler
    _scheduler = CollectorScheduler(redis_client, max_concurrent)
    await _scheduler.start()
    return _scheduler
