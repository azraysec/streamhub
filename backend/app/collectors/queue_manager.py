"""Redis Queue Manager

Provides task queue functionality using Redis Streams for distributed
collection task processing.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

import redis.asyncio as redis

logger = logging.getLogger(__name__)


class TaskPriority(str, Enum):
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    DEAD = "dead"  # Moved to DLQ after max retries


@dataclass
class Task:
    """Represents a collection task."""
    id: str
    task_type: str
    payload: Dict[str, Any]
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    stream_id: Optional[str] = None  # Redis stream message ID

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_type": self.task_type,
            "payload": json.dumps(self.payload),
            "priority": self.priority.value,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else "",
            "completed_at": self.completed_at.isoformat() if self.completed_at else "",
            "retry_count": str(self.retry_count),
            "max_retries": str(self.max_retries),
            "error_message": self.error_message or "",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], stream_id: str = None) -> "Task":
        return cls(
            id=data["id"],
            task_type=data["task_type"],
            payload=json.loads(data["payload"]) if isinstance(data["payload"], str) else data["payload"],
            priority=TaskPriority(data["priority"]),
            status=TaskStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.utcnow(),
            started_at=datetime.fromisoformat(data["started_at"]) if data.get("started_at") else None,
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            retry_count=int(data.get("retry_count", 0)),
            max_retries=int(data.get("max_retries", 3)),
            error_message=data.get("error_message") or None,
            stream_id=stream_id,
        )


class RedisQueueManager:
    """Manages task queues using Redis Streams."""

    STREAM_PREFIX = "streamhub:tasks"
    DLQ_STREAM = "streamhub:dlq"
    CONSUMER_GROUP = "collectors"

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self._initialized = False

    def _get_stream_name(self, priority: TaskPriority) -> str:
        """Get stream name for priority level."""
        return f"{self.STREAM_PREFIX}:{priority.value}"

    async def initialize(self):
        """Initialize streams and consumer groups."""
        if self._initialized:
            return

        for priority in TaskPriority:
            stream = self._get_stream_name(priority)
            try:
                await self.redis.xgroup_create(
                    stream, self.CONSUMER_GROUP, id="0", mkstream=True
                )
                logger.info(f"Created consumer group for {stream}")
            except redis.ResponseError as e:
                if "BUSYGROUP" not in str(e):
                    raise
                logger.debug(f"Consumer group already exists for {stream}")

        # Create DLQ stream
        try:
            await self.redis.xgroup_create(
                self.DLQ_STREAM, self.CONSUMER_GROUP, id="0", mkstream=True
            )
        except redis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

        self._initialized = True
        logger.info("Queue manager initialized")

    async def enqueue_task(
        self,
        task_type: str,
        payload: Dict[str, Any],
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
    ) -> Task:
        """Add a task to the queue."""
        await self.initialize()

        task = Task(
            id=str(uuid4()),
            task_type=task_type,
            payload=payload,
            priority=priority,
            max_retries=max_retries,
        )

        stream = self._get_stream_name(priority)
        stream_id = await self.redis.xadd(stream, task.to_dict())
        task.stream_id = stream_id

        logger.info(
            f"Enqueued task",
            extra={"task_id": task.id, "type": task_type, "priority": priority.value}
        )
        return task

    async def dequeue_task(
        self,
        consumer_name: str,
        block_ms: int = 5000,
    ) -> Optional[Task]:
        """Get the next task from the queue (priority ordered)."""
        await self.initialize()

        # Try each priority level in order
        for priority in [TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]:
            stream = self._get_stream_name(priority)
            
            try:
                messages = await self.redis.xreadgroup(
                    self.CONSUMER_GROUP,
                    consumer_name,
                    {stream: ">"},
                    count=1,
                    block=block_ms if priority == TaskPriority.LOW else 0,
                )
            except redis.ResponseError as e:
                logger.error(f"Error reading from stream {stream}: {e}")
                continue

            if messages:
                for stream_name, stream_messages in messages:
                    for msg_id, msg_data in stream_messages:
                        # Decode bytes to strings
                        decoded_data = {
                            k.decode() if isinstance(k, bytes) else k: 
                            v.decode() if isinstance(v, bytes) else v
                            for k, v in msg_data.items()
                        }
                        task = Task.from_dict(decoded_data, stream_id=msg_id)
                        task.status = TaskStatus.PROCESSING
                        task.started_at = datetime.utcnow()
                        
                        logger.info(
                            f"Dequeued task",
                            extra={"task_id": task.id, "consumer": consumer_name}
                        )
                        return task

        return None

    async def acknowledge_task(self, task: Task):
        """Mark a task as completed and remove from pending."""
        if not task.stream_id:
            logger.warning(f"Cannot acknowledge task without stream_id: {task.id}")
            return

        stream = self._get_stream_name(task.priority)
        await self.redis.xack(stream, self.CONSUMER_GROUP, task.stream_id)
        await self.redis.xdel(stream, task.stream_id)
        
        logger.info(f"Acknowledged task", extra={"task_id": task.id})

    async def handle_failed_task(
        self,
        task: Task,
        error: str,
    ) -> bool:
        """Handle a failed task - retry or move to DLQ.
        
        Returns:
            True if task was re-queued for retry, False if moved to DLQ.
        """
        task.retry_count += 1
        task.error_message = error
        task.status = TaskStatus.FAILED

        if task.retry_count < task.max_retries:
            # Re-queue for retry
            task.status = TaskStatus.PENDING
            stream = self._get_stream_name(task.priority)
            
            # Acknowledge original and add new
            if task.stream_id:
                await self.redis.xack(stream, self.CONSUMER_GROUP, task.stream_id)
                await self.redis.xdel(stream, task.stream_id)
            
            new_stream_id = await self.redis.xadd(stream, task.to_dict())
            task.stream_id = new_stream_id
            
            logger.warning(
                f"Task retry scheduled",
                extra={"task_id": task.id, "retry": task.retry_count, "error": error}
            )
            return True
        else:
            # Move to DLQ
            task.status = TaskStatus.DEAD
            await self.redis.xadd(self.DLQ_STREAM, task.to_dict())
            
            if task.stream_id:
                stream = self._get_stream_name(task.priority)
                await self.redis.xack(stream, self.CONSUMER_GROUP, task.stream_id)
                await self.redis.xdel(stream, task.stream_id)
            
            logger.error(
                f"Task moved to DLQ",
                extra={"task_id": task.id, "error": error}
            )
            return False

    async def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        stats = {"queues": {}, "dlq_length": 0}

        for priority in TaskPriority:
            stream = self._get_stream_name(priority)
            try:
                length = await self.redis.xlen(stream)
                stats["queues"][priority.value] = length
            except redis.ResponseError:
                stats["queues"][priority.value] = 0

        try:
            stats["dlq_length"] = await self.redis.xlen(self.DLQ_STREAM)
        except redis.ResponseError:
            pass

        return stats

    async def get_dlq_tasks(self, count: int = 100) -> List[Task]:
        """Get tasks from the dead letter queue."""
        tasks = []
        try:
            messages = await self.redis.xrange(self.DLQ_STREAM, count=count)
            for msg_id, msg_data in messages:
                decoded_data = {
                    k.decode() if isinstance(k, bytes) else k:
                    v.decode() if isinstance(v, bytes) else v
                    for k, v in msg_data.items()
                }
                task = Task.from_dict(decoded_data, stream_id=msg_id)
                tasks.append(task)
        except redis.ResponseError as e:
            logger.error(f"Error reading DLQ: {e}")
        
        return tasks

    async def retry_dlq_task(self, task_id: str) -> bool:
        """Retry a task from the DLQ."""
        dlq_tasks = await self.get_dlq_tasks(1000)
        for task in dlq_tasks:
            if task.id == task_id:
                task.status = TaskStatus.PENDING
                task.retry_count = 0
                task.error_message = None
                
                stream = self._get_stream_name(task.priority)
                await self.redis.xadd(stream, task.to_dict())
                
                if task.stream_id:
                    await self.redis.xdel(self.DLQ_STREAM, task.stream_id)
                
                logger.info(f"Retried DLQ task", extra={"task_id": task_id})
                return True
        
        return False
