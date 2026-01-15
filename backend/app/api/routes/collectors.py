"""Collector API Routes

FastAPI routes for managing content collectors.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.collectors.scheduler import get_scheduler, CollectorScheduler, HealthStatus

router = APIRouter(prefix="/collectors", tags=["collectors"])


# Response Models
class CollectorHealthResponse(BaseModel):
    """Collector health information."""
    status: str = Field(..., example="healthy")
    last_run: Optional[str] = Field(None, example="2024-01-15T10:30:00")
    last_success: Optional[str] = Field(None, example="2024-01-15T10:30:00")
    last_error: Optional[str] = Field(None, example=None)
    success_count: int = Field(..., example=42)
    error_count: int = Field(..., example=2)


class CollectorResponse(BaseModel):
    """Collector details."""
    collector_id: str = Field(..., example="rss:tech-feeds")
    config: Dict[str, Any]
    status: str = Field(..., example="idle")
    metrics: Dict[str, Any]
    health: CollectorHealthResponse


class CollectorListResponse(BaseModel):
    """List of collectors."""
    collectors: List[CollectorResponse]
    total: int


class SchedulerStatsResponse(BaseModel):
    """Overall scheduler statistics."""
    total_collectors: int = Field(..., example=5)
    healthy: int = Field(..., example=4)
    unhealthy: int = Field(..., example=1)
    disabled: int = Field(..., example=0)
    running: int = Field(..., example=1)
    scheduler_running: bool = Field(..., example=True)


class TriggerResponse(BaseModel):
    """Response from triggering a collector."""
    triggered: bool = Field(..., example=True)
    message: str = Field(..., example="Collector triggered successfully")


# Dependency
def get_collector_scheduler() -> CollectorScheduler:
    """Dependency to get scheduler instance."""
    try:
        return get_scheduler()
    except RuntimeError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Scheduler not initialized",
        )


# Routes
@router.get("", response_model=CollectorListResponse)
async def list_collectors(
    scheduler: CollectorScheduler = Depends(get_collector_scheduler),
) -> CollectorListResponse:
    """List all registered collectors with their health status."""
    collectors = scheduler.list_collectors()
    return CollectorListResponse(
        collectors=collectors,
        total=len(collectors),
    )


@router.get("/health", response_model=SchedulerStatsResponse)
async def get_scheduler_health(
    scheduler: CollectorScheduler = Depends(get_collector_scheduler),
) -> SchedulerStatsResponse:
    """Get overall scheduler health and statistics."""
    return SchedulerStatsResponse(**scheduler.get_stats())


@router.get("/{collector_id}", response_model=CollectorResponse)
async def get_collector(
    collector_id: str,
    scheduler: CollectorScheduler = Depends(get_collector_scheduler),
) -> CollectorResponse:
    """Get details for a specific collector."""
    collectors = scheduler.list_collectors()
    for collector in collectors:
        if collector["collector_id"] == collector_id:
            return CollectorResponse(**collector)
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Collector not found: {collector_id}",
    )


@router.post("/{collector_id}/trigger", response_model=TriggerResponse)
async def trigger_collector(
    collector_id: str,
    scheduler: CollectorScheduler = Depends(get_collector_scheduler),
) -> TriggerResponse:
    """Manually trigger a collector run."""
    success = await scheduler.trigger_collector(collector_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collector not found: {collector_id}",
        )
    
    return TriggerResponse(
        triggered=True,
        message=f"Collector {collector_id} triggered successfully",
    )


@router.put("/{collector_id}/enable", response_model=TriggerResponse)
async def enable_collector(
    collector_id: str,
    scheduler: CollectorScheduler = Depends(get_collector_scheduler),
) -> TriggerResponse:
    """Enable a disabled collector."""
    success = scheduler.enable_collector(collector_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collector not found: {collector_id}",
        )
    
    return TriggerResponse(
        triggered=True,
        message=f"Collector {collector_id} enabled",
    )


@router.put("/{collector_id}/disable", response_model=TriggerResponse)
async def disable_collector(
    collector_id: str,
    scheduler: CollectorScheduler = Depends(get_collector_scheduler),
) -> TriggerResponse:
    """Disable a collector (stops scheduled runs)."""
    success = scheduler.disable_collector(collector_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collector not found: {collector_id}",
        )
    
    return TriggerResponse(
        triggered=True,
        message=f"Collector {collector_id} disabled",
    )


@router.delete("/{collector_id}", response_model=TriggerResponse)
async def remove_collector(
    collector_id: str,
    scheduler: CollectorScheduler = Depends(get_collector_scheduler),
) -> TriggerResponse:
    """Remove a collector from the scheduler."""
    success = scheduler.remove_collector(collector_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collector not found: {collector_id}",
        )
    
    return TriggerResponse(
        triggered=True,
        message=f"Collector {collector_id} removed",
    )
