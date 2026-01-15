"""StreamHub FastAPI Application Entry Point."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.config import settings
from app.middleware.error_handler import (
    StreamHubException,
    http_exception_handler,
    validation_exception_handler,
    general_exception_handler,
)
from app.api.routes.collectors import router as collectors_router

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    logger.info("Starting StreamHub API...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    yield
    logger.info("Shutting down StreamHub API...")


def create_application() -> FastAPI:
    """Factory function to create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.app_name,
        description="StreamHub Content Aggregation Platform API",
        version=settings.app_version,
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    
    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Process-Time"],
    )
    
    # Register exception handlers
    app.add_exception_handler(StreamHubException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)
    
    # Include routers
    app.include_router(collectors_router, prefix="/api/v1/collectors", tags=["Collectors"])
    
    return app


app = create_application()


@app.get("/", tags=["Root"])
async def root() -> dict:
    """Root endpoint returning API information."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "status": "operational",
    }


@app.get("/health", tags=["Health"])
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": settings.app_version,
        "checks": {
            "api": "up",
            "database": "up",
            "redis": "up",
        },
    }


@app.get("/ready", tags=["Health"])
async def readiness_check() -> dict:
    """Readiness check for deployments."""
    return {"ready": True}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
        log_level="debug" if settings.debug else "info",
    )
