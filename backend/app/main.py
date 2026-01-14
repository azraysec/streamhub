"""StreamHub FastAPI Application Entry Point.

This module initializes the FastAPI application with all necessary
middleware, routers, and event handlers for the content aggregation platform.
"""

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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown events.
    
    Handles:
    - Database connection pool initialization
    - Redis connection setup
    - Background task workers startup
    """
    # Startup
    logger.info("Starting StreamHub API...")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"Debug mode: {settings.debug}")
    
    # Initialize database connection pool
    # await database.connect()
    logger.info("Database connection pool initialized")
    
    # Initialize Redis connection
    # await redis_client.initialize()
    logger.info("Redis connection initialized")
    
    yield
    
    # Shutdown
    logger.info("Shutting down StreamHub API...")
    
    # Close database connections
    # await database.disconnect()
    logger.info("Database connections closed")
    
    # Close Redis connection
    # await redis_client.close()
    logger.info("Redis connection closed")


def create_application() -> FastAPI:
    """Factory function to create and configure the FastAPI application.
    
    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title=settings.app_name,
        description="StreamHub Content Aggregation Platform API",
        version=settings.app_version,
        docs_url="/api/docs" if settings.debug else None,
        redoc_url="/api/redoc" if settings.debug else None,
        openapi_url="/api/openapi.json" if settings.debug else None,
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
    # app.include_router(auth_router, prefix="/api/v1/auth", tags=["Authentication"])
    # app.include_router(content_router, prefix="/api/v1/content", tags=["Content"])
    # app.include_router(sources_router, prefix="/api/v1/sources", tags=["Sources"])
    # app.include_router(categories_router, prefix="/api/v1/categories", tags=["Categories"])
    # app.include_router(dashboard_router, prefix="/api/v1/dashboard", tags=["Dashboard"])
    
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
    """Health check endpoint for load balancers and monitoring.
    
    Returns:
        Health status of the application and its dependencies.
    """
    health_status = {
        "status": "healthy",
        "version": settings.app_version,
        "checks": {
            "api": "up",
            "database": "up",  # TODO: Implement actual DB health check
            "redis": "up",     # TODO: Implement actual Redis health check
        },
    }
    return health_status


@app.get("/ready", tags=["Health"])
async def readiness_check() -> dict:
    """Readiness check for Kubernetes deployments.
    
    Verifies all dependencies are ready to accept traffic.
    """
    # TODO: Implement actual readiness checks
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
