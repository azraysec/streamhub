"""StreamHub Configuration Management.

This module provides centralized configuration using Pydantic Settings
with support for environment variables and .env files.
"""

from functools import lru_cache
from typing import List, Optional

from pydantic import Field, field_validator, PostgresDsn, RedisDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.
    
    Attributes:
        app_name: Name of the application
        app_version: Current version of the application
        environment: Deployment environment (development/staging/production)
        debug: Enable debug mode
        
        host: Server host address
        port: Server port number
        workers: Number of worker processes
        
        database_url: PostgreSQL connection string
        database_pool_size: Connection pool size
        database_max_overflow: Maximum overflow connections
        
        redis_url: Redis connection string
        redis_pool_size: Redis connection pool size
        
        jwt_secret_key: Secret key for JWT signing
        jwt_algorithm: JWT signing algorithm
        jwt_access_token_expire_minutes: Access token expiration time
        jwt_refresh_token_expire_days: Refresh token expiration time
        
        openai_api_key: OpenAI API key for LLM features
        openai_model: OpenAI model to use
        
        cors_origins: Allowed CORS origins
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # Application Settings
    app_name: str = Field(default="StreamHub", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: str = Field(
        default="development",
        description="Deployment environment",
    )
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Server Settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    workers: int = Field(default=4, description="Number of workers")
    
    # Database Settings (PostgreSQL + pgvector)
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/streamhub",
        description="PostgreSQL connection URL",
    )
    database_pool_size: int = Field(default=10, description="DB pool size")
    database_max_overflow: int = Field(default=20, description="DB max overflow")
    database_echo: bool = Field(default=False, description="Echo SQL queries")
    
    # Redis Settings
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )
    redis_pool_size: int = Field(default=10, description="Redis pool size")
    redis_cache_ttl: int = Field(default=3600, description="Default cache TTL in seconds")
    
    # JWT Authentication Settings
    jwt_secret_key: str = Field(
        default="your-super-secret-key-change-in-production",
        description="JWT secret key",
    )
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")
    jwt_access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration in minutes",
    )
    jwt_refresh_token_expire_days: int = Field(
        default=7,
        description="Refresh token expiration in days",
    )
    
    # OpenAI API Settings
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key for LLM features",
    )
    openai_model: str = Field(
        default="gpt-4o",
        description="OpenAI model to use",
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="OpenAI embedding model",
    )
    
    # Vector Search Settings
    vector_dimensions: int = Field(
        default=1536,
        description="Vector embedding dimensions",
    )
    similarity_threshold: float = Field(
        default=0.85,
        description="Similarity threshold for deduplication",
    )
    
    # CORS Settings
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        description="Allowed CORS origins",
    )
    
    # Rate Limiting
    rate_limit_requests: int = Field(
        default=100,
        description="Max requests per window",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        description="Rate limit window in seconds",
    )
    
    @field_validator("environment")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment is one of allowed values."""
        allowed = ["development", "staging", "production", "testing"]
        if v.lower() not in allowed:
            raise ValueError(f"Environment must be one of: {allowed}")
        return v.lower()
    
    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v
    
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == "development"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    
    Returns:
        Settings instance.
    """
    return Settings()


# Global settings instance
settings = get_settings()
