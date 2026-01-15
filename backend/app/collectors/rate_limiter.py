"""Rate Limiter Module

Provides distributed rate limiting using Redis with token bucket algorithm.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Lua script for atomic token bucket operations
TOKEN_BUCKET_SCRIPT = """
local key = KEYS[1]
local max_tokens = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])  -- tokens per second
local requested = tonumber(ARGV[3])
local now = tonumber(ARGV[4])

-- Get current state
local data = redis.call('HMGET', key, 'tokens', 'last_update')
local tokens = tonumber(data[1]) or max_tokens
local last_update = tonumber(data[2]) or now

-- Calculate tokens to add based on time passed
local elapsed = now - last_update
local new_tokens = elapsed * refill_rate
tokens = math.min(max_tokens, tokens + new_tokens)

-- Check if we can grant the request
if tokens >= requested then
    tokens = tokens - requested
    redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
    redis.call('EXPIRE', key, 3600)  -- 1 hour TTL
    return {1, tokens}  -- granted
else
    redis.call('HMSET', key, 'tokens', tokens, 'last_update', now)
    redis.call('EXPIRE', key, 3600)
    local wait_time = (requested - tokens) / refill_rate
    return {0, tokens, wait_time}  -- denied, current tokens, wait time
end
"""


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    max_requests: int = 10  # Maximum tokens in bucket
    period_seconds: int = 60  # Time to refill bucket
    
    @property
    def refill_rate(self) -> float:
        """Tokens added per second."""
        return self.max_requests / self.period_seconds


@dataclass
class RateLimitStatus:
    """Current rate limit status."""
    remaining_tokens: float
    max_tokens: int
    refill_rate: float
    reset_in_seconds: Optional[float] = None
    is_limited: bool = False


class RateLimiter:
    """Distributed rate limiter using Redis token bucket."""

    KEY_PREFIX = "streamhub:ratelimit"

    def __init__(
        self,
        redis_client: redis.Redis,
        source_id: str,
        config: RateLimitConfig = None,
    ):
        self.redis = redis_client
        self.source_id = source_id
        self.config = config or RateLimitConfig()
        self._script_sha: Optional[str] = None

    @property
    def key(self) -> str:
        return f"{self.KEY_PREFIX}:{self.source_id}"

    async def _ensure_script(self):
        """Load Lua script if not already loaded."""
        if self._script_sha is None:
            self._script_sha = await self.redis.script_load(TOKEN_BUCKET_SCRIPT)

    async def acquire(
        self,
        tokens: int = 1,
        wait: bool = False,
        max_wait_seconds: float = 30.0,
    ) -> bool:
        """Attempt to acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire.
            wait: If True, wait until tokens are available.
            max_wait_seconds: Maximum time to wait if wait=True.
            
        Returns:
            True if tokens were acquired, False otherwise.
        """
        await self._ensure_script()
        start_time = time.time()

        while True:
            now = time.time()
            result = await self.redis.evalsha(
                self._script_sha,
                1,
                self.key,
                self.config.max_requests,
                self.config.refill_rate,
                tokens,
                now,
            )

            granted = result[0] == 1
            
            if granted:
                return True
            
            if not wait:
                return False
            
            # Calculate wait time
            wait_time = result[2] if len(result) > 2 else 1.0
            elapsed = time.time() - start_time
            
            if elapsed + wait_time > max_wait_seconds:
                logger.warning(
                    f"Rate limit wait exceeded max time",
                    extra={"source_id": self.source_id, "wait_time": wait_time}
                )
                return False
            
            logger.debug(
                f"Rate limited, waiting {wait_time:.2f}s",
                extra={"source_id": self.source_id}
            )
            await asyncio.sleep(min(wait_time, max_wait_seconds - elapsed))

    async def get_status(self) -> RateLimitStatus:
        """Get current rate limit status."""
        data = await self.redis.hgetall(self.key)
        
        if not data:
            return RateLimitStatus(
                remaining_tokens=float(self.config.max_requests),
                max_tokens=self.config.max_requests,
                refill_rate=self.config.refill_rate,
                is_limited=False,
            )
        
        tokens = float(data.get(b"tokens", self.config.max_requests))
        last_update = float(data.get(b"last_update", time.time()))
        
        # Calculate current tokens
        elapsed = time.time() - last_update
        current_tokens = min(
            self.config.max_requests,
            tokens + elapsed * self.config.refill_rate
        )
        
        # Calculate time to full refill
        if current_tokens < self.config.max_requests:
            reset_in = (self.config.max_requests - current_tokens) / self.config.refill_rate
        else:
            reset_in = None
        
        return RateLimitStatus(
            remaining_tokens=current_tokens,
            max_tokens=self.config.max_requests,
            refill_rate=self.config.refill_rate,
            reset_in_seconds=reset_in,
            is_limited=current_tokens < 1,
        )

    async def reset(self):
        """Reset rate limit to full bucket."""
        await self.redis.delete(self.key)
        logger.info(f"Rate limit reset", extra={"source_id": self.source_id})


class MultiSourceRateLimiter:
    """Manages rate limiters for multiple sources."""

    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self._limiters: Dict[str, RateLimiter] = {}

    def get_limiter(
        self,
        source_id: str,
        config: RateLimitConfig = None,
    ) -> RateLimiter:
        """Get or create a rate limiter for a source."""
        if source_id not in self._limiters:
            self._limiters[source_id] = RateLimiter(
                self.redis,
                source_id,
                config,
            )
        return self._limiters[source_id]

    async def acquire(
        self,
        source_id: str,
        tokens: int = 1,
        wait: bool = False,
        config: RateLimitConfig = None,
    ) -> bool:
        """Acquire tokens for a source."""
        limiter = self.get_limiter(source_id, config)
        return await limiter.acquire(tokens, wait)

    async def get_all_status(self) -> Dict[str, RateLimitStatus]:
        """Get status for all tracked limiters."""
        return {
            source_id: await limiter.get_status()
            for source_id, limiter in self._limiters.items()
        }
