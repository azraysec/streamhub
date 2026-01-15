"""RSS Feed Collector

Collects content from RSS and Atom feeds.
"""

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import aiohttp
import feedparser
from bs4 import BeautifulSoup

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.content import Content
from app.collectors.base import BaseCollector, CollectorConfig

logger = logging.getLogger(__name__)


@dataclass
class FeedMetadata:
    """Metadata for a single feed."""
    url: str
    title: Optional[str] = None
    etag: Optional[str] = None
    last_modified: Optional[str] = None
    last_fetched: Optional[datetime] = None
    seen_ids: Set[str] = field(default_factory=set)


@dataclass
class FeedEntry:
    """Raw feed entry data."""
    id: str
    title: str
    link: str
    content: str
    content_html: str
    author: Optional[str]
    published: Optional[datetime]
    updated: Optional[datetime]
    categories: List[str]
    images: List[str]
    feed_url: str
    feed_title: Optional[str]


class RSSCollector(BaseCollector[FeedEntry]):
    """Collector for RSS/Atom feeds."""

    def __init__(
        self,
        config: CollectorConfig,
        db_session: AsyncSession,
        redis_client: redis.Redis,
        feed_urls: List[str],
        user_agent: str = "StreamHub/1.0 (+https://streamhub.app)",
    ):
        super().__init__(config, db_session, redis_client)
        self.feed_urls = feed_urls
        self.user_agent = user_agent
        self._feed_metadata: Dict[str, FeedMetadata] = {}
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            self._session = aiohttp.ClientSession(
                timeout=timeout,
                headers={"User-Agent": self.user_agent},
            )
        return self._session

    async def close(self):
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _get_entry_id(self, entry: dict, feed_url: str) -> str:
        """Generate a unique ID for a feed entry."""
        # Prefer entry's own ID
        if entry.get("id"):
            return entry["id"]
        # Fall back to link
        if entry.get("link"):
            return entry["link"]
        # Last resort: hash of title + feed
        title = entry.get("title", "")
        return hashlib.md5(f"{feed_url}:{title}".encode()).hexdigest()

    def _parse_date(self, entry: dict) -> Optional[datetime]:
        """Parse publication date from entry."""
        for field in ["published_parsed", "updated_parsed", "created_parsed"]:
            parsed = entry.get(field)
            if parsed:
                try:
                    return datetime(*parsed[:6])
                except (TypeError, ValueError):
                    continue
        return None

    def _extract_content(self, entry: dict) -> tuple[str, str]:
        """Extract content from entry, returning (plain_text, html)."""
        html = ""
        
        # Try content array first
        if entry.get("content"):
            for content in entry["content"]:
                if content.get("type", "").startswith("text/html") or "html" in content.get("type", ""):
                    html = content.get("value", "")
                    break
            if not html and entry["content"]:
                html = entry["content"][0].get("value", "")
        
        # Fall back to summary
        if not html:
            html = entry.get("summary", "")
        
        # Fall back to description
        if not html:
            html = entry.get("description", "")
        
        # Extract plain text
        if html:
            soup = BeautifulSoup(html, "html.parser")
            plain_text = soup.get_text(separator=" ", strip=True)
        else:
            plain_text = ""
        
        return plain_text, html

    def _extract_images(self, entry: dict, html_content: str, base_url: str) -> List[str]:
        """Extract image URLs from entry."""
        images = []
        
        # Check enclosures
        for enclosure in entry.get("enclosures", []):
            if enclosure.get("type", "").startswith("image/"):
                images.append(enclosure["href"])
        
        # Check media content
        for media in entry.get("media_content", []):
            if media.get("medium") == "image" or media.get("type", "").startswith("image/"):
                images.append(media["url"])
        
        # Check media thumbnails
        for thumb in entry.get("media_thumbnail", []):
            images.append(thumb.get("url"))
        
        # Extract from HTML content
        if html_content:
            soup = BeautifulSoup(html_content, "html.parser")
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
                if src:
                    # Make absolute URL
                    images.append(urljoin(base_url, src))
        
        # Deduplicate while preserving order
        seen = set()
        unique_images = []
        for url in images:
            if url and url not in seen:
                seen.add(url)
                unique_images.append(url)
        
        return unique_images[:10]  # Limit to 10 images

    async def _fetch_feed(self, feed_url: str) -> List[FeedEntry]:
        """Fetch and parse a single feed."""
        metadata = self._feed_metadata.get(feed_url, FeedMetadata(url=feed_url))
        entries = []
        
        try:
            session = await self._get_session()
            headers = {}
            
            # Add conditional headers
            if metadata.etag:
                headers["If-None-Match"] = metadata.etag
            if metadata.last_modified:
                headers["If-Modified-Since"] = metadata.last_modified
            
            async with session.get(feed_url, headers=headers) as response:
                if response.status == 304:
                    logger.debug(f"Feed not modified: {feed_url}")
                    return []
                
                if response.status != 200:
                    logger.warning(f"Feed fetch failed: {feed_url} ({response.status})")
                    return []
                
                content = await response.text()
                
                # Update metadata
                metadata.etag = response.headers.get("ETag")
                metadata.last_modified = response.headers.get("Last-Modified")
                metadata.last_fetched = datetime.utcnow()
            
            # Parse feed
            feed = feedparser.parse(content)
            
            if feed.bozo and not feed.entries:
                logger.warning(f"Feed parse error: {feed_url} - {feed.bozo_exception}")
                return []
            
            metadata.title = feed.feed.get("title")
            
            # Process entries
            for entry in feed.entries[:self.config.batch_size]:
                entry_id = self._get_entry_id(entry, feed_url)
                
                # Skip already seen entries
                if entry_id in metadata.seen_ids:
                    continue
                
                plain_text, html = self._extract_content(entry)
                link = entry.get("link", "")
                images = self._extract_images(entry, html, link or feed_url)
                
                feed_entry = FeedEntry(
                    id=entry_id,
                    title=entry.get("title", "Untitled"),
                    link=link,
                    content=plain_text,
                    content_html=html,
                    author=entry.get("author"),
                    published=self._parse_date(entry),
                    updated=self._parse_date(entry),
                    categories=[tag.get("term", "") for tag in entry.get("tags", [])],
                    images=images,
                    feed_url=feed_url,
                    feed_title=metadata.title,
                )
                
                entries.append(feed_entry)
                metadata.seen_ids.add(entry_id)
            
            self._feed_metadata[feed_url] = metadata
            
        except asyncio.TimeoutError:
            logger.warning(f"Feed timeout: {feed_url}")
        except aiohttp.ClientError as e:
            logger.warning(f"Feed fetch error: {feed_url} - {e}")
        except Exception as e:
            logger.error(f"Unexpected feed error: {feed_url} - {e}")
        
        return entries

    async def collect(self) -> List[FeedEntry]:
        """Collect entries from all configured feeds."""
        all_entries = []
        
        # Fetch feeds concurrently with rate limiting
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests
        
        async def fetch_with_semaphore(url: str):
            async with semaphore:
                return await self._fetch_feed(url)
        
        tasks = [fetch_with_semaphore(url) for url in self.feed_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Feed collection error: {result}")
                continue
            all_entries.extend(result)
        
        logger.info(f"Collected {len(all_entries)} entries from {len(self.feed_urls)} feeds")
        return all_entries

    async def transform(self, raw_data: FeedEntry) -> Optional[Content]:
        """Transform a feed entry to Content."""
        return Content(
            source_id=self.config.source_id,
            source_type="rss",
            original_url=raw_data.link,
            title=raw_data.title[:500] if raw_data.title else "Untitled",
            content=raw_data.content,
            content_html=raw_data.content_html,
            author=raw_data.author,
            published_at=raw_data.published or datetime.utcnow(),
            media_urls=raw_data.images,
            extra_data={
                "feed_url": raw_data.feed_url,
                "feed_title": raw_data.feed_title,
                "categories": raw_data.categories,
                "entry_id": raw_data.id,
            },
        )

    def get_state(self) -> Dict[str, Any]:
        """Export collector state for persistence."""
        return {
            "feed_metadata": {
                url: {
                    "url": meta.url,
                    "title": meta.title,
                    "etag": meta.etag,
                    "last_modified": meta.last_modified,
                    "last_fetched": meta.last_fetched.isoformat() if meta.last_fetched else None,
                    "seen_ids": list(meta.seen_ids)[-1000],  # Keep last 1000
                }
                for url, meta in self._feed_metadata.items()
            }
        }

    def load_state(self, state: Dict[str, Any]):
        """Load collector state from persistence."""
        for url, meta_dict in state.get("feed_metadata", {}).items():
            self._feed_metadata[url] = FeedMetadata(
                url=meta_dict["url"],
                title=meta_dict.get("title"),
                etag=meta_dict.get("etag"),
                last_modified=meta_dict.get("last_modified"),
                last_fetched=datetime.fromisoformat(meta_dict["last_fetched"]) if meta_dict.get("last_fetched") else None,
                seen_ids=set(meta_dict.get("seen_ids", [])),
            )
