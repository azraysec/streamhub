"""Telegram Collector

Collects content from Telegram channels and groups using Telethon.
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.content import Content
from app.collectors.base import BaseCollector, CollectorConfig

logger = logging.getLogger(__name__)

# Telethon imports - optional dependency
try:
    from telethon import TelegramClient
    from telethon.sessions import StringSession
    from telethon.tl.types import (
        Channel, Chat, User,
        Message, MessageMediaPhoto, MessageMediaDocument,
        DocumentAttributeFilename, DocumentAttributeVideo,
    )
    from telethon.errors import (
        FloodWaitError, ChannelPrivateError, ChatAdminRequiredError,
    )
    TELETHON_AVAILABLE = True
except ImportError:
    TELETHON_AVAILABLE = False
    logger.warning("Telethon not installed. Telegram collector unavailable.")


@dataclass
class TelegramMessage:
    """Raw Telegram message data."""
    id: int
    channel_id: int
    channel_title: str
    channel_username: Optional[str]
    text: str
    date: datetime
    sender_id: Optional[int]
    sender_name: Optional[str]
    views: Optional[int]
    forwards: Optional[int]
    media_type: Optional[str]
    media_urls: List[str]
    reply_to_id: Optional[int]


class TelegramCollector(BaseCollector[TelegramMessage]):
    """Collector for Telegram channels and groups."""

    def __init__(
        self,
        config: CollectorConfig,
        db_session: AsyncSession,
        redis_client: redis.Redis,
        api_id: int,
        api_hash: str,
        session_string: Optional[str] = None,
        bot_token: Optional[str] = None,
        channels: List[str] = None,
        message_limit: int = 100,
        download_media: bool = False,
        media_dir: str = "/tmp/telegram_media",
    ):
        if not TELETHON_AVAILABLE:
            raise RuntimeError("Telethon is required for Telegram collector")
        
        super().__init__(config, db_session, redis_client)
        
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_string = session_string
        self.bot_token = bot_token
        self.channels = channels or []
        self.message_limit = message_limit
        self.download_media = download_media
        self.media_dir = media_dir
        
        self._client: Optional[TelegramClient] = None
        self._connected = False

    async def _get_client(self) -> TelegramClient:
        """Get or create Telegram client."""
        if self._client is None:
            session = StringSession(self.session_string) if self.session_string else StringSession()
            self._client = TelegramClient(session, self.api_id, self.api_hash)
        
        if not self._connected:
            if self.bot_token:
                await self._client.start(bot_token=self.bot_token)
            else:
                await self._client.start()
            self._connected = True
            logger.info("Telegram client connected")
        
        return self._client

    async def close(self):
        """Disconnect the Telegram client."""
        if self._client and self._connected:
            await self._client.disconnect()
            self._connected = False
            logger.info("Telegram client disconnected")

    def _extract_sender(self, message) -> tuple[Optional[int], Optional[str]]:
        """Extract sender info from message."""
        if not message.sender:
            return None, None
        
        sender = message.sender
        sender_id = sender.id
        
        if isinstance(sender, User):
            name_parts = [sender.first_name or "", sender.last_name or ""]
            sender_name = " ".join(filter(None, name_parts)) or sender.username
        elif isinstance(sender, (Channel, Chat)):
            sender_name = sender.title
        else:
            sender_name = None
        
        return sender_id, sender_name

    async def _get_media_info(self, message) -> tuple[Optional[str], List[str]]:
        """Get media type and URLs from message."""
        if not message.media:
            return None, []
        
        media_urls = []
        media_type = None
        
        if isinstance(message.media, MessageMediaPhoto):
            media_type = "photo"
            if self.download_media:
                try:
                    os.makedirs(self.media_dir, exist_ok=True)
                    path = await message.download_media(file=self.media_dir)
                    if path:
                        media_urls.append(path)
                except Exception as e:
                    logger.warning(f"Failed to download photo: {e}")
        
        elif isinstance(message.media, MessageMediaDocument):
            doc = message.media.document
            
            # Determine type from attributes
            for attr in doc.attributes:
                if isinstance(attr, DocumentAttributeVideo):
                    media_type = "video"
                    break
                elif isinstance(attr, DocumentAttributeFilename):
                    if attr.file_name.lower().endswith(('.mp4', '.mov', '.avi')):
                        media_type = "video"
                    elif attr.file_name.lower().endswith(('.mp3', '.wav', '.ogg')):
                        media_type = "audio"
                    elif attr.file_name.lower().endswith(('.jpg', '.png', '.gif')):
                        media_type = "photo"
            
            if not media_type:
                media_type = "document"
        
        return media_type, media_urls

    async def _collect_channel(self, channel_identifier: str) -> List[TelegramMessage]:
        """Collect messages from a single channel."""
        messages = []
        client = await self._get_client()
        
        try:
            entity = await client.get_entity(channel_identifier)
            
            # Get channel info
            if isinstance(entity, (Channel, Chat)):
                channel_id = entity.id
                channel_title = entity.title
                channel_username = getattr(entity, 'username', None)
            else:
                logger.warning(f"Not a channel/group: {channel_identifier}")
                return []
            
            logger.info(f"Collecting from: {channel_title}")
            
            # Iterate through messages
            async for message in client.iter_messages(entity, limit=self.message_limit):
                if not isinstance(message, Message):
                    continue
                
                # Skip empty messages
                if not message.text and not message.media:
                    continue
                
                sender_id, sender_name = self._extract_sender(message)
                media_type, media_urls = await self._get_media_info(message)
                
                tg_message = TelegramMessage(
                    id=message.id,
                    channel_id=channel_id,
                    channel_title=channel_title,
                    channel_username=channel_username,
                    text=message.text or "",
                    date=message.date,
                    sender_id=sender_id,
                    sender_name=sender_name,
                    views=message.views,
                    forwards=message.forwards,
                    media_type=media_type,
                    media_urls=media_urls,
                    reply_to_id=message.reply_to_msg_id if message.reply_to else None,
                )
                messages.append(tg_message)
                
                # Rate limiting
                await asyncio.sleep(0.1)
            
            logger.info(f"Collected {len(messages)} messages from {channel_title}")
            
        except FloodWaitError as e:
            logger.warning(f"Flood wait: {e.seconds}s for {channel_identifier}")
            self.status = "rate_limited"
        except ChannelPrivateError:
            logger.warning(f"Channel is private: {channel_identifier}")
        except ChatAdminRequiredError:
            logger.warning(f"Admin required: {channel_identifier}")
        except Exception as e:
            logger.error(f"Error collecting from {channel_identifier}: {e}")
        
        return messages

    async def collect(self) -> List[TelegramMessage]:
        """Collect messages from all configured channels."""
        all_messages = []
        
        for channel in self.channels:
            try:
                messages = await self._collect_channel(channel)
                all_messages.extend(messages)
            except Exception as e:
                logger.error(f"Error collecting {channel}: {e}")
        
        logger.info(f"Total collected: {len(all_messages)} messages from {len(self.channels)} channels")
        return all_messages

    async def transform(self, raw_data: TelegramMessage) -> Optional[Content]:
        """Transform Telegram message to Content."""
        # Build title from first line or truncated text
        text = raw_data.text or f"[{raw_data.media_type}]" if raw_data.media_type else "[No content]"
        title_line = text.split('\n')[0][:200]
        
        return Content(
            source_id=self.config.source_id,
            source_type="telegram",
            original_url=f"https://t.me/{raw_data.channel_username}/{raw_data.id}" if raw_data.channel_username else None,
            title=title_line,
            content=raw_data.text,
            author=raw_data.sender_name,
            published_at=raw_data.date,
            media_urls=raw_data.media_urls,
            extra_data={
                "message_id": raw_data.id,
                "channel_id": raw_data.channel_id,
                "channel_title": raw_data.channel_title,
                "channel_username": raw_data.channel_username,
                "views": raw_data.views,
                "forwards": raw_data.forwards,
                "media_type": raw_data.media_type,
                "reply_to_id": raw_data.reply_to_id,
            },
        )

    async def backfill(self, channel: str, limit: int = 1000) -> Dict[str, Any]:
        """Backfill historical messages from a channel."""
        original_limit = self.message_limit
        self.message_limit = limit
        
        try:
            messages = await self._collect_channel(channel)
            contents = [await self.transform(m) for m in messages]
            contents = [c for c in contents if c is not None]
            stored, duplicates = await self.store(contents)
            
            return {
                "channel": channel,
                "collected": len(messages),
                "stored": stored,
                "duplicates": duplicates,
            }
        finally:
            self.message_limit = original_limit

    def get_session_string(self) -> Optional[str]:
        """Export session string for persistence."""
        if self._client:
            return self._client.session.save()
        return self.session_string
