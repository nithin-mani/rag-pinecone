import hashlib
import json
from typing import Optional
from functools import lru_cache

import redis

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

CACHE_TTL_SECONDS = 60 * 60  # 1 hour


def _make_serializable(obj):
    """Recursively convert non-serializable objects to strings."""
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(i) for i in obj]
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        return str(obj)


class CacheService:
    def __init__(self):
        self.settings = get_settings()
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                self._client = redis.from_url(
                    self.settings.redis_url,
                    decode_responses=True,
                    socket_timeout=2,
                )
                self._client.ping()
                logger.info("Redis connected")
            except Exception as e:
                logger.warning("Redis unavailable, caching disabled", error=str(e))
                self._client = None
        return self._client

    def _make_key(self, question: str, top_k: int) -> str:
        raw = f"{question.strip().lower()}:{top_k}"
        return "rag:" + hashlib.md5(raw.encode()).hexdigest()

    def get(self, question: str, top_k: int) -> Optional[dict]:
        client = self._get_client()
        if not client:
            return None
        try:
            key = self._make_key(question, top_k)
            cached = client.get(key)
            if cached:
                logger.info("Cache hit", question=question[:60])
                return json.loads(cached)
        except Exception as e:
            logger.warning("Cache get failed", error=str(e))
        return None

    def set(self, question: str, top_k: int, result: dict) -> None:
        client = self._get_client()
        if not client:
            return
        try:
            key = self._make_key(question, top_k)
            # Make sure everything is serializable
            clean_result = _make_serializable(result)
            client.setex(key, CACHE_TTL_SECONDS, json.dumps(clean_result))
            logger.info("Cache set", question=question[:60])
        except Exception as e:
            logger.warning("Cache set failed", error=str(e))

    def invalidate(self, pattern: str = "rag:*") -> int:
        client = self._get_client()
        if not client:
            return 0
        try:
            keys = client.keys(pattern)
            if keys:
                client.delete(*keys)
            logger.info("Cache invalidated", count=len(keys))
            return len(keys)
        except Exception as e:
            logger.warning("Cache invalidation failed", error=str(e))
            return 0


@lru_cache()
def get_cache_service() -> CacheService:
    return CacheService()