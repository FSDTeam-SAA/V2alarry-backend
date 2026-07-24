import json
import hashlib
from typing import Any, Optional
from app.core.config import settings

try:
    import redis.asyncio as aioredis
    _redis_client: Optional[aioredis.Redis] = None

    async def get_redis() -> aioredis.Redis:
        global _redis_client
        if _redis_client is None:
            _redis_client = aioredis.from_url(
                settings.REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2,
            )
        return _redis_client

    async def cache_get(key: str) -> Optional[Any]:
        try:
            r = await get_redis()
            value = await r.get(key)
            if value is not None:
                return json.loads(value)
        except Exception:
            pass
        return None

    async def cache_set(key: str, value: Any, ttl: int = 300) -> None:
        try:
            r = await get_redis()
            await r.set(key, json.dumps(value, default=str), ex=ttl)
        except Exception:
            pass

    async def cache_delete_pattern(pattern: str) -> None:
        try:
            r = await get_redis()
            keys = []
            async for key in r.scan_iter(match=pattern):
                keys.append(key)
            if keys:
                await r.delete(*keys)
        except Exception:
            pass

    def make_cache_key(*parts: str) -> str:
        raw = ":".join(str(p) for p in parts)
        return hashlib.md5(raw.encode()).hexdigest()

except ImportError:
    async def cache_get(key: str) -> Optional[Any]:
        return None

    async def cache_set(key: str, value: Any, ttl: int = 300) -> None:
        pass

    async def cache_delete_pattern(pattern: str) -> None:
        pass

    def make_cache_key(*parts: str) -> str:
        return ""
