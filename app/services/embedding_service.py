import asyncio
import logging
from typing import List

from sentence_transformers import SentenceTransformer

from app.core.config import settings
from app.core.cache import cache_get, cache_set, make_cache_key

_model: SentenceTransformer = None
logger = logging.getLogger(__name__)


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        try:
            _model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                local_files_only=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Embedding model '{settings.EMBEDDING_MODEL}' is not available in the local cache. "
                "Preload it before starting the backend."
            ) from exc
    return _model


def preload_embedding_model() -> SentenceTransformer:
    """Load the required model before the application begins serving requests."""
    model = _get_model()
    logger.info("Embedding model '%s' loaded from the local cache", settings.EMBEDDING_MODEL)
    return model


class EmbeddingService:
    def __init__(self):
        self.dimension = settings.EMBEDDING_DIMENSION

    async def embed(self, text: str) -> List[float]:
        cache_key = f"emb:{make_cache_key(text)}"
        cached = await cache_get(cache_key)
        if cached is not None:
            return cached

        model = _get_model()
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, model.encode, text)
        result = embedding.tolist()

        await cache_set(cache_key, result, ttl=settings.CACHE_EMBEDDING_TTL)
        return result

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        model = _get_model()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, model.encode, texts)
        return [emb.tolist() for emb in embeddings]
