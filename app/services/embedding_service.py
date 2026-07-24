from sentence_transformers import SentenceTransformer
from typing import List
import asyncio
from app.core.config import settings
from app.core.cache import cache_get, cache_set, make_cache_key

_model: SentenceTransformer = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


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
