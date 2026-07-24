from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List, Dict, Any, Optional
import uuid
import asyncio
from app.core.config import settings
from app.core.cache import cache_get, cache_set, make_cache_key


class VectorStore:
    def __init__(self):
        self.collection_name = settings.VECTOR_DB_COLLECTION
        self.vector_size = settings.EMBEDDING_DIMENSION
        self._client = None

    @property
    def client(self):
        if self._client is None:
            kwargs = {
                "url": settings.VECTOR_DB_URL,
                "prefer_grpc": False,
            }
            if settings.VECTOR_DB_API_KEY:
                kwargs["api_key"] = settings.VECTOR_DB_API_KEY
            self._client = QdrantClient(**kwargs)
            self._ensure_collection()
        return self._client

    def _ensure_collection(self):
        collections = self.client.get_collections()
        collection_names = [c.name for c in collections.collections]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def _sync_search(
        self,
        query_vector: List[float],
        limit: int,
        qdrant_filter: Optional[models.Filter] = None,
    ) -> List[Dict]:
        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
            query_filter=qdrant_filter,
            with_payload=True,
        )
        return [
            {
                "id": hit.id,
                "score": hit.score,
                "payload": hit.payload,
            }
            for hit in results.points
        ]

    async def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        filter_conditions: Optional[Dict] = None,
    ) -> List[Dict]:
        cache_key = f"vec:{make_cache_key(str(query_vector))}:{limit}"
        cached = await cache_get(cache_key)
        if cached is not None:
            return cached

        qdrant_filter = None
        if filter_conditions:
            qdrant_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                    for key, value in filter_conditions.items()
                ]
            )

        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(
            None, self._sync_search, query_vector, limit, qdrant_filter
        )

        await cache_set(cache_key, results, ttl=settings.CACHE_SEARCH_TTL)
        return results

    async def add_vectors(
        self,
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
    ) -> List[str]:
        points = []
        vector_ids = []

        for vector, meta in zip(vectors, metadata):
            point_id = str(uuid.uuid4())
            vector_ids.append(point_id)
            points.append(
                PointStruct(id=point_id, vector=vector, payload=meta)
            )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.upsert(
                collection_name=self.collection_name, points=points
            ),
        )

        return vector_ids

    async def delete_vectors(self, vector_ids: List[str]):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=vector_ids),
            ),
        )

    async def delete_by_filter(self, filter_conditions: Dict):
        qdrant_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
                for key, value in filter_conditions.items()
            ]
        )

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=qdrant_filter),
            ),
        )

    async def count(self) -> int:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self.client.count(collection_name=self.collection_name),
        )
        return result.count
