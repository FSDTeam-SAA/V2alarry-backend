from sentence_transformers import SentenceTransformer
from typing import List
import asyncio
from app.core.config import settings

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dimension = settings.EMBEDDING_DIMENSION
    
    async def embed(self, text: str) -> List[float]:
        """Generate embedding for single text"""
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, self.model.encode, text
        )
        return embedding.tolist()
    
    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(
            None, self.model.encode, texts
        )
        return [emb.tolist() for emb in embeddings]