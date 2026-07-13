# test_vector_store.py
import asyncio
from app.core.vector_store import VectorStore
from app.services.embedding_service import EmbeddingService

async def test():
    # Initialize
    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    
    # Test embedding
    text = "This is a test document about AI"
    embedding = await embedding_service.embed(text)
    print(f"✅ Embedding generated: {len(embedding)} dimensions")
    
    # Test adding vectors
    vectors = [embedding]
    metadata = [{"content": text, "test": True}]
    ids = await vector_store.add_vectors(vectors, metadata)
    print(f"✅ Vectors added: {ids}")
    
    # Test search
    results = await vector_store.search(embedding, limit=1)
    print(f"✅ Search results: {results}")
    
    # Test count
    count = await vector_store.count()
    print(f"✅ Total vectors: {count}")

if __name__ == "__main__":
    asyncio.run(test())