import unittest
from unittest.mock import AsyncMock, patch

from app.core.vector_store import VectorStore
from app.schemas.document import DocumentResponse


class DocumentScopeContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_vector_search_cache_is_partitioned_by_allowed_documents(self):
        store = VectorStore()
        store._sync_search = unittest.mock.Mock(return_value=[])
        written_keys: list[str] = []

        async def capture_cache(key, _value, ttl):
            written_keys.append(key)

        with (
            patch("app.core.vector_store.cache_get", new=AsyncMock(return_value=None)),
            patch("app.core.vector_store.cache_set", new=capture_cache),
        ):
            await store.search([0.1], allowed_document_ids=["global-doc", "user-a-doc"])
            await store.search([0.1], allowed_document_ids=["global-doc", "user-b-doc"])

        self.assertEqual(len(written_keys), 2)
        self.assertNotEqual(written_keys[0], written_keys[1])

    def test_document_response_exposes_resolved_scope(self):
        fields = DocumentResponse.model_fields
        for field in (
            "category",
            "file_size_bytes",
            "is_global",
            "target_user_id",
            "target_user_email",
        ):
            self.assertIn(field, fields)


if __name__ == "__main__":
    unittest.main()
