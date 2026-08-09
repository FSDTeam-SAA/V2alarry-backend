import unittest
from unittest.mock import patch

from app.core.config import settings
from app.services import embedding_service


class EmbeddingServiceTests(unittest.TestCase):
    def setUp(self):
        embedding_service._model = None

    def tearDown(self):
        embedding_service._model = None

    @patch("app.services.embedding_service.SentenceTransformer")
    def test_preload_uses_only_the_cached_embedding_model(self, model_class):
        model = model_class.return_value

        result = embedding_service.preload_embedding_model()

        self.assertIs(result, model)
        model_class.assert_called_once_with(
            settings.EMBEDDING_MODEL,
            local_files_only=True,
        )

    @patch(
        "app.services.embedding_service.SentenceTransformer",
        side_effect=OSError("model cache is empty"),
    )
    def test_preload_fails_clearly_when_the_model_is_not_cached(self, _model_class):
        with self.assertRaisesRegex(RuntimeError, "not available in the local cache"):
            embedding_service.preload_embedding_model()


if __name__ == "__main__":
    unittest.main()
