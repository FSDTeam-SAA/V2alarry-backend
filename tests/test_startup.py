import unittest
from unittest.mock import patch

from app import main


class StartupTests(unittest.IsolatedAsyncioTestCase):
    @patch("app.main.ensure_schema_compatibility")
    @patch(
        "app.main.preload_embedding_model",
        side_effect=RuntimeError("Embedding model is not available in the local cache"),
    )
    async def test_startup_stops_when_the_embedding_model_is_not_cached(
        self,
        preload_embedding_model,
        ensure_schema_compatibility,
    ):
        with self.assertRaisesRegex(RuntimeError, "not available in the local cache"):
            await main.startup_event()

        preload_embedding_model.assert_called_once_with()
        ensure_schema_compatibility.assert_not_called()


if __name__ == "__main__":
    unittest.main()
