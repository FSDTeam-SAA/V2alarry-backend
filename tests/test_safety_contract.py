import inspect
import unittest
from pathlib import Path

from app.db import database


class DatabaseSafetyContractTests(unittest.TestCase):
    def test_sqlalchemy_parameter_echo_is_disabled(self):
        self.assertFalse(database.engine.echo)
        self.assertFalse(database.sync_engine.echo)

    def test_runtime_schema_check_never_mutates_data_or_schema(self):
        source = inspect.getsource(database.ensure_schema_compatibility).upper()

        for destructive_statement in ("DELETE ", "DROP ", "ALTER "):
            self.assertNotIn(destructive_statement, source)

    def test_migrations_never_delete_application_rows(self):
        versions = Path(__file__).parents[1] / "alembic" / "versions"

        for migration in versions.glob("*.py"):
            source = migration.read_text(encoding="utf-8").upper()
            self.assertNotIn("DELETE FROM ", source, migration.name)

    def test_pending_migrations_tolerate_verified_legacy_objects(self):
        versions = Path(__file__).parents[1] / "alembic" / "versions"
        auth_migration = (
            versions / "c3f091d4ea62_add_auth_refresh_and_google_subject.py"
        ).read_text(encoding="utf-8")
        continuity_migration = (
            versions / "e8c4a1f6b920_add_two_role_coaching_continuity.py"
        ).read_text(encoding="utf-8")

        self.assertIn('if "google_subject" not in _column_names("users")', auth_migration)
        self.assertIn('if not inspector.has_table("refresh_tokens")', auth_migration)
        self.assertIn('if "is_global" not in document_columns', continuity_migration)
        self.assertIn('if "target_user_id" not in document_columns', continuity_migration)


if __name__ == "__main__":
    unittest.main()
