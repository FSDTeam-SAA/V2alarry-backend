import unittest

from sqlalchemy import CheckConstraint, ForeignKeyConstraint

from app.db.base import Base
from app.models import *  # noqa: F401,F403 - registers all mapped tables


class PersistenceContractTests(unittest.TestCase):
    def test_user_role_and_password_capability_contract(self):
        users = Base.metadata.tables["users"]

        self.assertEqual(users.c.role.default.arg, "user")
        self.assertIn("password_login_enabled", users.c)
        role_checks = {
            constraint.name: str(constraint.sqltext)
            for constraint in users.constraints
            if isinstance(constraint, CheckConstraint)
        }
        self.assertIn("ck_users_role", role_checks)
        self.assertIn("admin", role_checks["ck_users_role"])
        self.assertIn("user", role_checks["ck_users_role"])

    def test_coaching_state_summary_and_acceptance_tables_are_registered(self):
        self.assertIn("coaching_working_states", Base.metadata.tables)
        self.assertIn("coaching_summaries", Base.metadata.tables)
        self.assertIn("agreement_acceptances", Base.metadata.tables)

        state = Base.metadata.tables["coaching_working_states"]
        self.assertTrue(state.c.conversation_id.unique)
        self.assertIn("state", state.c)

        summary = Base.metadata.tables["coaching_summaries"]
        self.assertTrue(summary.c.conversation_id.unique)
        for field in (
            "presenting_focus",
            "primary_discovery",
            "developmental_theme",
            "commitment",
            "next_experiment",
            "follow_up_question",
            "coach_notes",
            "prior_session_continuity",
            "source_turn_count",
            "generation_status",
        ):
            self.assertIn(field, summary.c)

    def test_conversation_owned_records_cascade_on_delete(self):
        for table_name in ("messages", "coaching_working_states", "coaching_summaries"):
            table = Base.metadata.tables[table_name]
            conversation_fks = [
                constraint
                for constraint in table.constraints
                if isinstance(constraint, ForeignKeyConstraint)
                and constraint.referred_table.name == "conversations"
            ]
            self.assertEqual(len(conversation_fks), 1)
            self.assertEqual(conversation_fks[0].ondelete, "CASCADE")

    def test_documents_include_resolved_scope_metadata(self):
        documents = Base.metadata.tables["documents"]
        for field in ("category", "file_size_bytes", "is_global", "target_user_id"):
            self.assertIn(field, documents.c)


if __name__ == "__main__":
    unittest.main()
