"""add two-role coaching continuity and scoped documents

Revision ID: e8c4a1f6b920
Revises: c3f091d4ea62
Create Date: 2026-09-17 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "e8c4a1f6b920"
down_revision = "c3f091d4ea62"
branch_labels = None
depends_on = None


def _column_names(table_name: str) -> set[str]:
    return {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns(table_name)
    }


def _constraint_names(table_name: str) -> set[str]:
    inspector = sa.inspect(op.get_bind())
    names = {
        constraint["name"]
        for constraint in inspector.get_check_constraints(table_name)
        if constraint.get("name")
    }
    names.update(
        constraint["name"]
        for constraint in inspector.get_unique_constraints(table_name)
        if constraint.get("name")
    )
    return names


def _foreign_key(table_name: str, column_name: str) -> dict | None:
    inspector = sa.inspect(op.get_bind())
    for foreign_key in inspector.get_foreign_keys(table_name):
        if foreign_key.get("constrained_columns") == [column_name]:
            return foreign_key
    return None


def upgrade() -> None:
    bind = op.get_bind()
    unexpected_roles = bind.execute(
        sa.text(
            "SELECT DISTINCT role FROM users "
            "WHERE role IS NULL OR role NOT IN ('admin', 'user', 'candidate')"
        )
    ).scalars().all()
    if unexpected_roles:
        raise RuntimeError(
            "Unexpected user roles require manual review before migration: "
            + ", ".join(repr(role) for role in unexpected_roles)
        )

    orphan_messages = bind.execute(
        sa.text(
            "SELECT COUNT(*) FROM messages m "
            "LEFT JOIN conversations c ON c.id = m.conversation_id "
            "WHERE c.id IS NULL"
        )
    ).scalar_one()
    if orphan_messages:
        raise RuntimeError(
            f"Found {orphan_messages} orphan messages; no rows were changed. "
            "Repair their conversation mapping before migration."
        )

    op.execute("UPDATE users SET role = 'user' WHERE role = 'candidate'")
    op.alter_column("users", "role", existing_type=sa.String(length=50), nullable=False)
    if "ck_users_role" not in _constraint_names("users"):
        op.create_check_constraint("ck_users_role", "users", "role IN ('admin', 'user')")
    if "password_login_enabled" not in _column_names("users"):
        op.add_column(
            "users",
            sa.Column(
                "password_login_enabled",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            ),
        )
    op.execute(
        "UPDATE users SET password_login_enabled = false "
        "WHERE google_subject IS NOT NULL"
    )
    op.alter_column("users", "password_login_enabled", server_default=None)

    conversation_fk = _foreign_key("conversations", "user_id")
    if conversation_fk and conversation_fk.get("name"):
        op.drop_constraint(conversation_fk["name"], "conversations", type_="foreignkey")
    op.create_foreign_key(
        "fk_conversations_user_id_users",
        "conversations",
        "users",
        ["user_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_messages_conversation_id_conversations",
        "messages",
        "conversations",
        ["conversation_id"],
        ["id"],
        ondelete="CASCADE",
    )

    document_columns = _column_names("documents")
    if "category" not in document_columns:
        op.add_column(
            "documents",
            sa.Column(
                "category",
                sa.String(length=100),
                nullable=False,
                server_default="general",
            ),
        )
    if "file_size_bytes" not in document_columns:
        op.add_column(
            "documents",
            sa.Column(
                "file_size_bytes",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )
    if "is_global" not in document_columns:
        op.add_column(
            "documents",
            sa.Column(
                "is_global",
                sa.Boolean(),
                nullable=False,
                server_default=sa.true(),
            ),
        )
    else:
        op.execute("UPDATE documents SET is_global = true")
        op.alter_column(
            "documents",
            "is_global",
            existing_type=sa.Boolean(),
            nullable=False,
        )
    if "target_user_id" not in document_columns:
        op.add_column("documents", sa.Column("target_user_id", sa.Integer(), nullable=True))

    document_target_fk = _foreign_key("documents", "target_user_id")
    if document_target_fk and document_target_fk.get("name"):
        op.drop_constraint(
            document_target_fk["name"],
            "documents",
            type_="foreignkey",
        )
    op.create_foreign_key(
        "fk_documents_target_user_id_users",
        "documents",
        "users",
        ["target_user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.alter_column("documents", "category", server_default=None)
    op.alter_column("documents", "file_size_bytes", server_default=None)
    op.alter_column("documents", "is_global", server_default=None)

    op.create_table(
        "agreement_acceptances",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("agreement_version", sa.String(length=100), nullable=False),
        sa.Column("source", sa.String(length=50), nullable=False),
        sa.Column(
            "accepted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.UniqueConstraint(
            "user_id",
            "agreement_version",
            name="uq_user_agreement_version",
        ),
    )
    op.create_index(
        "ix_agreement_acceptances_user_id",
        "agreement_acceptances",
        ["user_id"],
    )

    op.create_table(
        "coaching_working_states",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.UUID(),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("state", sa.JSON(), nullable=False),
        sa.Column("source_turn_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_coaching_working_states_conversation_id",
        "coaching_working_states",
        ["conversation_id"],
        unique=True,
    )

    op.create_table(
        "coaching_summaries",
        sa.Column("id", sa.UUID(), primary_key=True),
        sa.Column(
            "conversation_id",
            sa.UUID(),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column(
            "user_id",
            sa.Integer(),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("presenting_focus", sa.Text(), nullable=True),
        sa.Column("primary_discovery", sa.Text(), nullable=True),
        sa.Column("developmental_theme", sa.Text(), nullable=True),
        sa.Column("commitment", sa.Text(), nullable=True),
        sa.Column("next_experiment", sa.Text(), nullable=True),
        sa.Column("follow_up_question", sa.Text(), nullable=True),
        sa.Column("coach_notes", sa.Text(), nullable=True),
        sa.Column("prior_session_continuity", sa.Text(), nullable=True),
        sa.Column("source_turn_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("generation_status", sa.String(length=20), nullable=False, server_default="current"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_coaching_summaries_conversation_id",
        "coaching_summaries",
        ["conversation_id"],
        unique=True,
    )
    op.create_index("ix_coaching_summaries_user_id", "coaching_summaries", ["user_id"])


def downgrade() -> None:
    op.drop_index("ix_coaching_summaries_user_id", table_name="coaching_summaries")
    op.drop_index("ix_coaching_summaries_conversation_id", table_name="coaching_summaries")
    op.drop_table("coaching_summaries")
    op.drop_index("ix_coaching_working_states_conversation_id", table_name="coaching_working_states")
    op.drop_table("coaching_working_states")
    op.drop_index("ix_agreement_acceptances_user_id", table_name="agreement_acceptances")
    op.drop_table("agreement_acceptances")

    op.drop_constraint("fk_documents_target_user_id_users", "documents", type_="foreignkey")
    op.drop_column("documents", "target_user_id")
    op.drop_column("documents", "is_global")
    op.drop_column("documents", "file_size_bytes")
    op.drop_column("documents", "category")

    op.drop_constraint("fk_messages_conversation_id_conversations", "messages", type_="foreignkey")
    op.drop_constraint("fk_conversations_user_id_users", "conversations", type_="foreignkey")
    op.create_foreign_key(
        "conversations_user_id_fkey",
        "conversations",
        "users",
        ["user_id"],
        ["id"],
    )
    op.drop_column("users", "password_login_enabled")
    op.drop_constraint("ck_users_role", "users", type_="check")
