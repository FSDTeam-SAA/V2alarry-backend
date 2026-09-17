"""add refresh tokens and Google subject

Revision ID: c3f091d4ea62
Revises: 8d9c7b5e4a21
Create Date: 2026-08-04 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


revision = "c3f091d4ea62"
down_revision = "8d9c7b5e4a21"
branch_labels = None
depends_on = None


def _column_names(table_name: str) -> set[str]:
    return {
        column["name"]
        for column in sa.inspect(op.get_bind()).get_columns(table_name)
    }


def _index_names(table_name: str) -> set[str]:
    return {
        index["name"]
        for index in sa.inspect(op.get_bind()).get_indexes(table_name)
    }


def upgrade() -> None:
    if "google_subject" not in _column_names("users"):
        op.add_column(
            "users",
            sa.Column("google_subject", sa.String(length=255), nullable=True),
        )
    if "ix_users_google_subject" not in _index_names("users"):
        op.create_index(
            "ix_users_google_subject",
            "users",
            ["google_subject"],
            unique=True,
        )

    inspector = sa.inspect(op.get_bind())
    if not inspector.has_table("refresh_tokens"):
        op.create_table(
            "refresh_tokens",
            sa.Column("id", sa.Integer(), primary_key=True),
            sa.Column(
                "user_id",
                sa.Integer(),
                sa.ForeignKey("users.id", ondelete="CASCADE"),
                nullable=False,
            ),
            sa.Column("token_hash", sa.String(length=64), nullable=False),
            sa.Column("expires_at", sa.DateTime(), nullable=False),
            sa.Column("revoked_at", sa.DateTime(), nullable=True),
            sa.Column("created_at", sa.DateTime(), nullable=False),
        )
    else:
        required_columns = {
            "id",
            "user_id",
            "token_hash",
            "expires_at",
            "revoked_at",
            "created_at",
        }
        missing_columns = required_columns - _column_names("refresh_tokens")
        if missing_columns:
            raise RuntimeError(
                "Existing refresh_tokens table is incompatible; missing columns: "
                + ", ".join(sorted(missing_columns))
            )

    refresh_indexes = _index_names("refresh_tokens")
    if "ix_refresh_tokens_user_id" not in refresh_indexes:
        op.create_index("ix_refresh_tokens_user_id", "refresh_tokens", ["user_id"])
    if "ix_refresh_tokens_token_hash" not in refresh_indexes:
        op.create_index(
            "ix_refresh_tokens_token_hash",
            "refresh_tokens",
            ["token_hash"],
            unique=True,
        )


def downgrade() -> None:
    op.drop_index("ix_refresh_tokens_token_hash", table_name="refresh_tokens")
    op.drop_index("ix_refresh_tokens_user_id", table_name="refresh_tokens")
    op.drop_table("refresh_tokens")
    op.drop_index("ix_users_google_subject", table_name="users")
    op.drop_column("users", "google_subject")
