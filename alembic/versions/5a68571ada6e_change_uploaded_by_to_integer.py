"""change uploaded_by to integer

Revision ID: 5a68571ada6e
Revises: 3ebc8355a760
Create Date: 2026-07-23 21:27:08.963951

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '5a68571ada6e'
down_revision: Union[str, Sequence[str], None] = '3ebc8355a760'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DELETE FROM document_chunks")
    op.execute("DELETE FROM documents")
    op.execute("ALTER TABLE documents DROP COLUMN uploaded_by")
    op.execute("ALTER TABLE documents ADD COLUMN uploaded_by INTEGER NOT NULL REFERENCES users(id)")


def downgrade() -> None:
    op.execute("ALTER TABLE documents DROP COLUMN uploaded_by")
    op.execute("ALTER TABLE documents ADD COLUMN uploaded_by UUID NOT NULL")
