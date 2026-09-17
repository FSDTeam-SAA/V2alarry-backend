"""change conversation user_id to integer

Revision ID: d4ebf69cbca6
Revises: 5a68571ada6e
Create Date: 2026-07-23 21:37:53.187536

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4ebf69cbca6'
down_revision: Union[str, Sequence[str], None] = '5a68571ada6e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM conversations LIMIT 1) THEN
                RAISE EXCEPTION
                    'Cannot infer integer users from legacy UUID conversation owners. '
                    'No rows were changed; perform an operator-reviewed mapping.';
            END IF;

            ALTER TABLE conversations DROP COLUMN user_id;
            ALTER TABLE conversations
                ADD COLUMN user_id INTEGER NOT NULL REFERENCES users(id);
        END $$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM conversations LIMIT 1) THEN
                RAISE EXCEPTION
                    'Downgrade cannot infer UUID owners from integer user IDs. '
                    'No rows were changed.';
            END IF;

            ALTER TABLE conversations DROP COLUMN user_id;
            ALTER TABLE conversations ADD COLUMN user_id UUID NOT NULL;
        END $$;
        """
    )
