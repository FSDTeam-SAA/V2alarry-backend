"""repair conversation user_id type

Revision ID: 8d9c7b5e4a21
Revises: d4ebf69cbca6
Create Date: 2026-07-23 22:05:00.000000

"""
from typing import Sequence, Union

from alembic import op


# revision identifiers, used by Alembic.
revision: str = "8d9c7b5e4a21"
down_revision: Union[str, Sequence[str], None] = "d4ebf69cbca6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        DO $$
        DECLARE
            current_type TEXT;
        BEGIN
            SELECT data_type
            INTO current_type
            FROM information_schema.columns
            WHERE table_name = 'conversations'
              AND column_name = 'user_id'
              AND table_schema = 'public';

            IF current_type = 'uuid' THEN
                RAISE EXCEPTION
                    'Legacy UUID conversation owners require an operator-reviewed mapping. '
                    'No rows were changed.';
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        BEGIN
            RAISE EXCEPTION
                'Downgrade cannot infer legacy UUID owners from integer user IDs. '
                'No rows were changed.';
        END $$;
        """
    )
