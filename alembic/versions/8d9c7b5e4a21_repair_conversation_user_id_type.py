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
            constraint_name TEXT;
        BEGIN
            SELECT data_type
            INTO current_type
            FROM information_schema.columns
            WHERE table_name = 'conversations'
              AND column_name = 'user_id'
              AND table_schema = 'public';

            IF current_type = 'uuid' THEN
                DELETE FROM messages;
                DELETE FROM conversations;

                SELECT tc.constraint_name
                INTO constraint_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                WHERE tc.table_name = 'conversations'
                  AND tc.table_schema = 'public'
                  AND tc.constraint_type = 'FOREIGN KEY'
                  AND kcu.column_name = 'user_id'
                LIMIT 1;

                IF constraint_name IS NOT NULL THEN
                    EXECUTE format(
                        'ALTER TABLE public.conversations DROP CONSTRAINT %I',
                        constraint_name
                    );
                END IF;

                ALTER TABLE public.conversations DROP COLUMN user_id;
                ALTER TABLE public.conversations
                    ADD COLUMN user_id INTEGER NOT NULL REFERENCES public.users(id);
            END IF;
        END $$;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DO $$
        DECLARE
            current_type TEXT;
            constraint_name TEXT;
        BEGIN
            SELECT data_type
            INTO current_type
            FROM information_schema.columns
            WHERE table_name = 'conversations'
              AND column_name = 'user_id'
              AND table_schema = 'public';

            IF current_type = 'integer' THEN
                DELETE FROM messages;
                DELETE FROM conversations;

                SELECT tc.constraint_name
                INTO constraint_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                WHERE tc.table_name = 'conversations'
                  AND tc.table_schema = 'public'
                  AND tc.constraint_type = 'FOREIGN KEY'
                  AND kcu.column_name = 'user_id'
                LIMIT 1;

                IF constraint_name IS NOT NULL THEN
                    EXECUTE format(
                        'ALTER TABLE public.conversations DROP CONSTRAINT %I',
                        constraint_name
                    );
                END IF;

                ALTER TABLE public.conversations DROP COLUMN user_id;
                ALTER TABLE public.conversations ADD COLUMN user_id UUID NOT NULL;
            END IF;
        END $$;
        """
    )
