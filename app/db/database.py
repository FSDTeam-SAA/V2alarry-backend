from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

database_url = make_url(settings.DATABASE_URL)
async_database_url = database_url.set(
    drivername="postgresql+asyncpg",
).difference_update_query(["sslmode", "channel_binding"])
sync_database_url = database_url.set(drivername="postgresql+psycopg2")
async_connect_args = {}

if database_url.query.get("sslmode") in {"require", "verify-ca", "verify-full"}:
    async_connect_args["ssl"] = True

engine = create_async_engine(
    async_database_url,
    echo=True,
    future=True,
    connect_args=async_connect_args,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)

sync_engine = create_engine(
    sync_database_url,
    echo=True,
    future=True,
)

SessionLocal = sessionmaker(
    bind=sync_engine,
    autocommit=False,
    autoflush=False,
)


def ensure_schema_compatibility() -> None:
    """
    Repair legacy schemas that still use UUID foreign keys for integer-backed users.
    This keeps older environments working even if Alembic migrations were skipped.
    """
    with sync_engine.begin() as connection:
        current_type = connection.execute(
            text(
                """
                SELECT data_type
                FROM information_schema.columns
                WHERE table_schema = 'public'
                  AND table_name = 'conversations'
                  AND column_name = 'user_id'
                """
            )
        ).scalar_one_or_none()

        if current_type != "uuid":
            return

        constraint_name = connection.execute(
            text(
                """
                SELECT tc.constraint_name
                FROM information_schema.table_constraints AS tc
                JOIN information_schema.key_column_usage AS kcu
                  ON tc.constraint_name = kcu.constraint_name
                 AND tc.table_schema = kcu.table_schema
                WHERE tc.table_schema = 'public'
                  AND tc.table_name = 'conversations'
                  AND tc.constraint_type = 'FOREIGN KEY'
                  AND kcu.column_name = 'user_id'
                LIMIT 1
                """
            )
        ).scalar_one_or_none()

        # Legacy rows can't be mapped to integer user IDs, so clear chat history
        # before rebuilding the foreign key column with the current schema type.
        connection.execute(text("DELETE FROM messages"))
        connection.execute(text("DELETE FROM conversations"))

        if constraint_name:
            connection.execute(
                text(f'ALTER TABLE public.conversations DROP CONSTRAINT "{constraint_name}"')
            )

        connection.execute(text("ALTER TABLE public.conversations DROP COLUMN user_id"))
        connection.execute(
            text(
                """
                ALTER TABLE public.conversations
                ADD COLUMN user_id INTEGER NOT NULL REFERENCES public.users(id)
                """
            )
        )
