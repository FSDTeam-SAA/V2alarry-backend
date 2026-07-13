from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import create_engine
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
