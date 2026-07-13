from app.db.database import AsyncSessionLocal, SessionLocal

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
