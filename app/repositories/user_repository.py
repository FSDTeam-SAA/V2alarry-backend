from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models.user import User

class UserRepository: 
    async def get_by_email(self, db: AsyncSession, email: str):
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()
    
    async def create(self, db: AsyncSession, email: str, user: User):
        db.add(user)
        await db.commit()
        await db.refresh(user)
        return user 
    
    async def get_by_id(
            self,
            db: AsyncSession,
            user_id: int,
    ):
        result = await db.execute(
            select(User).where(User.id == user_id)
        )

        return result.scalar_one_or_none()