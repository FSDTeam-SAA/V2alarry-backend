from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select

from app.models.user import User

class UserRepository: 
    async def get_by_email(self, db: AsyncSession, email: str):
        result = await db.execute(
            select(User).where(func.lower(User.email) == email.lower())
        )
        return result.scalar_one_or_none()

    async def get_by_google_subject(self, db: AsyncSession, google_subject: str):
        result = await db.execute(
            select(User).where(User.google_subject == google_subject)
        )
        return result.scalar_one_or_none()
    
    async def create(self, db: AsyncSession, user: User, *, commit: bool = True):
        db.add(user)
        if commit:
            await db.commit()
            await db.refresh(user)
        else:
            await db.flush()
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
