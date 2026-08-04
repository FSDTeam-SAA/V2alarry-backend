from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.refresh_token import RefreshToken


class RefreshTokenRepository:
    async def create(self, db: AsyncSession, token: RefreshToken) -> RefreshToken:
        db.add(token)
        await db.commit()
        await db.refresh(token)
        return token

    async def get_active_by_hash(
        self,
        db: AsyncSession,
        token_hash: str,
    ) -> RefreshToken | None:
        result = await db.execute(
            select(RefreshToken).where(
                RefreshToken.token_hash == token_hash,
                RefreshToken.revoked_at.is_(None),
            )
        )
        return result.scalar_one_or_none()

    async def revoke(self, db: AsyncSession, token: RefreshToken) -> None:
        token.revoked_at = datetime.utcnow()
        await db.commit()
