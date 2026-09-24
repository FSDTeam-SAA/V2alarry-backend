from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.password_reset_code import PasswordResetCode


class PasswordResetCodeRepository:
    async def count_requests_since(
        self,
        db: AsyncSession,
        user_id: int,
        since: datetime,
    ) -> int:
        result = await db.execute(
            select(func.count(PasswordResetCode.id)).where(
                PasswordResetCode.user_id == user_id,
                PasswordResetCode.requested_at >= since,
            )
        )
        return int(result.scalar_one())

    async def get_latest_for_user(
        self,
        db: AsyncSession,
        user_id: int,
    ) -> PasswordResetCode | None:
        result = await db.execute(
            select(PasswordResetCode)
            .where(PasswordResetCode.user_id == user_id)
            .order_by(PasswordResetCode.requested_at.desc(), PasswordResetCode.id.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def invalidate_active_for_user(
        self,
        db: AsyncSession,
        user_id: int,
        now: datetime,
    ) -> None:
        await db.execute(
            update(PasswordResetCode)
            .where(
                PasswordResetCode.user_id == user_id,
                PasswordResetCode.used_at.is_(None),
            )
            .values(used_at=now)
        )

    async def create(
        self,
        db: AsyncSession,
        code: PasswordResetCode,
    ) -> PasswordResetCode:
        db.add(code)
        await db.flush()
        return code

    async def increment_attempts(
        self,
        db: AsyncSession,
        code: PasswordResetCode,
    ) -> None:
        code.attempt_count += 1
        await db.flush()

    async def mark_verified(
        self,
        db: AsyncSession,
        code: PasswordResetCode,
        now: datetime,
    ) -> None:
        code.verified_at = now
        await db.flush()

    async def consume(
        self,
        db: AsyncSession,
        code: PasswordResetCode,
        now: datetime,
    ) -> None:
        code.used_at = now
        await db.flush()
