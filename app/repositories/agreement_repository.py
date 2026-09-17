from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.agreement_acceptance import AgreementAcceptance


class AgreementRepository:
    async def get_acceptance(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        agreement_version: str,
    ) -> AgreementAcceptance | None:
        result = await db.execute(
            select(AgreementAcceptance).where(
                AgreementAcceptance.user_id == user_id,
                AgreementAcceptance.agreement_version == agreement_version,
            )
        )
        return result.scalar_one_or_none()

    async def accept(
        self,
        db: AsyncSession,
        *,
        user_id: int,
        agreement_version: str,
        source: str,
        commit: bool = True,
    ) -> AgreementAcceptance:
        existing = await self.get_acceptance(
            db,
            user_id=user_id,
            agreement_version=agreement_version,
        )
        if existing:
            return existing

        acceptance = AgreementAcceptance(
            user_id=user_id,
            agreement_version=agreement_version,
            source=source,
        )
        db.add(acceptance)
        if commit:
            await db.commit()
            await db.refresh(acceptance)
        return acceptance
