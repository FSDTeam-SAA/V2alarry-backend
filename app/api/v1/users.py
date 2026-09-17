from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies.auth import get_current_user
from app.core.agreements import CURRENT_AGREEMENT_VERSION
from app.db.session import get_db
from app.repositories.agreement_repository import AgreementRepository
from app.repositories.refresh_token_repository import RefreshTokenRepository
from app.repositories.user_repository import UserRepository
from app.schemas.user import (
    AgreementAcceptanceCreate,
    AgreementAcceptanceResponse,
    PasswordChangeRequest,
    UserResponse,
    UserUpdate,
)
from app.utils.security import has_password, verify_password

router = APIRouter()
user_repo = UserRepository()
agreement_repo = AgreementRepository()
refresh_token_repo = RefreshTokenRepository()


def _auth_provider(user) -> str:
    if user.google_subject and user.password_login_enabled:
        return "credentials+google"
    if user.google_subject:
        return "google"
    return "credentials"


async def _user_response(db: AsyncSession, user) -> dict:
    acceptance = await agreement_repo.get_acceptance(
        db,
        user_id=user.id,
        agreement_version=CURRENT_AGREEMENT_VERSION,
    )
    return {
        "id": user.id,
        "email": user.email,
        "full_name": user.full_name,
        "role": "user" if user.role == "candidate" else user.role,
        "is_active": user.is_active,
        "auth_provider": _auth_provider(user),
        "password_login_enabled": user.password_login_enabled,
        "accepted_agreement_version": (
            acceptance.agreement_version if acceptance else None
        ),
    }

@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await _user_response(db, current_user)


@router.patch("/me", response_model=UserResponse)
async def update_me(
    changes: UserUpdate,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    update_data = changes.model_dump(exclude_unset=True)
    if "email" in update_data:
        email = str(update_data["email"]).lower()
        existing = await user_repo.get_by_email(db, email)
        if existing and existing.id != current_user.id:
            raise HTTPException(status_code=409, detail="Email is already in use")
        current_user.email = email
    if "full_name" in update_data:
        current_user.full_name = update_data["full_name"]

    await db.commit()
    await db.refresh(current_user)
    return await _user_response(db, current_user)


@router.post("/me/password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    request: PasswordChangeRequest,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    if not current_user.password_login_enabled:
        raise HTTPException(
            status_code=409,
            detail="This account does not have a local password",
        )
    if not verify_password(request.current_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    if verify_password(request.new_password, current_user.hashed_password):
        raise HTTPException(
            status_code=400,
            detail="New password must be different from the current password",
        )

    current_user.hashed_password = has_password(request.new_password)
    await db.flush()
    await refresh_token_repo.revoke_all_for_user(db, current_user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/me/agreements",
    response_model=AgreementAcceptanceResponse,
    status_code=status.HTTP_201_CREATED,
)
async def accept_agreement(
    request: AgreementAcceptanceCreate,
    current_user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    return await agreement_repo.accept(
        db,
        user_id=current_user.id,
        agreement_version=request.agreement_version,
        source=request.source,
    )
