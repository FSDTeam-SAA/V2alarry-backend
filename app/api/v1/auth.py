import hashlib
import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.session import get_db
from app.models.refresh_token import RefreshToken
from app.schemas.user import (
    GoogleLoginRequest,
    RefreshTokenRequest,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserResponse,
)
from app.repositories.refresh_token_repository import RefreshTokenRepository
from app.repositories.user_repository import UserRepository
from app.services.auth_service import AuthService
from app.services.google_auth_service import GoogleAuthService, GoogleIdentityError

router = APIRouter()

user_repo = UserRepository()
auth_service = AuthService()
refresh_token_repo = RefreshTokenRepository()
google_auth_service = GoogleAuthService()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


async def _issue_tokens(db: AsyncSession, user) -> TokenResponse:
    refresh_token = secrets.token_urlsafe(48)
    await refresh_token_repo.create(
        db,
        RefreshToken(
            user_id=user.id,
            token_hash=_token_hash(refresh_token),
            expires_at=datetime.utcnow()
            + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        ),
    )
    return TokenResponse(
        access_token=auth_service.create_token(user),
        refresh_token=refresh_token,
        access_token_expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user={
            "id": user.id,
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
        },
    )


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):

    existing_user = await user_repo.get_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    user = auth_service.register_user(user_data)
    created_user = await user_repo.create(db, user)

    return created_user


@router.post("/login", response_model=TokenResponse)
async def login(user_data: UserLogin, db: AsyncSession = Depends(get_db)):

    user = await user_repo.get_by_email(db, user_data.email)

    auth_user = auth_service.authenticate_user(user, user_data.password)

    if not auth_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    return await _issue_tokens(db, auth_user)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_tokens(
    request: RefreshTokenRequest,
    db: AsyncSession = Depends(get_db),
):
    stored_token = await refresh_token_repo.get_active_by_hash(
        db,
        _token_hash(request.refresh_token),
    )
    if not stored_token or stored_token.expires_at <= datetime.utcnow():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    user = await user_repo.get_by_id(db, stored_token.user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    await refresh_token_repo.revoke(db, stored_token)
    return await _issue_tokens(db, user)


@router.post("/google", response_model=TokenResponse)
async def google_login(
    request: GoogleLoginRequest,
    db: AsyncSession = Depends(get_db),
):
    if not settings.GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Google sign-in is not configured",
        )

    try:
        identity = await run_in_threadpool(
            google_auth_service.verify_id_token,
            request.id_token,
            settings.GOOGLE_CLIENT_ID,
        )
    except GoogleIdentityError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google credentials",
        )

    user = await user_repo.get_by_google_subject(db, identity.subject)
    if not user:
        user = await user_repo.get_by_email(db, identity.email)
        if user and not user.google_subject:
            user.google_subject = identity.subject
            await db.commit()
            await db.refresh(user)
        elif user and user.google_subject != identity.subject:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Google credentials",
            )
        elif not user:
            user = await user_repo.create(
                db,
                auth_service.register_google_user(
                    email=identity.email,
                    full_name=identity.full_name,
                    google_subject=identity.subject,
                ),
            )

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google credentials",
        )

    return await _issue_tokens(db, user)
