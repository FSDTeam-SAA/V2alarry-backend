import hashlib
import hmac
import logging
import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Response, status
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.agreements import CURRENT_AGREEMENT_VERSION
from app.db.session import get_db
from app.models.refresh_token import RefreshToken
from app.models.password_reset_code import PasswordResetCode
from app.schemas.user import (
    GoogleLoginRequest,
    RefreshTokenRequest,
    PasswordResetRequest,
    ResetPasswordRequest,
    TokenResponse,
    UserCreate,
    UserLogin,
    UserResponse,
    VerifyResetOtpRequest,
)
from app.repositories.password_reset_code_repository import PasswordResetCodeRepository
from app.repositories.refresh_token_repository import RefreshTokenRepository
from app.repositories.agreement_repository import AgreementRepository
from app.repositories.user_repository import UserRepository
from app.services.auth_service import AuthService
from app.services.google_auth_service import GoogleAuthService, GoogleIdentityError
from app.services.password_reset_email_service import (
    PasswordResetEmailDeliveryError,
    PasswordResetEmailService,
)
from app.utils.security import has_password

router = APIRouter()
logger = logging.getLogger(__name__)

user_repo = UserRepository()
auth_service = AuthService()
refresh_token_repo = RefreshTokenRepository()
password_reset_code_repo = PasswordResetCodeRepository()
agreement_repo = AgreementRepository()
google_auth_service = GoogleAuthService()
password_reset_email_service = PasswordResetEmailService()


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _utcnow() -> datetime:
    return datetime.utcnow()


def _new_reset_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def _reset_code_hash(user_id: int, code: str) -> str:
    return hmac.new(
        settings.SECRET_KEY.encode("utf-8"),
        f"{user_id}:{code}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _password_recovery_is_available(user) -> bool:
    return bool(user and user.is_active and user.password_login_enabled)


def _invalid_reset_code() -> HTTPException:
    return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Code is invalid or expired")


async def _get_reset_record(
    db: AsyncSession,
    email: str,
) -> tuple[object, PasswordResetCode]:
    user = await user_repo.get_by_email(db, email)
    if not _password_recovery_is_available(user):
        raise _invalid_reset_code()

    reset_code = await password_reset_code_repo.get_latest_for_user(db, user.id)
    now = _utcnow()
    if (
        not reset_code
        or reset_code.used_at is not None
        or reset_code.expires_at <= now
        or reset_code.attempt_count >= settings.PASSWORD_RESET_MAX_ATTEMPTS
    ):
        raise _invalid_reset_code()

    return user, reset_code


async def _validate_reset_code(
    db: AsyncSession,
    user,
    reset_code: PasswordResetCode,
    code: str,
) -> datetime:
    now = _utcnow()
    if not hmac.compare_digest(reset_code.code_hash, _reset_code_hash(user.id, code)):
        await password_reset_code_repo.increment_attempts(db, reset_code)
        await db.commit()
        raise _invalid_reset_code()
    return now


async def _issue_tokens(db: AsyncSession, user) -> TokenResponse:
    acceptance = await agreement_repo.get_acceptance(
        db,
        user_id=user.id,
        agreement_version=CURRENT_AGREEMENT_VERSION,
    )
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
            "role": "user" if user.role == "candidate" else user.role,
            "accepted_agreement_version": (
                acceptance.agreement_version if acceptance else None
            ),
        },
    )


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):

    existing_user = await user_repo.get_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    user = auth_service.register_user(user_data)
    created_user = await user_repo.create(db, user, commit=False)
    acceptance = await agreement_repo.accept(
        db,
        user_id=created_user.id,
        agreement_version=CURRENT_AGREEMENT_VERSION,
        source="credentials",
        commit=False,
    )
    await db.commit()
    await db.refresh(created_user)
    await db.refresh(acceptance)

    return {
        "id": created_user.id,
        "email": created_user.email,
        "full_name": created_user.full_name,
        "role": created_user.role,
        "is_active": created_user.is_active,
        "auth_provider": "credentials",
        "password_login_enabled": created_user.password_login_enabled,
        "accepted_agreement_version": acceptance.agreement_version,
    }


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


@router.post("/forgot-password", status_code=status.HTTP_202_ACCEPTED)
async def forgot_password(
    request: PasswordResetRequest,
    db: AsyncSession = Depends(get_db),
):
    user = await user_repo.get_by_email(db, request.email)
    if not _password_recovery_is_available(user):
        return Response(status_code=status.HTTP_202_ACCEPTED)

    now = _utcnow()
    latest_code = await password_reset_code_repo.get_latest_for_user(db, user.id)
    request_count = await password_reset_code_repo.count_requests_since(
        db,
        user.id,
        now - timedelta(minutes=settings.PASSWORD_RESET_REQUEST_WINDOW_MINUTES),
    )
    is_cooling_down = bool(
        latest_code
        and latest_code.requested_at
        > now - timedelta(seconds=settings.PASSWORD_RESET_REQUEST_COOLDOWN_SECONDS)
    )
    if request_count >= settings.PASSWORD_RESET_REQUEST_LIMIT or is_cooling_down:
        return Response(status_code=status.HTTP_202_ACCEPTED)

    code = _new_reset_code()
    reset_code = PasswordResetCode(
        user_id=user.id,
        code_hash=_reset_code_hash(user.id, code),
        expires_at=now + timedelta(minutes=settings.PASSWORD_RESET_CODE_TTL_MINUTES),
        requested_at=now,
        attempt_count=0,
    )
    try:
        await password_reset_code_repo.invalidate_active_for_user(db, user.id, now)
        await password_reset_code_repo.create(db, reset_code)
        await password_reset_email_service.send_code(user.email, code)
        await db.commit()
    except PasswordResetEmailDeliveryError as error:
        await db.rollback()
        logger.warning("Password recovery email delivery failed: %s", error)
        return Response(status_code=status.HTTP_202_ACCEPTED)

    return Response(status_code=status.HTTP_202_ACCEPTED)


@router.post("/verify-reset-otp", status_code=status.HTTP_204_NO_CONTENT)
async def verify_reset_otp(
    request: VerifyResetOtpRequest,
    db: AsyncSession = Depends(get_db),
):
    user, reset_code = await _get_reset_record(db, request.email)
    now = await _validate_reset_code(db, user, reset_code, request.code)
    if reset_code.verified_at is None:
        await password_reset_code_repo.mark_verified(db, reset_code, now)
        await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/reset-password", status_code=status.HTTP_204_NO_CONTENT)
async def reset_password(
    request: ResetPasswordRequest,
    db: AsyncSession = Depends(get_db),
):
    user, reset_code = await _get_reset_record(db, request.email)
    now = await _validate_reset_code(db, user, reset_code, request.code)
    if reset_code.verified_at is None:
        raise _invalid_reset_code()

    user.hashed_password = has_password(request.new_password)
    await password_reset_code_repo.consume(db, reset_code, now)
    await refresh_token_repo.revoke_all_for_user(db, user.id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


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
