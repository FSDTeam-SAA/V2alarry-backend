from datetime import datetime
from typing import Literal

from pydantic import BaseModel, EmailStr, Field, field_validator

from app.core.agreements import CURRENT_AGREEMENT_VERSION

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=1, max_length=255)
    agreement_version: str

    @field_validator("agreement_version")
    @classmethod
    def require_current_agreement(cls, value: str) -> str:
        if value != CURRENT_AGREEMENT_VERSION:
            raise ValueError("The current agreement version must be accepted")
        return value

    @field_validator("full_name")
    @classmethod
    def normalize_full_name(cls, value: str) -> str:
        return value.strip()


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(min_length=1, max_length=512)


class GoogleLoginRequest(BaseModel):
    id_token: str = Field(min_length=1, max_length=8192)

    @field_validator("id_token")
    @classmethod
    def reject_blank_id_token(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("id_token must not be blank")
        return value


class AuthenticatedUserResponse(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    role: str
    accepted_agreement_version: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    access_token_expires_in: int
    user: AuthenticatedUserResponse


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    role: Literal["admin", "user"]
    is_active: bool
    auth_provider: Literal["credentials", "google", "credentials+google"]
    password_login_enabled: bool
    accepted_agreement_version: str | None = None

    class Config: 
        from_attributes = True


class UserUpdate(BaseModel):
    email: EmailStr | None = None
    full_name: str | None = Field(default=None, min_length=1, max_length=255)

    @field_validator("full_name")
    @classmethod
    def normalize_optional_full_name(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        if not normalized:
            raise ValueError("full_name must not be blank")
        return normalized


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=8, max_length=128)


class AgreementAcceptanceCreate(BaseModel):
    agreement_version: str
    source: Literal["credentials", "google", "consent-gate", "settings"]

    @field_validator("agreement_version")
    @classmethod
    def require_current_version(cls, value: str) -> str:
        if value != CURRENT_AGREEMENT_VERSION:
            raise ValueError("Unsupported agreement version")
        return value


class AgreementAcceptanceResponse(BaseModel):
    agreement_version: str
    source: str
    accepted_at: datetime

    class Config:
        from_attributes = True
