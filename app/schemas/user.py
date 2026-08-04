from pydantic import BaseModel, EmailStr, Field, field_validator

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str


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


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    access_token_expires_in: int
    user: AuthenticatedUserResponse


class UserResponse(BaseModel):
    id: int
    email: EmailStr
    role: str
    is_active: bool

    class Config: 
        from_attributes = True
