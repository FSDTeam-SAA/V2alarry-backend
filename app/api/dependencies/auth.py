from jose import JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.utils.jwt import decode_token
from app.repositories.user_repository import UserRepository
from app.core.agreements import CURRENT_AGREEMENT_VERSION
from app.repositories.agreement_repository import AgreementRepository
from app.models.user import User

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login"
)

user_repo = UserRepository()
agreement_repo = AgreementRepository()

async def get_current_user(
        token: str = Depends(oauth2_scheme),
        db: AsyncSession = Depends(get_db),
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
    )
    try:
        payload = decode_token(token)
        user_id = int(payload.get("sub"))

        if not user_id or payload.get("token_type") != "access":
            raise credentials_exception
        
    except (JWTError, TypeError, ValueError):
        raise credentials_exception
    
    user = await user_repo.get_by_id(db, user_id)

    if not user or not user.is_active:
        raise credentials_exception
    
    return user


async def require_current_agreement(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> User:
    acceptance = await agreement_repo.get_acceptance(
        db,
        user_id=current_user.id,
        agreement_version=CURRENT_AGREEMENT_VERSION,
    )
    if not acceptance:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Current agreement acceptance required",
        )
    return current_user


async def require_admin(
    current_user: User = Depends(require_current_agreement),
) -> User:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required",
        )
    return current_user
