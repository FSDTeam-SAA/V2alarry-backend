from jose import JWTError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.utils.jwt import decode_token
from app.repositories.user_repository import UserRepository

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login"
)

user_repo = UserRepository()

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

        if not user_id: 
            raise credentials_exception
        
    except JWTError:
        raise credentials_exception
    
    user = await user_repo.get_by_email(db, user_id)

    if not user:
        raise credentials_exception
    
    return user 
