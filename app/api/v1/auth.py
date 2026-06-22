from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.schemas.user import UserCreate, UserLogin, UserResponse
from app.repositories.user_repository import UserRepository
from app.services.auth_service import AuthService

router = APIRouter()

user_repo = UserRepository()
auth_service = AuthService()


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):

    existing_user = await user_repo.get_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="User already exists")

    user = auth_service.register_user(user_data)
    created_user = await user_repo.create(db, user)

    return created_user


@router.post("/login")
async def login(user_data: UserLogin, db: AsyncSession = Depends(get_db)):

    user = await user_repo.get_by_email(db, user_data.email)

    auth_user = auth_service.authenticate_user(user, user_data.password)

    if not auth_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    token = auth_service.create_token(auth_user)

    return {"access_token": token, "token_type": "bearer"}