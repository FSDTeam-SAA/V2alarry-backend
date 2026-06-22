from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.auth import router as auth_router
from app.api.v1.users import router as users_router


app = FastAPI(tittle=settings.PROJECT_NAME)

app.include_router(auth_router, prefix="/api/v1/auth", tags=["Auth"])
app.include_router( users_router,prefix="/api/v1/users", tags=["Users"])

@app.get("/")
async def root():
    return {"message": "Interview Platform API running"}