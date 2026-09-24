import os
from typing import List

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "Interview Platform"
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5500",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5500",
    ]
    DATABASE_URL: str
    SQL_ECHO: bool = False

    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    GOOGLE_CLIENT_ID: str = "16217334115-0o69vmkih52rqbmkh3cnbpslmk68h0dp.apps.googleusercontent.com"
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 465
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_EMAIL: str = ""
    SMTP_TIMEOUT_SECONDS: int = 10
    PASSWORD_RESET_CODE_TTL_MINUTES: int = 10
    PASSWORD_RESET_REQUEST_COOLDOWN_SECONDS: int = 60
    PASSWORD_RESET_REQUEST_LIMIT: int = 3
    PASSWORD_RESET_REQUEST_WINDOW_MINUTES: int = 15
    PASSWORD_RESET_MAX_ATTEMPTS: int = 5

    # Vector DB
    VECTOR_DB_URL: str = "http://localhost:6333"
    VECTOR_DB_API_KEY: str = ""
    VECTOR_DB_COLLECTION: str = "knowledge_base"

    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # RAG Settings
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    TOP_K_RETRIEVAL: int = 10
    SIMILARITY_THRESHOLD: float = 0.45

    # LLM Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = "gpt-3.5-turbo"
    LLM_TEMPERATURE: float = 0.7
    MAX_TOKENS: int = 2048

    # Chat Tuning
    HISTORY_LIMIT: int = 10
    CONTEXT_DOC_COUNT: int = 5
    TITLE_TRUNCATE_LENGTH: int = 50

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_EMBEDDING_TTL: int = 600
    CACHE_HISTORY_TTL: int = 300
    CACHE_SEARCH_TTL: int = 600

    class Config:
        env_file = ".env"

settings = Settings()
