import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load the .env file using an absolute path so it works no matter where uvicorn is started from
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = BASE_DIR / ".env"

if ENV_FILE.exists():
    load_dotenv(dotenv_path=ENV_FILE, override=True)
else:
    # Fallback to backend/.env if starting inside backend
    BACKEND_ENV = Path(__file__).resolve().parent.parent / ".env"
    if BACKEND_ENV.exists():
        load_dotenv(dotenv_path=BACKEND_ENV, override=True)
    else:
        load_dotenv(override=True)

class Settings(BaseModel):
    APP_NAME: str = "Arise"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api"
    ENABLE_DOCS: bool = Field(default_factory=lambda: os.getenv("ENABLE_DOCS", "true").lower() in ("true", "1", "yes"))
    
    # Vector store config ("chroma" or "pgvector")
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "chroma")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/mrsptu_rag")
    
    # Storage
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./uploads")
    
    # RAG parameters
    CHUNK_SIZE_TOKENS: int = 500
    CHUNK_OVERLAP_TOKENS: int = 50
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.32"))
    RRF_K: int = int(os.getenv("RRF_K", "60"))
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    NEIGHBOR_EXPANSION: bool = os.getenv("NEIGHBOR_EXPANSION", "true").lower() in ("true", "1", "yes")
    
    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "auto") # auto, gemini, mock
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", "")

    # Auth & Security
    AUTH_ENABLED: bool = Field(default_factory=lambda: os.getenv("AUTH_ENABLED", "true").lower() in ("true", "1", "yes"))
    USERS_DB_PATH: str = os.getenv("USERS_DB_PATH", "data/users.db")
    SESSION_SECRET: Optional[str] = os.getenv("SESSION_SECRET", None)
    COOKIE_SECURE: bool = Field(default_factory=lambda: os.getenv("COOKIE_SECURE", "false").lower() in ("true", "1", "yes"))
    ALLOW_SIGNUP: bool = Field(default_factory=lambda: os.getenv("ALLOW_SIGNUP", "true").lower() in ("true", "1", "yes"))
    ADMIN_EMAIL: Optional[str] = os.getenv("ADMIN_EMAIL", None)
    ADMIN_PASSWORD: Optional[str] = os.getenv("ADMIN_PASSWORD", None)
    ADMIN_TOKEN: Optional[str] = os.getenv("ADMIN_TOKEN", "")
    DAILY_QUERY_LIMIT: int = int(os.getenv("DAILY_QUERY_LIMIT", "0"))
    ENV: str = os.getenv("ENV", os.getenv("ENVIRONMENT", "development")).lower()
    
    class Config:
        arbitrary_types_allowed = True

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
