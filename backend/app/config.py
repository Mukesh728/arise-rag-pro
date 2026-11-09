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
    RRF_K: int = 60
    DEFAULT_TOP_K: int = 3
    
    # LLM Settings
    LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "auto") # auto, gemini, openai, groq, mock
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY", "")
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY", "")
    
    class Config:
        arbitrary_types_allowed = True

settings = Settings()

# Ensure directories exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
