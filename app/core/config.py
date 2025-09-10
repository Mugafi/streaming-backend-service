"""Application configuration settings"""
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Application settings
    APP_NAME: str = "Asyn Backend Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database settings
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str
    DB_DRIVER: str = "postgresql"
    
    # MongoDB settings
    MONGO_HOST: str
    MONGO_DB: str
    CHARACTER_MONGO_DB: str = "character"
    DOCUMENT_COLLECTION: str = "documents"
    DOCUMENT_ASK_VED: str = "document_ask_ved"
    DOCUMENT_SEGMENTS: str = "segments"
    
    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    
    # Security
    SECRET_KEY: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 10080
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # External services
    VERIFY_SSL: bool = False  # Set to True in production
    OPENAI_API_KEY: Optional[str] = None
    CORE_BACKEND_BASE_URL: str = "http://localhost:8002"
    
    # Prompt Executor Service
    PROMPT_EXECUTOR_BASE_URL: str
    PROMPT_EXECUTOR_API_KEY: str
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Allow extra fields from .env

settings = Settings()