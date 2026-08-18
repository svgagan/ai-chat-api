# app/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class AIConfig:
    """
    Single source of truth for AI configuration.
    To switch providers: change AI_MODEL and AI_API_KEY in .env only.
    Nothing else in the codebase needs to change.
    """
    MODEL: str = os.getenv("AI_MODEL", "groq/llama-3.1-8b-instant")
    API_KEY: str = os.getenv("AI_API_KEY", "")
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 1000
    MAX_CONVERSATION_MESSAGES: int = int(os.getenv("MAX_CONVERSATION_MESSAGES", "20"))
    # Embedding model — separate from chat model
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "ollama/nomic-embed-text")
    EMBEDDING_API_KEY: str = os.getenv("EMBEDDING_API_KEY", "")

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

ai_config = AIConfig()