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
    MODEL: str = os.getenv("AI_MODEL", "gemini/gemini-1.5-flash")
    API_KEY: str = os.getenv("AI_API_KEY", "")
    DEFAULT_TEMPERATURE: float = 0.7
    DEFAULT_MAX_TOKENS: int = 1000

ai_config = AIConfig()