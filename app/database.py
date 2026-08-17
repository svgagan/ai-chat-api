# app/database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import ai_config

engine = create_engine(
    ai_config.DATABASE_URL,
    pool_pre_ping=True
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

class Base(DeclarativeBase):
    pass

def get_db():
    """
    FastAPI dependency.
    Provides one database session per request.
    Automatically closes when request ends.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()