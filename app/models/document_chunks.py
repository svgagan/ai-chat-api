# app/models/document_chunks.py
from sqlalchemy import Column, Text, Integer, DateTime, Boolean, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from pgvector.sqlalchemy import Vector
import uuid
from app.database import Base

class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id              = Column(UUID(as_uuid=True), primary_key=True,
                             default=uuid.uuid4)
    content         = Column(Text, nullable=False)
    embedding       = Column(Vector(768), nullable=False)
    source          = Column(Text, nullable=False)
    chunk_index     = Column(Integer, nullable=False, default=0)
    doc_metadata    = Column("metadata", JSONB, default=dict)
    created_at      = Column(DateTime(timezone=True), server_default=func.now())
    created_by      = Column(Text)
    document_id     = Column(UUID(as_uuid=True))
    is_deleted      = Column(Boolean, default=False)
    content_hash    = Column(Text)
    embedding_model = Column(Text, default="gemini/text-embedding-004")