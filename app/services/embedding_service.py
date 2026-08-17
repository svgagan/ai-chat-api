# app/services/embedding_service.py
import litellm
import os
import hashlib
from sqlalchemy.orm import Session
from app.config import ai_config
from app.models.document_chunks import DocumentChunk

def _set_embedding_api_key():
    """
    Same model-agnostic pattern as chat model.
    Detect provider from model string, set correct env variable.
    Add new provider here when needed — nowhere else changes.
    """
    model = ai_config.EMBEDDING_MODEL
    key = ai_config.EMBEDDING_API_KEY

    if not key:
        return  # local model like Ollama — no key needed

    if model.startswith("gemini/") or model.startswith("google/"):
        os.environ["GEMINI_API_KEY"] = key
        os.environ["GOOGLE_API_KEY"] = key
    elif model.startswith("openai/"):
        os.environ["OPENAI_API_KEY"] = key
    elif model.startswith("cohere/"):
        os.environ["COHERE_API_KEY"] = key
    elif model.startswith("anthropic/"):
        os.environ["ANTHROPIC_API_KEY"] = key
    elif model.startswith("ollama/"):
        pass  # local, no key needed
    elif model.startswith("huggingface/"):
        os.environ["HUGGINGFACE_API_KEY"] = key

_set_embedding_api_key()

class EmbeddingService:

    def __init__(self):
        self.embedding_model = ai_config.EMBEDDING_MODEL
        self.dimensions = 768

    def embed_text(self, text: str) -> list[float]:
        """
        Convert text into a vector of 768 floats.
        Core operation — everything else builds on this.
        """
        response = litellm.embedding(
            model=self.embedding_model,
            input=text
        )
        return response.data[0]["embedding"]

    def _compute_hash(self, text: str) -> str:
        """SHA256 hash of content for deduplication."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    def _check_duplicate(
        self, content_hash: str, source: str, db: Session
    ) -> DocumentChunk | None:
        """
        Check if this content already exists for this source.
        content_hash is now a real column — direct comparison, very fast.
        """
        return db.query(DocumentChunk).filter(
            DocumentChunk.content_hash == content_hash,
            DocumentChunk.source == source,
            DocumentChunk.is_deleted == False
        ).first()

    def index_document(
        self,
        content: str,
        source: str,
        chunk_index: int = 0,
        metadata: dict = None,
        created_by: str = None,
        db: Session = None
    ) -> tuple[DocumentChunk, bool]:
        """
        Embed a text chunk and store it in the database.
        Returns (chunk, was_created) tuple.
        was_created=False means duplicate was found and skipped.
        """
        # Step 1 — compute hash
        content_hash = self._compute_hash(content)

        # Step 2 — check duplicate using real column now
        existing = self._check_duplicate(content_hash, source, db)
        if existing:
            return existing, False   # duplicate found

        # Step 3 — embed
        embedding = self.embed_text(content)

        # Step 4 — create record
        chunk = DocumentChunk(
            content=content,
            embedding=embedding,
            source=source,
            chunk_index=chunk_index,
            doc_metadata=metadata or {},
            content_hash=content_hash,          # real column now
            embedding_model=self.embedding_model,
            created_by=created_by
        )

        db.add(chunk)
        db.commit()
        db.refresh(chunk)

        return chunk, True   # newly created

    def search(
        self,
        query: str,
        limit: int = 5,
        source_filter: str = None,
        db: Session = None
    ) -> list[dict]:
        """
        Find semantically similar chunks to the query.
        """
        # Step 1 — embed query with SAME model as indexing
        query_embedding = self.embed_text(query)

        # Step 2 — base query, exclude soft deleted
        query_builder = (
            db.query(DocumentChunk)
            .filter(DocumentChunk.is_deleted == False)
        )

        # Step 3 — optional source filter
        if source_filter:
            query_builder = query_builder.filter(
                DocumentChunk.source == source_filter
            )

        # Step 4 — cosine distance search
        results = (
            query_builder
            .order_by(
                DocumentChunk.embedding.cosine_distance(query_embedding)
            )
            .limit(limit)
            .all()
        )

        # Step 5 — format for caller
        return [
            {
                "id": str(chunk.id),
                "content": chunk.content,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.doc_metadata,
                "created_at": str(chunk.created_at)
            }
            for chunk in results
        ]

    def delete_by_source(self, source: str, db: Session) -> int:
        """
        Soft delete all chunks from a source.
        Returns count of deleted chunks.
        """
        chunks = db.query(DocumentChunk).filter(
            DocumentChunk.source == source,
            DocumentChunk.is_deleted == False
        ).all()

        for chunk in chunks:
            chunk.is_deleted = True

        db.commit()
        return len(chunks)


embedding_service = EmbeddingService()