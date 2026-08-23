# app/services/embedding_service.py
import litellm
import os
import hashlib
from sqlalchemy import func
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
        Now returns similarity_score for each result.
        """
        # Step 1 — embed query with SAME model as indexing
        query_embedding = self.embed_text(query)

        # cosine_distance computed as a column we can select
        distance_column = DocumentChunk.embedding.cosine_distance(query_embedding)

        # Step 2 — base query, exclude soft deleted
        query_builder = (
            db.query(DocumentChunk, distance_column.label("distance"))
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
            .order_by(distance_column)
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
                "created_at": str(chunk.created_at),
                "similarity_score": round(1 - distance, 4)
            }
            for chunk, distance in results
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

    def keyword_search(
        self,
        query: str,
        limit: int = 5,
        source_filter: str = None,
        db: Session = None
    ) -> list[dict]:

        """
        Full-text keyword search using PostgreSQL's native
        tsvector/ts_rank. Uses OR logic between query terms
        so natural language questions still match relevant
        documents even when not every word is present.
        """
        or_query_string = self._build_or_tsquery(query)
        ts_query = func.to_tsquery('english', or_query_string)
        rank_column = func.ts_rank(DocumentChunk.content_tsv, ts_query)

        query_builder = (
            db.query(DocumentChunk, rank_column.label("rank_score"))
            .filter(DocumentChunk.is_deleted == False)
            .filter(DocumentChunk.content_tsv.op('@@')(ts_query))
        )

        if source_filter:
            query_builder = query_builder.filter(
                DocumentChunk.source == source_filter
            )

        results = (
            query_builder
            .order_by(rank_column.desc())
            .limit(limit)
            .all()
        )

        return [
            {
                "id": str(chunk.id),
                "content": chunk.content,
                "source": chunk.source,
                "chunk_index": chunk.chunk_index,
                "metadata": chunk.doc_metadata,
                "created_at": str(chunk.created_at),
                "keyword_rank_score": round(float(rank_score), 4)
            }
            for chunk, rank_score in results
        ]

    def _build_or_tsquery(self, query: str) -> str:
        """
        Converts a natural language query into an OR-based tsquery
        string. PostgreSQL's plainto_tsquery defaults to AND logic,
        which fails on natural language questions where not every
        word appears in the matching document. OR logic means a
        document matches if it contains ANY meaningful query term,
        then ts_rank naturally scores documents with MORE matching
        terms higher than documents with just one.
        """
        words = query.split()
        or_query = " | ".join(words)
        return or_query

    def hybrid_search(
        self,
        query: str,
        limit: int = 5,
        source_filter: str = None,
        rrf_k: int = 60,
        pool_multiplier: int = 4,
        db: Session = None
    ) -> list[dict]:
        """
        Combines semantic search and keyword search using
        Reciprocal Rank Fusion (RRF).

        Retrieves a wider candidate pool from EACH method
        (not just `limit`), then fuses rankings, then trims
        to `limit`. This matters because a document might rank
        poorly in one method but well in the other — a narrow
        pool per method risks losing it before fusion even happens.
        """
        # Retrieve a wider pool from each method than the final limit
        candidate_pool_size = limit * pool_multiplier

        semantic_results = self.search(
            query=query,
            limit=candidate_pool_size,
            source_filter=source_filter,
            db=db
        )

        keyword_results = self.keyword_search(
            query=query,
            limit=candidate_pool_size,
            source_filter=source_filter,
            db=db
        )

        # Build rank lookups: chunk_id -> rank position (0-indexed)
        semantic_ranks = {
            chunk["id"]: rank for rank, chunk in enumerate(semantic_results)
        }
        keyword_ranks = {
            chunk["id"]: rank for rank, chunk in enumerate(keyword_results)
        }

        # Union of all chunk ids seen in either list
        all_chunk_ids = set(semantic_ranks.keys()) | set(keyword_ranks.keys())

        # Keep a lookup to the full chunk data regardless of which
        # list it came from.
        # Merge instead of overwrite — preserve fields from BOTH methods
        chunk_data_by_id = {}
        for c in semantic_results:
            chunk_data_by_id[c["id"]] = dict(c)

        for c in keyword_results:
            if c["id"] in chunk_data_by_id:
                # Chunk found in both — merge keyword fields IN,
                # don't lose the semantic fields already there
                chunk_data_by_id[c["id"]].update({
                    "keyword_rank_score": c["keyword_rank_score"]
                })
            else:
                # Chunk found only in keyword search
                chunk_data_by_id[c["id"]] = dict(c)

        # Compute RRF score for every chunk that appeared in EITHER list
        rrf_scores = []
        for chunk_id in all_chunk_ids:
            s_rank = semantic_ranks.get(chunk_id)
            k_rank = keyword_ranks.get(chunk_id)

            s_score = 1 / (rrf_k + s_rank + 1) if s_rank is not None else 0
            k_score = 1 / (rrf_k + k_rank + 1) if k_rank is not None else 0

            combined_score = s_score + k_score

            chunk = chunk_data_by_id[chunk_id]
            rrf_scores.append({
                **chunk,
                "rrf_score": round(combined_score, 5),
                "found_in_semantic": s_rank is not None,
                "found_in_keyword": k_rank is not None,
                "semantic_rank": s_rank + 1 if s_rank is not None else None,
                "keyword_rank": k_rank + 1 if k_rank is not None else None
            })

        # Sort by combined RRF score, highest first
        rrf_scores.sort(key=lambda x: x["rrf_score"], reverse=True)

        return rrf_scores[:limit]

embedding_service = EmbeddingService()