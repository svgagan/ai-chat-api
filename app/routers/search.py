# app/routers/search.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from app.services.embedding_service import embedding_service
from app.database import get_db

router = APIRouter()

# ─── Index Document ───────────────────────────────────────────

class DocumentRequest(BaseModel):
    content: str
    source: str
    chunk_index: int = 0
    metadata: Optional[dict] = None
    created_by: Optional[str] = None

class DocumentResponse(BaseModel):
    id: str
    content: str
    source: str
    chunk_index: int
    is_newly_created: bool
    message: str

@router.post("/documents", response_model=DocumentResponse)
def index_document(
    request: DocumentRequest,
    db: Session = Depends(get_db)   # FastAPI injects session automatically
):
    """
    POST /documents
    Embed and index a text chunk into the vector database.
    Skips duplicates automatically using content hash.
    """
    try:
        chunk, is_newly_created = embedding_service.index_document(
            content=request.content,
            source=request.source,
            chunk_index=request.chunk_index,
            metadata=request.metadata,
            created_by=request.created_by,
            db=db
        )

        return DocumentResponse(
            id=str(chunk.id),
            content=chunk.content,
            source=chunk.source,
            chunk_index=chunk.chunk_index,
            is_newly_created=is_newly_created,
            message="Indexed successfully" if is_newly_created else "Already exists"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Semantic Search ──────────────────────────────────────────

class SearchRequest(BaseModel):
    query: str
    limit: int = 5
    source_filter: Optional[str] = None

class SearchResult(BaseModel):
    id: str
    content: str
    source: str
    chunk_index: int
    metadata: Optional[dict] = None
    created_at: str

class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int

@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    db: Session = Depends(get_db)
):
    """
    POST /search
    Semantic search over indexed documents.
    Returns most similar chunks to the query by meaning, not keywords.
    """
    try:
        results = embedding_service.search(
            query=request.query,
            limit=request.limit,
            source_filter=request.source_filter,
            db=db
        )

        return SearchResponse(
            query=request.query,
            results=[SearchResult(**r) for r in results],
            total=len(results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Delete By Source ─────────────────────────────────────────

class DeleteResponse(BaseModel):
    message: str
    deleted_count: int

@router.delete("/documents/{source}", response_model=DeleteResponse)
def delete_by_source(
    source: str,
    db: Session = Depends(get_db)
):
    """
    DELETE /documents/{source}
    Soft delete all chunks from a source document.
    Data is preserved but excluded from all searches.
    """
    try:
        deleted_count = embedding_service.delete_by_source(
            source=source,
            db=db
        )

        return DeleteResponse(
            message=f"Source '{source}' soft deleted successfully",
            deleted_count=deleted_count
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))