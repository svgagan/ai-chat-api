# app/routers/rag.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Optional
from sqlalchemy.orm import Session
from app.services.rag_service import rag_service
from app.database import get_db
import uuid

router = APIRouter()

class RagRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    isChunkCitationRequired: bool = False

class ChunkCitation(BaseModel):
    source: str
    chunk_index: int
    content: str
    similarity_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    original_rank: Optional[int] = None
    final_rank: Optional[int] = None

class RagResponse(BaseModel):
    answer: str
    sources_used: list[str]
    chunks_retrieved: int
    chunks_used: int
    chunk_citations: Optional[list[ChunkCitation]] = None
    rrf_threshold: Optional[float] = None

@router.post("/ask", response_model=RagResponse)
def ask(
    request: RagRequest,
    db: Session = Depends(get_db)
):
    """
    POST /ask
    RAG pipeline — retrieves relevant context via semantic search,
    then generates an answer grounded in that context.
    Returns "I don't have information about that" if nothing
    relevant is found above the similarity threshold.
    """
    try:
        results = rag_service.answer_question(
            query=request.query,
            source_filter=request.source_filter,
            db=db
        )

        return RagResponse(
            answer=results["answer"],
            sources_used=results["sources_used"],
            chunks_retrieved=results["chunks_retrieved"],
            chunks_used=results["chunks_used"],
            chunk_citations=results["chunk_citations"] if request.isChunkCitationRequired else None,
            rrf_threshold=rag_service.rrf_threshold if request.isChunkCitationRequired else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RagConversationalRequest(BaseModel):
    query: str
    source_filter: Optional[str] = None
    isChunkCitationRequired: bool = False
    session_id: Optional[str] = None

class RagConversationalResponse(BaseModel):
    answer: str
    sources_used: list[str]
    chunks_retrieved: int
    chunks_used: int
    chunk_citations: Optional[list[ChunkCitation]] = None
    rrf_threshold: Optional[float] = None
    rewritten_query: str
    session_id: str



@router.post("/ask-conversational", response_model=RagConversationalResponse)
def askConversational(
    request: RagConversationalRequest,
    db: Session = Depends(get_db)
):
    try:
        session_id = request.session_id or str(uuid.uuid4())

        results = rag_service.answer_question_conversational(
            session_id=session_id,
            query=request.query,
            source_filter=request.source_filter,
            db=db
        )

        return RagConversationalResponse(
            answer=results["answer"],
            sources_used=results["sources_used"],
            rewritten_query=results["rewritten_query"],
            chunks_retrieved=results["chunks_retrieved"],
            chunks_used=results["chunks_used"],
            chunk_citations=results["chunk_citations"] if request.isChunkCitationRequired else None,
            rrf_threshold=rag_service.rrf_threshold if request.isChunkCitationRequired else None,
            session_id=session_id
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
