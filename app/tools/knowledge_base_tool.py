# app/tools/knowledge_base_tool.py
from app.services.rag_service import rag_service
from app.database import SessionLocal

def search_knowledge_base(question: str) -> dict:
    """
    Reuses the entire hybrid search + RAG pipeline from Projects
    5-7.5 as a TOOL the model can choose to call. This is worth
    noticing: a fully separate system built earlier becomes one
    option among several here, not a separate code path.
    """
    db = SessionLocal()
    try:
        result = rag_service.answer_question(query=question, db=db)
        return {"answer": result["answer"], "sources": result["sources_used"]}
    finally:
        db.close()

KNOWLEDGE_BASE_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "search_knowledge_base",
        "description": "Search the company's internal documents for policy questions, such as return policy, shipping policy, or refund questions. Use this for any question about company policies.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The user's question about company policy"}
            },
            "required": ["question"]
        }
    }
}