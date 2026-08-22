# app/main.py
from fastapi import FastAPI
from app.routers import chat,explain,extract,stream,conversation,search,rag

app = FastAPI(
    title="AI Chat API",
    description="Model-agnostic AI chat service",
    version="1.0.0"
)

# Register the chat router
# All routes in chat.py will be prefixed with /api/v1
app.include_router(chat.router, prefix="/api/v1", tags=["Foundations"])
app.include_router(explain.router, prefix="/api/v1", tags=["Foundations"])
app.include_router(extract.router, prefix="/api/v1", tags=["Structured"])
app.include_router(stream.router, prefix="/api/v1", tags=["Server-Sent Events"])
app.include_router(conversation.router, prefix="/api/v1", tags=["Memory"])
app.include_router(search.router, prefix="/api/v1", tags=["Embeddings"])
app.include_router(rag.router, prefix="/api/v1", tags=["RAG"])

@app.get("/health", tags=["System"])
def health():
    """Simple health check — useful for AWS load balancers later."""
    return {"status": "healthy"}