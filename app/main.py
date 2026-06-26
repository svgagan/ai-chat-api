# app/main.py
from fastapi import FastAPI
from app.routers import chat,explain

app = FastAPI(
    title="AI Chat API",
    description="Model-agnostic AI chat service",
    version="1.0.0"
)

# Register the chat router
# All routes in chat.py will be prefixed with /api/v1
app.include_router(chat.router, prefix="/api/v1")
app.include_router(explain.router, prefix="/api/v1")

@app.get("/health")
def health():
    """Simple health check — useful for AWS load balancers later."""
    return {"status": "healthy"}