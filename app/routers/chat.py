# app/routers/chat.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.ai_service import ai_service

router = APIRouter()

# Pydantic model defines exactly what the request body must look like
# FastAPI validates this automatically — wrong types return 422 errors
class ChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant."
    temperature: float = None
    max_tokens: int = None

class ChatResponse(BaseModel):
    reply: str
    model: str

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """
    POST /chat
    Send a message, get an AI response.
    """
    try:
        reply = ai_service.chat(
            user_message=request.message,
            system_prompt=request.system_prompt,
            temperature=request.temperature,
            max_tokens=request.max_tokens
        )

        return ChatResponse(
            reply=reply,
            model=ai_service.model  # tells the caller which model answered
        )

    except Exception as e:
        # We will make this much better in Phase 8
        # For now, surface the error clearly
        raise HTTPException(status_code=500, detail=str(e))