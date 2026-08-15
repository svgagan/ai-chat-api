# app/routers/conversation.py
import uuid
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.services.ai_service import ai_service
from app.services.conversation_service import conversation_service

router = APIRouter()

class ConversationRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant."
    session_id: Optional[str] = None  # server generates if not provided

class ConversationResponse(BaseModel):
    reply: str
    model: str
    session_id: str  # always echo back so client knows which session

@router.post("/conversation", response_model=ConversationResponse)
def create_conversation(request: ConversationRequest):
    """
    POST /conversation
    Send a message, get AI response with full conversation memory.
    Session persists across multiple requests using session_id.
    """
    try:
        # Generate session_id server-side if client did not provide one
        session_id = request.session_id or str(uuid.uuid4())

        reply = conversation_service.chat(
            session_id=session_id,
            user_message=request.message,
            system_prompt=request.system_prompt
        )

        return ConversationResponse(
            reply=reply,
            model=ai_service.model,
            session_id=session_id  # client must save this for next request
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/conversation/{session_id}")
def delete_conversation(session_id: str):
    """
    DELETE /conversation/{session_id}
    Clear conversation history for a session.
    Use when: user logs out, starts fresh, or privacy requirements.
    """
    try:
        conversation_service.clear_session(session_id)
        return {"message": f"Session {session_id} cleared successfully"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))