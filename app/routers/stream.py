# app/routers/stream.py
from fastapi import APIRouter
from pydantic import BaseModel
from app.services.ai_service import ai_service
from fastapi.responses import StreamingResponse
from typing import Generator

router = APIRouter()

class StreamChatRequest(BaseModel):
    message: str
    system_prompt: str = "You are a helpful assistant."

def token_generator(user_message: str, system_prompt: str) -> Generator:
    """
    Wraps ai_service.stream_chat() and formats each token
    as a Server-Sent Event (SSE).
    This is a generator — it yields one formatted chunk at a time.
    """
    try:
        for token in ai_service.stream_chat(
            user_message=user_message,
            system_prompt=system_prompt
        ):
            # SSE format: every chunk must look like "data: {content}\n\n"
            yield f"data: {token}\n\n"

        # Signal to client that stream is complete
        yield "data: [DONE]\n\n"

    except Exception as e:
        # Yield error as SSE so client receives it cleanly
        yield f"data: [ERROR] {str(e)}\n\n"

@router.post("/stream-chat")   # no response_model — streaming is not JSON
async def stream_chat(request: StreamChatRequest):
    """
    POST /stream-chat
    Stream AI response token by token using Server-Sent Events.
    Client receives words as they are generated — not all at once.
    """
    return StreamingResponse(
        content=token_generator(
            user_message=request.message,
            system_prompt=request.system_prompt
        ),
        media_type="text/event-stream"
    )