# app/routers/tools.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.tool_service import tool_service

router = APIRouter()

class ToolChatRequest(BaseModel):
    message: str

class ToolChatResponse(BaseModel):
    answer: str

# app/routers/tools.py
@router.post("/tool-chat", response_model=ToolChatResponse)
async def tool_chat(request: ToolChatRequest):
    try:
        answer = await tool_service.chat_with_tools(user_message=request.message)
        return ToolChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))