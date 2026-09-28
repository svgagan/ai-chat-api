# app/routers/agent.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.agent_service import agent_service
from app.services.langchain_agent_service import langchain_agent_service

router = APIRouter()

class AgentRequest(BaseModel):
    goal: str

class AgentResponse(BaseModel):
    answer: str
    steps_taken: int
    trace: list

@router.post("/agent", response_model=AgentResponse)
async def run_agent(request: AgentRequest):
    try:
        result = await agent_service.run_agent(goal=request.goal)
        return AgentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class LangChainAgentResponse(BaseModel):
    answer: str
    message_count: int
    raw_messages: list

@router.post("/agent-langchain", response_model=LangChainAgentResponse)
async def run_langchain_agent(request: AgentRequest):
    try:
        result = await langchain_agent_service.run_agent(goal=request.goal)
        return LangChainAgentResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

