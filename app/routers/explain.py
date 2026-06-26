# app/routers/chat.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.ai_service import ai_service

router = APIRouter()

# Pydantic model defines exactly what the request body must look like
# FastAPI validates this automatically — wrong types return 422 errors
class ExplainRequest(BaseModel):
    topic: str
    audience: str

class ExplainResponse(BaseModel):
    explanation: str
    model: str

@router.post("/explain", response_model=ExplainResponse)
def explain(request: ExplainRequest):
    """
    POST /explain
    Send a topic and audience, get an AI response.
    """
    try:
        system_prompt = f"""You are an expert teacher who explains complex topics clearly.
                        Your audience is: {request.audience}
                        Adjust your language, tone, and complexity entirely to suit this audience.
                        Use simple words for younger audiences. Use technical depth for expert audiences.
                        Keep your explanation concise and engaging."""
        reply = ai_service.chat(
            user_message=request.topic,
            system_prompt=system_prompt
        )

        return ExplainResponse(
            explanation=reply,
            model=ai_service.model  # tells the caller which model answered
        )

    except Exception as e:
        # We will make this much better in Phase 8
        # For now, surface the error clearly
        raise HTTPException(status_code=500, detail=str(e))