# app/routers/extract.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.schemas.customer_inquiry import CustomerInquiry
from app.services.ai_service import ai_service

router = APIRouter()

class ExtractRequest(BaseModel):
    text: str

class ExtractResponse(BaseModel):
    extracted_data: CustomerInquiry
    model: str

@router.post("/extract", response_model=ExtractResponse)
def extract(request: ExtractRequest):
    try:
        system_prompt = """You are a precise data extraction assistant.
Extract only the information explicitly present in the customer's message.
Do not guess or fabricate values for fields that are not mentioned — leave them empty.

For sentiment, use these guidelines:
- happy: customer expresses satisfaction or gratitude
- neutral: customer is simply stating facts or asking questions, no clear emotion
- frustrated: customer expresses mild to moderate annoyance, impatience, or disappointment
- angry: customer expresses strong anger, uses aggressive language, or threatens action
"""

        result = ai_service.extract_structured(
            user_message=request.text,
            response_model=CustomerInquiry,
            system_prompt=system_prompt
        )
        return ExtractResponse(extracted_data=result, model=ai_service.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))