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
Do not guess or fabricate values for fields not mentioned. 
 For missing fields, use JSON null — not the string 'null', not 'N/A', 
 not 'unknown'. Only actual JSON null.

For sentiment classification, use these examples as your guide:

HAPPY examples:
- Thank you so much, my order arrived early and everything looks perfect!
- My order arrived and thanks for delivering on time.

NEUTRAL examples:
- Hi, I wanted to check the status of my order 1234.
- Hi, I placed the order and looking forward to seeing it delivered on time.

FRUSTRATED examples:
- I have been waiting for a week and my order still hasn't arrived.
- Not sure if my order will be delivered or not despite providing all details.
- I've contacted support three times and nobody has helped me. This is ridiculous.

ANGRY examples:
- This is completely unacceptable. I want a refund immediately or I'm disputing the charge.
- I want to speak to a manager right now. This is the worst service I have ever experienced.

Use these boundaries strictly:
- frustrated = mild to moderate annoyance, impatience, disappointment
- angry = strong anger, aggressive language, threats, demands for escalation
"""

        result = ai_service.extract_structured(
            user_message=request.text,
            response_model=CustomerInquiry,
            system_prompt=system_prompt
        )
        return ExtractResponse(extracted_data=result, model=ai_service.model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))