# app/schemas/customer_inquiry.py
from enum import Enum
from typing import Optional
from pydantic import BaseModel

class Sentiment(str, Enum):
    HAPPY = "happy"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"

class CustomerInquiry(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    orderNumber: Optional[str] = None
    email: Optional[str] = None
    issue: Optional[str] = None
    userSentiment: Optional[Sentiment] = None