from uuid import UUID
from pydantic import BaseModel, Field
from datetime import datetime

class VerificationConfirmDTO(BaseModel):
    code: str = Field(..., min_length=6, max_length=128)
    method: str = Field(..., pattern="^(qr_code|otp|location|virtual_tour)$")
    lat: float
    lng: float 