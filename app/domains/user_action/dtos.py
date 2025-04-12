from typing import Optional
from uuid import UUID
from pydantic import BaseModel, ConfigDict


class CreateUserActionDTO(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    action: str
    user_id: Optional[UUID]
    property_id: UUID