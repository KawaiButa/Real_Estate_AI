from typing import Optional
import uuid
from litestar.datastructures import UploadFile
from pydantic import BaseModel, ConfigDict


class CreateMessageDTO(BaseModel):
    session_id: Optional[uuid.UUID] = None
    user_id: Optional[uuid.UUID] = None
    content: Optional[str] = None
    image_list: Optional[list[UploadFile]] = []
    is_ai: bool = False
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AskAIDTO(BaseModel):
    content: str
    session_id: Optional[uuid.UUID]
    image_list: Optional[list[UploadFile]]
    save_message: Optional[bool]
    model_config = ConfigDict(arbitrary_types_allowed=True)
