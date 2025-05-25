from typing import Optional
import uuid
from litestar.datastructures import UploadFile
from pydantic import BaseModel, ConfigDict


class CreateMessageDTO(BaseModel):
    session_id: Optional[uuid.UUID]
    user_id: Optional[uuid.UUID]
    content: Optional[str]
    image_list: Optional[list[UploadFile]]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class AskAIDTO(BaseModel):
    content: str
    image_list: Optional[list[UploadFile]]
    save_message: Optional[bool]
    model_config = ConfigDict(arbitrary_types_allowed=True)
