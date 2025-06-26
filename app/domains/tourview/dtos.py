from litestar.datastructures import UploadFile
from pydantic import BaseModel, ConfigDict


class CreatePanoramaDTO(BaseModel):
    images: list[UploadFile]
    name: str
    model_config = ConfigDict(arbitrary_types_allowed=True)
