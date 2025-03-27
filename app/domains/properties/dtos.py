from typing import Optional
import uuid
from litestar.plugins.sqlalchemy import SQLAlchemyDTO, SQLAlchemyDTOConfig
from pydantic import BaseModel, ConfigDict, create_model
from litestar.plugins.pydantic import PydanticDTO
from database.models.image import Image, ImageSchema
from database.models.property import Property, PropertySchema
from litestar.dto import DTOConfig
from litestar.datastructures import UploadFile


class PropertySearchReturnDTO(SQLAlchemyDTO[Property]):
    config = SQLAlchemyDTOConfig()


class CreatePropertySchema(BaseModel):
    title: str
    property_category: str
    transaction_type: str
    price: float
    bedrooms: float
    bathrooms: float
    sqm: int
    image_list: list[UploadFile]
    latitude: float
    longitude: float
    street: str
    city: str
    model_config = ConfigDict(arbitrary_types_allowed=True)


class CreatePropertyDTO(PydanticDTO[CreatePropertySchema]):
    config = DTOConfig()


class CreatePropertyReturnDTO(SQLAlchemyDTO[Property]):
    config = SQLAlchemyDTOConfig()


class UpdateStatusModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    active: Optional[bool]


class UpdatePropertySchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    image_list: Optional[list[UploadFile]]
    deleted_images: Optional[list[uuid.UUID]]
    title: Optional[str]
    property_category: Optional[str]
    price: Optional[float]
    bedrooms: Optional[float]
    bathrooms: Optional[float]
    sqm: Optional[int]
    latitude: Optional[float]
    longitude: Optional[float]
    street: Optional[str]
    city: Optional[str]


class UpdatePropertyDTO(PydanticDTO[UpdatePropertySchema]):
    config = DTOConfig()
