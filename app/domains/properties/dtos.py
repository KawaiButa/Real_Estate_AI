from typing import Optional, Self
import uuid
from litestar.plugins.sqlalchemy import SQLAlchemyDTO, SQLAlchemyDTOConfig
from pydantic import (
    BaseModel,
    ConfigDict,
    create_model,
    field_validator,
    model_validator,
)
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
    description: str
    city: str
    neighborhood: Optional[str]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_validator("price", "bedrooms", "bathrooms", "sqm")
    def non_negative(cls, v, field):
        if v < 0:
            raise ValueError(f"{field.name} must be non-negative")
        return v

    @field_validator("latitude")
    def valid_latitude(cls, v):
        if not (-90 <= v <= 90):
            raise ValueError("latitude must be between -90 and 90")
        return v

    @field_validator("longitude")
    def valid_longitude(cls, v):
        if not (-180 <= v <= 180):
            raise ValueError("longitude must be between -180 and 180")
        return v

    @model_validator(mode="after")
    def check_address_and_images(self) -> Self:
        if len(self.image_list) < 3:
            raise ValueError("image_list must contain at least 3 items")
        return self


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
    description: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    street: Optional[str]
    city: Optional[str]
    neighborhood: Optional[str]

    @field_validator("price", "bedrooms", "bathrooms", "sqm")
    def non_negative(cls, v, field):
        if v is None:
            return v
        if v < 0:
            raise ValueError(f"{field.name} must be non-negative")
        return v

    @field_validator("latitude")
    def valid_latitude(cls, v):
        if v is not None and not (-90 <= v <= 90):
            raise ValueError("latitude must be between -90 and 90")
        return v

    @field_validator("longitude")
    def valid_longitude(cls, v):
        if v is not None and not (-180 <= v <= 180):
            raise ValueError("longitude must be between -180 and 180")
        return v

    @model_validator(mode="after")
    def check_address_and_images(self) -> Self:
        # If any address field is provided, ensure all are provided.
        address_fields = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "street": self.street,
            "city": self.city,
        }
        provided = [value is not None for value in address_fields.values()]
        if any(provided) and not all(provided):
            missing = [name for name, value in address_fields.items() if value is None]
            raise ValueError(
                f"If updating address, the following fields must also be provided: {missing}"
            )
        # Validate image_list length if provided.
        if self.image_list is not None and len(self.image_list) < 3:
            raise ValueError("image_list must contain at least 3 items")
        return self


class CreatePropertyDTO(PydanticDTO[CreatePropertySchema]):
    config = DTOConfig()


class CreatePropertyReturnDTO(SQLAlchemyDTO[Property]):
    config = SQLAlchemyDTOConfig()


class UpdateStatusModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    active: Optional[bool]


class UpdatePropertyDTO(PydanticDTO[UpdatePropertySchema]):
    config = DTOConfig()
