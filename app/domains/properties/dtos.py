from typing import Annotated, Optional
from litestar.plugins.sqlalchemy import SQLAlchemyDTO, SQLAlchemyDTOConfig
from pydantic import BaseModel, ConfigDict, Field

from database.models.property import Property


class PropertySearchReturnDTO(SQLAlchemyDTO[Property]):
    config = SQLAlchemyDTOConfig()


class CreatePropertyDTO(SQLAlchemyDTO[Property]):
    config = SQLAlchemyDTOConfig(
        include={
            "title",
            "property_category",
            "transaction_type",
            "price",
            "bedrooms",
            "bathrooms",
            "sqm",
            "address.latitude",
            "address.longitude",
            "address.street",
            "address.city",
            "address.neighborhood",
        },
    )


class CreatePropertyReturnDTO(SQLAlchemyDTO[Property]):
    config = SQLAlchemyDTOConfig(exclude={"owner_id", "address_id"})


class UpdatePropertyDTO(SQLAlchemyDTO[Property]):
    config = SQLAlchemyDTOConfig(exclude={"owner_id", "address_id"})


class UpdateStatusModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    active: Optional[bool]
