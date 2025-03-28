from typing import Optional
from litestar.plugins.sqlalchemy import SQLAlchemyDTO, SQLAlchemyDTOConfig
from database.models.address import Address, AddressSchema
from database.models.user import User, UserSchema
from pydantic import BaseModel, ConfigDict
from litestar.datastructures import UploadFile
from litestar.plugins.pydantic import PydanticDTO
from litestar.dto import DataclassDTO, DTOConfig


class ProfileUpdateDTO(SQLAlchemyDTO[User]):
    config = SQLAlchemyDTOConfig()


class UpdateAddress(BaseModel):
    latitude: Optional[str]
    longitude: Optional[str]
    street: Optional[str]
    city: Optional[str]


class UpdateUserSchema(BaseModel):
    username: Optional[str]
    phone: Optional[str]
    address: Optional[UpdateAddress]
    profile_image: Optional[UploadFile]
    model_config = ConfigDict(arbitrary_types_allowed=True)


class UpdateUserDTO(PydanticDTO[UpdateUserSchema]):
    config = DTOConfig(
        include={
            "username",
            "phone",
            "address",
            "profile_image",
        }
    )


class ProfileReturnDTO(PydanticDTO[UserSchema]):
    config = DTOConfig(exclude={"properties.owner", "favorites.owner"})
