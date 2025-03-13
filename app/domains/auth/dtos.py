from dataclasses import dataclass
from typing import Any
from litestar import Response
from litestar.plugins.sqlalchemy import SQLAlchemyDTO, SQLAlchemyDTOConfig
from pydantic import BaseModel, ConfigDict
from database.models.user import User, UserSchema
from litestar.dto import DataclassDTO, DTOConfig


class RegisterDTO(SQLAlchemyDTO[User]):
    config = SQLAlchemyDTOConfig(include={"name", "email", "password", "phone"})


@dataclass
class RegisterReturnModel:
    auth: Any
    user: SQLAlchemyDTO[User]


class RegisterReturnDTO(DataclassDTO[RegisterReturnModel]):
    config = DTOConfig()


class LoginDTO(SQLAlchemyDTO[User]):
    config = SQLAlchemyDTOConfig(include={"email", "password", "device_token"})


class MyProfileReturnDTO(SQLAlchemyDTO[User]):
    config = SQLAlchemyDTOConfig(
        exclude={
            User.password,
            User.address_id,
        }
    )


class LoginReturnSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    token: str
    user: UserSchema


class ForgotPasswordSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    email: str


class ResetPasswordSchema(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    new_password: str
    token: str
