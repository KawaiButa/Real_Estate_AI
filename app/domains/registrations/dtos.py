import datetime
from typing import Optional
import uuid
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from litestar.datastructures import UploadFile
from database.models.partner_registration import PartnerRegistrationSchema, PartnerType
from litestar.plugins.pydantic.dto import PydanticDTO
from litestar.dto import DTOConfig


class RegisterWriteModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    profile_img: UploadFile
    business_registration_certificate_img: Optional[UploadFile] = None
    date_of_birth: datetime.date
    authorized_representative_name: Optional[str] = None
    type: PartnerType
    business_registration_certificate_url: Optional[str] = None
    profile_url: Optional[str] = None
    tax_id: Optional[str] = None
    user_id: Optional[uuid.UUID] = None
    @field_validator("date_of_birth")
    def validate_date_of_birth(cls, v):
        if v and v > datetime.date.today():
            raise ValueError("Date of birth cannot be in the future")
        return v

    @field_validator("tax_id")
    def validate_tax_id(cls, v):
        if v and len(v) > 100:
            raise ValueError("Tax ID must not exceed 100 characters")
        return v

    @field_validator("authorized_representative_name")
    def validate_authorized_representative_name(cls, v):
        if v and len(v) > 255:
            raise ValueError(
                "Authorized representative name must not exceed 255 characters"
            )
        return v

    @model_validator(mode="before")
    def validate_type_specific_fields(cls, values):
        type = values.get("type")
        if type == PartnerType.ENTERPRISE:
            if not values.get("tax_id"):
                raise ValueError("Tax ID is required for enterprise partners")
            if not values.get("authorized_representative_name"):
                raise ValueError(
                    "Authorized representative name is required for enterprise partners"
                )
        return values

class RegisterWriteDTO(PydanticDTO[RegisterWriteModel]):
    config = DTOConfig(
        exclude={
            "user",
            "user_id",
            "id",
            "created_at",
            "updated_at",
            "profile_url",
            "business_registration_certificate_url",
        }
    )