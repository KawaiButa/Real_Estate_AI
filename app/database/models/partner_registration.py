from __future__ import annotations
import enum
from typing import TYPE_CHECKING
import uuid
from dataclasses import dataclass
from datetime import date
from sqlalchemy import Enum, String, Date, ForeignKey
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database.models.base import BaseModel, BaseSchema
from typing import Optional

if TYPE_CHECKING:
    from database.models.user import UserSchema
    from database.models.user import User


class PartnerType(enum.Enum):
    INDIVIDUAL = "Individual"
    ENTERPRISE = "Enterprise"


@dataclass
class PartnerRegistration(BaseModel):
    __tablename__ = "partner_registrations"
    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    user: Mapped["User"] = relationship(
        "User", back_populates="partner_registration", lazy="joined"
    )

    type: Mapped[PartnerType] = mapped_column(
        Enum(PartnerType, name="partner_type"), nullable=False
    )

    profile_url: Mapped[str] = mapped_column(String(255), nullable=True)
    type: Mapped[PartnerType] = mapped_column(Enum(PartnerType), nullable=False)
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)

    # Corporate Credentials (for corporate partners)
    business_registration_certificate_url: Mapped[str] = mapped_column(
        String(255), nullable=True
    )
    tax_id: Mapped[str] = mapped_column(String(100), nullable=True)
    authorized_representative_name: Mapped[str] = mapped_column(
        String(255), nullable=True
    )
    approved: Mapped[bool | None] = mapped_column(
        String(100), nullable=True, default=None, server_default="NULL"
    )


class PartnerRegistrationSchema(BaseSchema):
    user_id: Optional[uuid.UUID] = None
    user: Optional["UserSchema"] = None
    date_of_birth: date
    profile_url: Optional[str] = None
    type: PartnerType
    business_registration_certificate_url: Optional[str] = None
    tax_id: Optional[str] = None
    authorized_representative_name: Optional[str] = None
    approved: Optional[bool] = False
