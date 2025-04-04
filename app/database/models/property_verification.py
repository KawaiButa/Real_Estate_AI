from typing import TYPE_CHECKING
from uuid import UUID
from sqlalchemy import ForeignKey, Index, String
from sqlalchemy.orm import mapped_column, relationship, Mapped
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from database.models.base import BaseModel

if TYPE_CHECKING:
    from database.models.property import Property
    from database.models.user import User


class PropertyVerification(BaseModel):
    __tablename__ = "property_verifications"
    property_id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), ForeignKey("properties.id")
    )
    user_id: Mapped[UUID] = mapped_column(PGUUID(as_uuid=True), ForeignKey("users.id"))
    verification_method: Mapped[str] = mapped_column(String(50))
    verification_code: Mapped[str] = mapped_column(String(100))
    property: Mapped["Property"] = relationship("Property", lazy="selectin")
    user: Mapped["User"] = relationship("User", lazy="selectin")

    __table_args__ = (Index("ix_verification_user_property", "user_id", "property_id"),)
