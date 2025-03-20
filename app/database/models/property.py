from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import uuid
from sqlalchemy import (
    Boolean,
    Integer,
    Numeric,
    String,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database.models.base import BaseModel, BaseSchema
from litestar.plugins.sqlalchemy import (
    base,
)

if TYPE_CHECKING:
    from database.models.address import AddressSchema
    from database.models.user import UserSchema
    from database.models.address import Address
    from database.models.user import User


@dataclass
class Favorite(BaseModel):
    __tablename__ = "favorites"
    user_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
    )
    property_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("properties.id", ondelete="CASCADE"),
    )


@dataclass
class Property(BaseModel):
    __tablename__ = "properties"
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    property_category: Mapped[str] = mapped_column(String(50), nullable=False)
    transaction_type: Mapped[str] = mapped_column(String(50), nullable=False)
    price: Mapped[float] = mapped_column(Numeric(12, 2), nullable=False)
    bedrooms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    bathrooms: Mapped[float] = mapped_column(Numeric(3, 1), default=0, nullable=False)
    sqm: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False)
    active: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    owner_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
    )
    address_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("addresses.id", ondelete="SET NULL"),
        unique=True,
        nullable=True,
    )
    owner: Mapped["User"] = relationship(
        "User", back_populates="properties", lazy="selectin"
    )
    address: Mapped["Address"] = relationship(
        "Address", back_populates="property", uselist=False, lazy="selectin"
    )


class PropertySchema(BaseSchema):
    title: str
    property_category: str
    transaction_type: str
    price: float
    bedrooms: float
    bathrooms: float
    sqm: int
    status: str
    owner_id: uuid.UUID
    address_id: uuid.UUID | None
    owner: "UserSchema"
    address: "AddressSchema"
