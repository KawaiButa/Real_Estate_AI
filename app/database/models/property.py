from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
import uuid
from sqlalchemy import (
    Boolean,
    Integer,
    Numeric,
    String,
    ForeignKey,
    and_,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database.models.property_type import PropertyType, PropertyTypeSchema
from database.models.base import BaseModel, BaseSchema
from database.models.image import Image, ImageSchema
from sqlalchemy.orm import foreign, relationship, Mapped

if TYPE_CHECKING:
    from database.models.tag import Tag, TagSchema
    from database.models.address import Address, AddressSchema
    from database.models.user import User, UserSchema


class PropertyTag(BaseModel):
    __tablename__ = "property_tags"
    property_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("properties.id", ondelete="CASCADE")
    )
    tag_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tags.id", ondelete="CASCADE")
    )


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


class Property(BaseModel):
    __tablename__ = "properties"
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    property_category: Mapped[str] = mapped_column(String(50), nullable=False)
    property_type_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("property_types.id", ondelete="CASCADE"),
        nullable=False,
    )
    property_type: Mapped[PropertyType] = relationship("PropertyType", lazy="selectin")
    transaction_type: Mapped[str] = mapped_column(String(50), nullable=False)
    price: Mapped[float] = mapped_column(
        Numeric(12, 2, asdecimal=False), nullable=False
    )
    bedrooms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    bathrooms: Mapped[int] = mapped_column(Numeric(3, 1), default=0, nullable=False)
    sqm: Mapped[float] = mapped_column(
        Numeric(6, 2, asdecimal=False),
        nullable=False,
    )
    description: Mapped[str] = mapped_column(String(), nullable=False)
    status: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )
    images: Mapped[list[Image]] = relationship(
        "Image",
        primaryjoin=(
            "and_(foreign(Image.model_id) == Property.id, "
            "Image.model_type == 'property')"
        ),
        foreign_keys=[foreign(Image.model_id)],
        remote_side=[Image.model_id],
        backref="property",
        lazy="selectin",
    )
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
        "User", back_populates="properties", lazy="joined"
    )
    address: Mapped["Address"] = relationship("Address", uselist=False, lazy="selectin")
    tags: Mapped[list[Tag]] = relationship(
        "Tag", secondary=PropertyTag.__table__, lazy="selectin"
    )


class PropertySchema(BaseSchema):
    title: str
    property_category: str
    property_type_id: str
    property_type: PropertyTypeSchema
    transaction_type: str
    price: float
    bedrooms: int
    bathrooms: int
    sqm: float
    description: str
    status: str
    owner_id: uuid.UUID
    address_id: uuid.UUID | None
    owner: Optional["UserSchema"]
    address: Optional["AddressSchema"]
    images: list[ImageSchema]
    tags: list[TagSchema]
