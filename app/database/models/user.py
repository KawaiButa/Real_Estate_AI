from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Optional
import uuid
from sqlalchemy import (
    DateTime,
    String,
    Text,
    Boolean,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database.models.image import Image, ImageSchema
from database.models.tag import Tag, TagSchema
from database.models.partner_registration import (
    PartnerRegistration,
    PartnerRegistrationSchema,
)
from database.models.base import BaseModel, BaseSchema
from database.models.role import RoleSchema, UserRole, Role
from litestar.plugins.sqlalchemy import base
from database.models.address import Address, AddressSchema
from database.models.property import Favorite, Property, PropertySchema
from litestar.dto import dto_field


class UserTag(BaseModel):
    __tablename__ = "user_tags"
    user_id: Mapped[PG_UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id"), primary_key=False
    )
    tag_id: Mapped[PG_UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tags.id"), primary_key=False
    )


class User(BaseModel):
    __tablename__ = "users"
    name: Mapped[str] = mapped_column(String(255), unique=False, nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    phone: Mapped[str | None] = mapped_column(String(20), nullable=True)
    password: Mapped[str] = mapped_column(
        Text, nullable=False, info=dto_field("write-only")
    )
    verified: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    address_id: Mapped[uuid.UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("addresses.id", ondelete="SET NULL"),
        unique=True,
        nullable=True,
    )
    image_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("images.id", ondelete="SET NULL"),
        nullable=True,
        unique=True,
    )
    device_token: Mapped[Optional[str]] = mapped_column(
        String(255), unique=True, nullable=True, default=None
    )
    reset_password_token: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, info=dto_field("private")
    )
    reset_password_expires: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, info=dto_field("private")
    )
    profile_image: Mapped[Optional["Image"]] = relationship(
        "Image",
        lazy="selectin",
        uselist=False,
    )
    # Relationships
    address: Mapped[Optional["Address"]] = relationship(
        "Address", uselist=False, lazy="selectin"
    )
    roles: Mapped[list["Role"]] = relationship(
        "Role", secondary=UserRole.__table__, lazy="selectin"
    )
    properties: Mapped[list["Property"]] = relationship(
        "Property", back_populates="owner", lazy="noload"
    )
    partner_registration: Mapped[PartnerRegistration | None] = relationship(
        "PartnerRegistration",
        back_populates="user",
        lazy="joined",
    )
    favorites: Mapped[list[Property]] = relationship(
        "Property",
        secondary=Favorite.__table__,
        lazy="noload",
    )
    tags: Mapped[list[Tag]] = relationship(
        "Tag", secondary=UserTag.__table__, lazy="selectin"
    )


class UserSchema(BaseSchema):
    name: str
    email: str
    phone: str
    verified: bool
    address_id: Optional[uuid.UUID] = None
    device_token: Optional[str] = None
    profile_image: Optional[ImageSchema] = None
    # Relationships
    address: Optional[AddressSchema] = None
    roles: list[RoleSchema]
    properties: Optional[list[PropertySchema]]
    registration: Optional[PartnerRegistrationSchema] = None
    favorites: Optional[list[PropertySchema]]
    tags: list[TagSchema]


UserSchema.model_rebuild()
