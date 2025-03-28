from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Hashable
import uuid
from sqlalchemy import (
    Boolean,
    ColumnElement,
    Integer,
    Numeric,
    String,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database.models.base import BaseModel, BaseSchema

from advanced_alchemy.mixins import UniqueMixin
from advanced_alchemy.utils.text import slugify
from litestar.plugins.sqlalchemy import (
    base,
)

if TYPE_CHECKING:
    from database.models.address import AddressSchema
    from database.models.user import UserSchema
    from database.models.address import Address
    from database.models.user import User


class Tag(BaseModel, base.SlugKey, UniqueMixin):
    __tablename__ = "tags"
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    type: Mapped[str] = mapped_column(String(255), nullable=True, unique=False)

    @classmethod
    def unique_hash(cls, name: str, slug: str | None = None) -> Hashable:
        return slugify(name)

    @classmethod
    def unique_filter(
        cls,
        name: str,
        slug: str | None = None,
    ) -> ColumnElement[bool]:
        return cls.slug == slugify(name)


class TagSchema(BaseSchema):
    name: str
    type: str