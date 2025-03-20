from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Hashable
from sqlalchemy import ColumnElement, ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from advanced_alchemy.mixins import UniqueMixin
from advanced_alchemy.utils.text import slugify
from litestar.plugins.sqlalchemy import (
    base,
)

from database.models.base import BaseModel, BaseSchema


class UserRole(base.DefaultBase):
    __tablename__ = "user_roles"
    user_id: Mapped[PG_UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id"), primary_key=True
    )
    role_id: Mapped[PG_UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("roles.id"), primary_key=True
    )


@dataclass
class Role(BaseModel, base.SlugKey, UniqueMixin):
    __tablename__ = "roles"
    name: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
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

class RoleSchema(BaseSchema):
    name: str