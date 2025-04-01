from __future__ import annotations
from dataclasses import dataclass
from typing import Hashable
from sqlalchemy import ColumnElement, String
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.orm import Mapped, mapped_column
from advanced_alchemy.mixins import UniqueMixin
from advanced_alchemy.utils.text import slugify
from litestar.plugins.sqlalchemy import (
    base,
)

from database.models.base import BaseModel, BaseSchema


@dataclass
class PropertyType(BaseModel, base.SlugKey, UniqueMixin):
    __tablename__ = "property_types"
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


class PropertyTypeSchema(BaseSchema):
    name: str
