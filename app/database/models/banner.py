from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Hashable, Optional
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


class Banner(BaseModel):
    __tablename__="banners"
    title: Mapped[str] = mapped_column(String(), nullable=True)
    url: Mapped[str] = mapped_column(String(), nullable=False)
    content: Mapped[str] = mapped_column(String(),nullable=False)
class BannerSchema(BaseSchema):
    title: Optional[str]
    url: str
    content: str