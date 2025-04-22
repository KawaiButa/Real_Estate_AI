from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import uuid
from sqlalchemy import (
    Numeric,
    String,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database.models.base import BaseModel

if TYPE_CHECKING:
    from database.models.property import Property

class UserSearchProperty(BaseModel):
    __tablename__ = "user_search_properties"
    user_search_id: Mapped[PG_UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("user_searches.id"), primary_key=True
    )
    property_id: Mapped[PG_UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("properties.id"), primary_key=True
    )


class UserSearch(BaseModel):
    __tablename__ = "user_searches"
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        unique=False,
    )
    search_query: Mapped[Optional[str]] = mapped_column(
        String, nullable=True, unique=False
    )
    type: Mapped[Optional[str]] = mapped_column(String, nullable=True, unique=False)
    min_price: Mapped[Optional[float]] = mapped_column(
        Numeric(12, 2, asdecimal=False), nullable=True
    )
    max_price: Mapped[Optional[float]] = mapped_column(
        Numeric(12, 2, asdecimal=False), nullable=True
    )
    properties: Mapped[list["Property"]] = relationship(
        "Property", secondary=UserSearchProperty.__table__, lazy="selectin"
    )
