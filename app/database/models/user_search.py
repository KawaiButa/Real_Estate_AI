from __future__ import annotations
from typing import Optional
import uuid
from sqlalchemy import (
    Numeric,
    String,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column
from database.models.base import BaseModel


class UserSearch(BaseModel):
    __tablename__="user_searchs"
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("images.id", ondelete="SET NULL"),
        nullable=True,
        unique=True,
    )
    search_query: Mapped[Optional[str]] = mapped_column(String, nullable=False, unique=False)
    type: Mapped[Optional[str]] = mapped_column(String, nullable=False, unique=False)
    min_price: Mapped[Optional[float]] = mapped_column(
        Numeric(12, 2, asdecimal=False), nullable=True
    )
    max_price: Mapped[Optional[float]] = mapped_column(
        Numeric(12, 2, asdecimal=False), nullable=True
    )