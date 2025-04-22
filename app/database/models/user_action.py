from __future__ import annotations
from datetime import datetime
from typing import Optional
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

from database.models.base import BaseModel


class UserAction(BaseModel):
    __tablename__ = "user_actions"
    user_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        unique=False,
    )
    property_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("properties.id", ondelete="SET NULL"),
        nullable=True,
        unique=False,
    )
    action: Mapped[str] = mapped_column(String, nullable=False, unique=False)
