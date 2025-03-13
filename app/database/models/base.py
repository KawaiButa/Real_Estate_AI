import datetime
from typing import Optional
from litestar.plugins.sqlalchemy import base
from dataclasses import dataclass
from sqlalchemy import DateTime, func
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column
from litestar.dto import dto_field
from pydantic import BaseModel as _BaseModel, ConfigDict
import uuid
class BaseModel(base.DefaultBase):
    __abstract__ = True
    id: Mapped[PG_UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        nullable=False,
        unique=True,
        primary_key=True,
        info=dto_field("read-only"),
        default=uuid.uuid4
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.now, info=dto_field("read-only")
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.now,
        onupdate=datetime.datetime.now,
        info=dto_field("read-only"),
    )
class BaseSchema(_BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: Optional[uuid.UUID] = None
    created_at: Optional[datetime.datetime] = None
    updated_at: Optional[datetime.datetime] = None
    