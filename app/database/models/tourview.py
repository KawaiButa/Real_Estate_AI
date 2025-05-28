from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
import uuid
from sqlalchemy import (
    String,
    ForeignKey,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database.models.image import Image, ImageSchema
from database.models.base import (
    BaseModel,
    BaseSchema,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

if TYPE_CHECKING:
    from database.models.property import Property


@dataclass
class Tourview(BaseModel):
    __tablename__ = "tourviews"
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    image_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("images.id", ondelete="SET NULL"),
        nullable=False,
        unique=True,
    )
    property_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("properties.id", ondelete="SET NULL"),
        nullable=False,
        unique=True,
    )
    # relationship
    image: Mapped["Image"] = relationship(
        "Image",
        lazy="selectin",
        uselist=False,
    )
    property: Mapped["Property"] = relationship("Property", lazy="noload")


class TourviewSchema(BaseSchema):
    image: ImageSchema
    name: str
