from __future__ import annotations
from typing import TYPE_CHECKING
import uuid
from sqlalchemy import (
    ForeignKey,
    String,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database.models.base import BaseModel, BaseSchema

if TYPE_CHECKING:
    from database.models.tag import Tag, TagSchema


class ImageTag(BaseModel):
    __tablename__ = "image_tags"
    image_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("images.id", ondelete="CASCADE")
    )
    tag_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tags.id", ondelete="CASCADE")
    )


class Image(BaseModel):
    __tablename__ = "images"
    url: Mapped[str] = mapped_column(String(), nullable=False, unique=False)
    tags: Mapped[list["Tag"]] = relationship(
        "Tag",
        secondary=ImageTag.__table__,
        lazy="joined",
    )
    model_id = mapped_column(PG_UUID, nullable=True)
    model_type = mapped_column(String(50), nullable=True)
    __mapper_args__ = {"polymorphic_on": model_type}


class ImageSchema(BaseSchema):
    url: str
    model_id: uuid.UUID
    model_type: str
