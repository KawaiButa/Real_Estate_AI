from __future__ import annotations
import datetime
from typing import TYPE_CHECKING
import uuid
from sqlalchemy import (
    DateTime,
    String,
    ForeignKey,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database.models.base import BaseModel, BaseSchema

if TYPE_CHECKING:
    from database.models.tag import Tag, TagSchema


class ArticleTags(BaseModel):
    __tablename__ = "article_tags"
    article_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("articles.id", ondelete="CASCADE")
    )
    tag_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("tags.id", ondelete="CASCADE")
    )


class Article(BaseModel):
    __tablename__ = "articles"
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    publish_date: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.datetime.now,
    )
    content: Mapped[str] = mapped_column(String, nullable=False)
    short_description: Mapped[str] = mapped_column(String, nullable=False)
    author: Mapped[str] = mapped_column(String(255), nullable=False)
    tags: Mapped[list["Tag"]] = relationship(
        "Tag", secondary=ArticleTags.__table__, lazy="selectin"
    )


class ArticleSchema(BaseSchema):
    title: str
    publish_date: datetime.datetime
    content: str
    short_description: str
    author: str
    tags: list["TagSchema"]
