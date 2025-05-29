from __future__ import annotations
from dataclasses import dataclass
import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship, foreign
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from database.models.image import Image, ImageSchema
from database.models.user import User, UserSchema
from database.models.base import BaseModel, BaseSchema

if TYPE_CHECKING:
    from database.models.chat_session import ChatSession
    
@dataclass
class ChatMessage(BaseModel):
    __tablename__ = "chat_messages"
    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    content: Mapped[Optional[str]] = mapped_column(Text())
    images: Mapped[list[Image]] = relationship(
        "Image",
        primaryjoin=(
            "and_(foreign(Image.model_id) == ChatMessage.id, "
            "Image.model_type == 'chat_message')"
        ),
        foreign_keys=[foreign(Image.model_id)],
        remote_side=[Image.model_id],
        backref="chat_message",
        lazy="selectin",
    )
    sender_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("chat_sessions.id", ondelete="CASCADE")
    )

    sender: Mapped["User"] = relationship("User", lazy="joined")
    session: Mapped["ChatSession"] = relationship("ChatSession", lazy="noload", foreign_keys=[session_id])

class ChatMessageSchema(BaseSchema):
    id: uuid.UUID
    session_id: uuid.UUID
    sender_id: uuid.UUID
    content: Optional[str]
    images: list[ImageSchema]
    sender: Optional[UserSchema]
