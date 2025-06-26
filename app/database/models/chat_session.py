from __future__ import annotations
from dataclasses import dataclass
import datetime
from typing import TYPE_CHECKING, Optional
import uuid
from sqlalchemy import (
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database.models.base import (
    BaseModel,
    BaseSchema,
)
from litestar.dto import dto_field
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from database.models.chat_message import ChatMessage, ChatMessageSchema

if TYPE_CHECKING:
    from database.models.user import User, UserSchema


@dataclass
class ChatSession(BaseModel):
    __tablename__ = "chat_sessions"
    __table_args__ = (
        ForeignKeyConstraint(
            ["last_message_id"],
            ["chat_messages.id"],
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    user_1_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=True
    )

    user_2_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )

    last_message_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("chat_messages.id", ondelete="SEt NULL"),
        nullable=True,
    )
    user_1: Mapped[Optional["User"]] = relationship(
        "User", foreign_keys=[user_1_id], lazy="joined"
    )
    user_2: Mapped["User"] = relationship(
        "User", foreign_keys=[user_2_id], lazy="joined"
    )

    last_message = relationship(
        "ChatMessage",
        primaryjoin=(last_message_id == ChatMessage.id),
        uselist=False,
        lazy="selectin"
    )

    messages = relationship(
        "ChatMessage",
        primaryjoin=(id == ChatMessage.session_id),
        order_by=ChatMessage.created_at,
        lazy="noload"
    )

class ChatSessionSchema(BaseSchema):
    user_1: UserSchema
    user_2: UserSchema
    last_message: Optional[ChatMessageSchema]
    message: Mapped[list[ChatMessageSchema]]
