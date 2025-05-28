from __future__ import annotations
from dataclasses import dataclass
import datetime
from typing import TYPE_CHECKING
import uuid
from sqlalchemy import (
    DateTime,
    ForeignKey,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database.models.base import (
    BaseModel,
    BaseSchema,
)
from litestar.dto import dto_field
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
if TYPE_CHECKING:
    from database.models.chat_message import ChatMessage, ChatMessageSchema
    from database.models.user import User, UserSchema

@dataclass
class ChatSession(BaseModel):
    __tablename__ = "chat_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    user_1_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )

    user_2_id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE")
    )

    last_message: Mapped[String | None] = mapped_column(String())
    last_message_time: Mapped[datetime.datetime | None] = mapped_column(
        DateTime(timezone=True), info=dto_field("read-only")
    )

    user_1: Mapped["User"] = relationship("User", foreign_keys=[user_1_id], lazy="joined")
    user_2: Mapped["User"] = relationship("User", foreign_keys=[user_2_id], lazy="joined")

    message: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage", back_populates="session", lazy="noload"
    )


class ChatSessionSchema(BaseSchema):
    user_1: UserSchema
    user_2: UserSchema
    last_message: str
    last_message_time: datetime.datetime
    message: Mapped[list[ChatMessageSchema]]
