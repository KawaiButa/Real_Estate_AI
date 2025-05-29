from typing import Any
import uuid
from litestar import Controller, Request, get
from litestar.di import Provide
from litestar.security.jwt import Token
from sqlalchemy import and_, or_
from database.models.chat_session import ChatSession
from database.models.user import User
from litestar.pagination import OffsetPagination
from litestar.repository.filters import LimitOffset
from domains.chat_session.service import (
    ChatSessionService,
    provide_chat_session_service,
)
from litestar.params import Parameter
from litestar.exceptions import NotFoundException

def provide_limit_offset_pagination(
    page_size: int | None = Parameter(
        query="page_size",
        default=10,
        ge=0,
        required=False,
    ),
    page: int | None = Parameter(
        query="page",
        default=1,
        ge=1,
        required=False,
    ),
) -> LimitOffset:
    return LimitOffset(page_size, (page - 1) * page_size)


class ChatSessionController(Controller):
    path = "chat_session"
    tags = ["Chat"]

    dependencies = {
        "chat_session_service": Provide(provide_chat_session_service),
        "limit_offset": Provide(provide_limit_offset_pagination, sync_to_thread=True),
    }

    @get("")
    async def get_chat_session_list(
        self,
        request: Request[User, Token, Any],
        limit_offset: LimitOffset,
        chat_session_service: ChatSessionService,
    ) -> OffsetPagination[ChatSession]:
        session_list, count = await chat_session_service.list_and_count(
            ChatSession.user_1_id == request.user.id
            or ChatSession.user_2_id == request.user.id,
            limit_offset,
        )
        return OffsetPagination(
            items=list(session_list),
            limit=limit_offset.limit,
            offset=limit_offset.offset,
            total=count,
        )
    @get("/{chat_session_id:uuid}")
    async def get_chat_session(self, chat_session_id: uuid.UUID, chat_session_service: ChatSessionService) -> ChatSession:
        chat_session = await chat_session_service.get_one_or_none(ChatSession.id == chat_session_id)
        if not chat_session:
            raise NotFoundException(f"No chat session found with id: {chat_session_id}")
        return chat_session