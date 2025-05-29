from typing import Annotated, Any
import uuid
from litestar import Controller, Request, Response, get, post
from litestar.di import Provide
from litestar.background_tasks import BackgroundTask, BackgroundTasks
from domains.chat_session.service import (
    ChatSessionService,
    provide_chat_session_service,
)
from domains.chat_session.controller import provide_limit_offset_pagination
from database.models.chat_message import ChatMessage, ChatMessageSchema
from database.models.user import User
from domains.chat_message.dtos import AskAIDTO, CreateMessageDTO
from domains.chat_message.service import (
    ChatMessageService,
    provide_chat_message_service,
)
from domains.notification.service import NotificationService
from litestar.params import Body
from litestar.enums import RequestEncodingType
from litestar.security.jwt import Token
from litestar.status_codes import HTTP_200_OK
from litestar.repository.filters import LimitOffset

from litestar.pagination import OffsetPagination


class ChatMessageController(Controller):
    path = "chat_message"
    tags = ["Chat"]

    dependencies = {
        "chat_service": Provide(provide_chat_message_service),
        "chat_session_service": Provide(provide_chat_session_service),
        "limit_offset": Provide(provide_limit_offset_pagination, sync_to_thread=True),
    }

    def notify_message(self, user: User, message: ChatMessage) -> None:
        if not user.device_token:
            return
        notify_service = NotificationService()
        title = "You have a new message"
        body = f"You have a new message from {user.name}. \n{message.content}"
        notify_service.send_to_token(
            token=user.device_token,
            title=title,
            body=body,
            data={
                "type": "chat",
                "content": message.content,
                "sender_id": str(message.sender_id),
                "chat_session_id": str(message.session_id),
                "created_at": message.created_at.timestamp(),
            },
        )

    @post("")
    async def create_message(
        self,
        data: Annotated[
            CreateMessageDTO, Body(media_type=RequestEncodingType.MULTI_PART)
        ],
        request: Request[User, Token, Any],
        chat_service: ChatMessageService,
        chat_session_service: ChatSessionService,
    ) -> Response:
        message, data = await chat_service.create_message(data, request.user.id)
        return Response(
            chat_service.to_schema(message, schema_type=ChatMessageSchema),
            background=BackgroundTasks(
                [
                    BackgroundTask(
                        self.notify_message,
                        request.user,
                        message,
                    ),
                    BackgroundTask(
                        chat_session_service.update_last_message,
                        data.session_id,
                        message,
                    ),
                ]
            ),
        )

    @post("/ai", no_auth=True, status_code=HTTP_200_OK)
    async def ask_ai(
        self,
        data: Annotated[AskAIDTO, Body(media_type=RequestEncodingType.MULTI_PART)],
        chat_service: ChatMessageService,
    ) -> str:
        return await chat_service.ask_ai(data)

    @get("/user/{user_id:uuid}")
    async def get_chat_by_user_id(
        self,
        user_id: uuid.UUID,
        limit_offset: LimitOffset,
        request: Request[User, Token, Any],
        chat_service: ChatMessageService,
    ) -> Any:
        return await chat_service.chat_messages_by_user_id(
            user_id, request.user.id, limit_offset
        )

    @get("/chat_session/{chat_session_id: uuid}")
    async def get_chat_by_session_id(
        self,
        chat_session_id: uuid.UUID,
        limit_offset: LimitOffset,
        chat_service: ChatMessageService,
    ) -> Any:
        return await chat_service.chat_messages_by_session_id(
            chat_session_id, limit_offset
        )
