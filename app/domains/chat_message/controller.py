from typing import Annotated, Any
import uuid
from litestar import Controller, Request, Response, get, post
from litestar.di import Provide
from litestar.background_tasks import BackgroundTask, BackgroundTasks
from sqlalchemy import select
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
from litestar.exceptions import NotFoundException
from litestar.pagination import OffsetPagination
from sqlalchemy.orm import noload


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
        title = "AI Assistant"
        body = (
            f"You have a new message from {user.name}."
            if message.sender_id
            else "AI has the answer you need"
        )
        notify_service.send_to_token(
            token=user.device_token,
            title=title,
            body=body,
            data={
                "type": "chat",
                "id": str(message.id),
                "sender_id": str(message.sender_id),
                "chat_session_id": str(message.session_id),
                "created_at": str(message.created_at.timestamp()),
            },
        )

    async def chat_with_ai(
        self,
        data: CreateMessageDTO,
        user: User,
        chat_service: ChatMessageService,
    ):
        message = await chat_service.ai_respond_to_user(data, user_id=user.id)
        self.notify_message(user, message)

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
        message = await chat_service.create_message(data, request.user.id)
        background_task_list = [
            BackgroundTask(
                chat_session_service.update_last_message,
                message.session_id,
                message,
            ),
        ]
        if data.is_ai:
            background_task_list.append(
                BackgroundTask(self.chat_with_ai, data, request.user, chat_service)
            )
        else:
            background_task_list.append(
                BackgroundTask(
                    self.notify_message,
                    request.user,
                    message,
                ),
            )
        return Response(
            chat_service.to_schema(message, schema_type=ChatMessageSchema),
            background=BackgroundTasks(background_task_list),
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

    @get("/{chat_message_id:uuid}")
    async def get_chat_message_by_id(
        self, chat_message_id: uuid.UUID, chat_service: ChatMessageService
    ) -> ChatMessageSchema:
        query = (
            select(ChatMessage)
            .options(noload(ChatMessage.sender), noload(ChatMessage.session))
            .where(ChatMessage.id == chat_message_id)
        )
        message = (await chat_service.repository.session.execute(query)).scalar()
        if not message:
            raise NotFoundException(f"No message found with id {chat_message_id}")
        return chat_service.to_schema(message, schema_type=ChatMessageSchema)
