from typing import Annotated, Any
from litestar import Controller, Request, Response, get, post
from litestar.di import Provide
from litestar.background_tasks import BackgroundTask, BackgroundTasks
from database.models.chat_message import ChatMessage
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


class ChatMessageController(Controller):
    path = "chat_message"
    tags = ["Chat"]

    dependencies = {"chat_service": Provide(provide_chat_message_service)}

    def notify_message(user: User, message: ChatMessage) -> None:
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
                "sender_id": str(message.sender_id),
                "receiver_id": str(message.receiver_id),
                "chat_session_id": str(message.session_id),
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
    ) -> Response:
        message = await chat_service.create_message(data, request.user.id)
        return Response(
            {"message_id": message.id},
            background=BackgroundTasks(
                [
                    BackgroundTask(
                        self.notify_message,
                        user=request.user,
                        message=message,
                    )
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
