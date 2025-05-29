from collections.abc import AsyncGenerator
from datetime import datetime
from uuid import UUID
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from sqlalchemy.ext.asyncio import AsyncSession
from database.models.chat_message import ChatMessage
from database.models.chat_session import ChatSession
from configs.firebase_admin import db


class ChatSessionRepository(SQLAlchemyAsyncRepository[ChatSession]):
    model_type = ChatSession


class ChatSessionService(SQLAlchemyAsyncRepositoryService[ChatSession]):
    repository_type = ChatSessionRepository

    async def create_session(self, user_1_id: UUID, user_2_id: UUID) -> ChatSession:
        return await self.create(
            {
                "user_1_id": user_1_id,
                "user_2_id": user_2_id,
            },
            auto_commit=True,
            auto_refresh=True,
        )

    def update_last_message_on_firebase(self, chat_session: ChatSession) -> None:
        doc_ref = db.collection("chat_sessions").document(
            f"{chat_session.user_1_id}_{chat_session.user_2_id}"
        )
        doc_ref.set(
            {
                "sender_id": chat_session.last_message.id,
                "last_message": chat_session.last_message.content,
                "last_message_time": chat_session.last_message.created_at,
            },
            merge=True,
        )

    async def update_last_message(self, session_id: UUID, target: ChatMessage):
        try:
            chat_session = await self.update(
                data={
                    "last_message_id": target.id
                },
                item_id=session_id,
            )
            # self.update_last_message_on_firebase(chat_session)
            return chat_session
        except Exception as e:
            print(e)
            await self.repository.session.rollback()
            raise e
        finally:
            await self.repository.session.commit()


async def provide_chat_session_service(
    db_session: AsyncSession,
) -> AsyncGenerator[ChatSessionService, None]:
    async with ChatSessionService.new(session=db_session) as service:
        yield service
