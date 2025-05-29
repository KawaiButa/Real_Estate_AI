from collections.abc import AsyncGenerator
from datetime import datetime
from uuid import UUID
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from sqlalchemy.ext.asyncio import AsyncSession
from database.models.chat_message import ChatMessage
from database.models.chat_session import ChatSession


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

    async def update_last_message(self, session_id: UUID, target: ChatMessage):
        try:
            chat_session = await self.update(
                data={
                    "last_message": target.content,
                    "last_message_time": datetime.now(),
                },
                item_id=session_id,
            )
            return chat_session
        except Exception as e:
            print(e)
            await self.repository.session.rollback()
        finally:
            await self.repository.session.commit()


async def provide_chat_session_service(
    db_session: AsyncSession,
) -> AsyncGenerator[ChatSessionService, None]:
    async with ChatSessionService.new(session=db_session) as service:
        yield service
