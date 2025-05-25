from collections.abc import AsyncGenerator
from uuid import UUID
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from sqlalchemy.ext.asyncio import AsyncSession
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


async def provide_chat_session_service(
    db_session: AsyncSession,
) -> AsyncGenerator[ChatSessionService, None]:
    async with ChatSessionService.new(session=db_session) as service:
        yield service
