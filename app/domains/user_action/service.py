from collections.abc import AsyncGenerator
from database.models.user_action import UserAction
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from sqlalchemy.ext.asyncio import AsyncSession


class UserActionRepository(SQLAlchemyAsyncRepository[UserAction]):
    model_type = UserAction


class UserActionService(SQLAlchemyAsyncRepositoryService[UserAction]):
    repository_type = UserActionRepository


async def provide_user_action_service(
    db_session: AsyncSession,
) -> AsyncGenerator[UserActionService, None]:

    async with UserActionService.new(session=db_session) as service:
        yield service
