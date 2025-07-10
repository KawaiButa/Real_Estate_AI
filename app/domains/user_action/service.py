from collections import defaultdict
from collections.abc import AsyncGenerator
from typing import List
import uuid

from sqlalchemy import select
from database.models.user_action import UserAction
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from sqlalchemy.ext.asyncio import AsyncSession


class UserActionRepository(SQLAlchemyAsyncRepository[UserAction]):
    model_type = UserAction

    async def get_relevant_properties(self, user_id: uuid.UUID) -> List[uuid.UUID]:
        prop_ids_subq = (
            select(UserAction)
            .where(UserAction.user_id == user_id)
            .where(UserAction.action == "view")
            .distinct()
            .limit(5)
        )

        result = await self.session.execute(prop_ids_subq)
        actions = result.scalars().all()
        return [action.property_id for action in actions]


class UserActionService(SQLAlchemyAsyncRepositoryService[UserAction]):
    repository_type = UserActionRepository


async def provide_user_action_service(
    db_session: AsyncSession,
) -> AsyncGenerator[UserActionService, None]:

    async with UserActionService.new(session=db_session) as service:
        yield service
