from collections import defaultdict
from collections.abc import AsyncGenerator
import uuid

from sqlalchemy import select
from database.models.user_action import UserAction
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from sqlalchemy.ext.asyncio import AsyncSession


class UserActionRepository(SQLAlchemyAsyncRepository[UserAction]):
    model_type = UserAction
    async def get_relevant_properties(self, user_id: uuid.UUID) -> dict:
        prop_ids_subq = (
            select(UserAction.property_id)
            .where(UserAction.user_id == user_id)
            .distinct()
            .limit(10)
        ).subquery()

        # Step 2: fetch all actions for those properties
        result = await self.session.execute(
            select(UserAction)
            .where(
                UserAction.user_id == user_id,
                UserAction.property_id.in_(select(prop_ids_subq))
            )
            .order_by(UserAction.property_id, UserAction.created_at)
        )
        actions = result.scalars().all()

        grouped: dict = {}
        for act in actions:
            grouped[str(act.property_id)].append(act)
        return grouped
class UserActionService(SQLAlchemyAsyncRepositoryService[UserAction]):
    repository_type = UserActionRepository


async def provide_user_action_service(
    db_session: AsyncSession,
) -> AsyncGenerator[UserActionService, None]:

    async with UserActionService.new(session=db_session) as service:
        yield service
