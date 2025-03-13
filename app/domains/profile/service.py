import uuid
from collections.abc import AsyncGenerator
from database.models.user import User
from domains.auth.repository import UserRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from sqlalchemy.ext.asyncio import AsyncSession
from litestar.exceptions import ValidationException
from litestar.dto import DTOData


class ProfileService(SQLAlchemyAsyncRepositoryService[User]):
    repository_type = UserRepository

    async def get_profile(self, user_id: uuid.UUID) -> User:
        profile = await self.get_one_or_none(User.id.__eq__(user_id))
        if not profile:
            raise ValidationException(f"Cannot find user with id {user_id}")
        return profile

    async def update_profile(self, data: DTOData[User], user_id: uuid.UUID) -> User:
        profile = await self.get_one_or_none(User.id.__eq__(user_id))
        if not profile:
            raise ValidationException(f"Cannot find user with id {user_id}")
        profile = data.update_instance(profile)
        profile = await self.update(
            data=profile, item_id=profile.id, auto_commit=True, auto_refresh=True
        )
        return profile


async def provide_profile_service(
    db_session: AsyncSession,
) -> AsyncGenerator[ProfileService, None]:
    async with ProfileService.new(session=db_session) as service:
        yield service
