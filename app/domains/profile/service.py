import uuid
from collections.abc import AsyncGenerator
from database.models.property import Property
from database.models.user import User, UserSchema
from domains.auth.repository import UserRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from sqlalchemy.ext.asyncio import AsyncSession
from litestar.exceptions import ValidationException
from litestar.dto import DTOData


class ProfileService(SQLAlchemyAsyncRepositoryService[User]):
    repository_type = UserRepository

    async def get_profile(self, user_id: uuid.UUID) -> UserSchema:
        profile = await self.get_one_or_none(User.id.__eq__(user_id))
        if not profile:
            raise ValidationException(f"Cannot find user with id {user_id}")
        return self.to_schema(profile, schema_type=UserSchema)

    async def update_profile(
        self, data: DTOData[User], user_id: uuid.UUID
    ) -> UserSchema:
        profile = await self.get_one_or_none(User.id.__eq__(user_id))
        if not profile:
            raise ValidationException(f"Cannot find user with id {user_id}")
        profile = data.update_instance(profile)
        profile = await self.update(
            data=profile, item_id=profile.id, auto_commit=True, auto_refresh=True
        )
        return self.to_schema(profile, schema_type=UserSchema)

    async def toggle_favorite(self, user_id: uuid.UUID, property: Property) -> bool:
        user = await self.get_one_or_none(User.id.__eq__(user_id))
        if not user:
            raise ValidationException("No property found")
        try:
            user.favorites.remove(property)
        except ValueError as e:
            user.favorites.append(property)
        await self.update(data=user, item_id=user_id)
        return True

async def provide_profile_service(
    db_session: AsyncSession,
) -> AsyncGenerator[ProfileService, None]:
    async with ProfileService.new(session=db_session) as service:
        yield service
