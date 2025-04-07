import uuid
from collections.abc import AsyncGenerator
from domains.image.service import ImageService
from domains.auth.dtos import LoginReturnSchema
from domains.supabase.service import SupabaseService, provide_supabase_service
from domains.profile.dto import UpdateUserSchema
from database.models.property import Property
from database.models.user import User, UserSchema
from domains.auth.repository import UserRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from sqlalchemy.ext.asyncio import AsyncSession
from litestar.exceptions import ValidationException
from sqlalchemy.orm import selectinload, noload
from security.oauth2 import oauth2_auth


class ProfileService(SQLAlchemyAsyncRepositoryService[User]):
    repository_type = UserRepository
    supabase_service: SupabaseService = provide_supabase_service(bucket_name="profile")

    async def get_profile(self, user_id: uuid.UUID) -> UserSchema:
        profile = await self.get_one_or_none(
            User.id.__eq__(user_id),
            load=[
                selectinload(User.properties).noload(Property.owner),
                selectinload(User.favorites).noload(Property.owner),
            ],
            uniquify=True,
        )
        if not profile:
            raise ValidationException(f"Cannot find user with id {user_id}")
        return self.to_schema(profile, schema_type=UserSchema)

    async def update_profile(
        self,
        data: UpdateUserSchema,
        user_id: uuid.UUID,
    ) -> UserSchema:
        profile = await self.get_one_or_none(
            User.id == user_id
        )

        if not profile:
            raise ValidationException(f"Cannot find user with id {user_id}")
        update_data = data.model_dump(exclude={"profile_image"}, exclude_none=True)
        if data.profile_image:
            if profile.profile_image:
                await self.supabase_service.update_image(
                    profile.profile_image, data.profile_image
                )
            else:
                url = await self.supabase_service.upload_file(
                    data.profile_image, name=f"{profile.id}"
                )
                image_service = ImageService(session=self.repository.session)
                update_data["image_id"] = (
                    await image_service.create(
                        data={"url": url, "model_type": None, "model_id": None},
                        auto_refresh=True,
                    )
                ).id
        updated_profile = await self.update(
            update_data,
            item_id=profile.id,
            auto_commit=True,
            auto_refresh=True,
        )

        return self.to_schema(updated_profile, schema_type=UserSchema)

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

    async def refresh_token(self, user_id: uuid) -> LoginReturnSchema:
        user = await self.get_one_or_none(User.id == user_id)
        if not user:
            raise ValidationException("Invalid credential")
        return LoginReturnSchema(
            token=oauth2_auth.create_token(
                identifier=str(
                    {
                        "id": str(user.id),
                        "name": user.name,
                        "roles": [
                            {
                                "id": str(user.id),
                                "name": role.name,
                            }
                            for role in user.roles
                        ],
                    }
                ),
            ),
            user=self.to_schema(data=user, schema_type=UserSchema),
        )


async def provide_profile_service(
    db_session: AsyncSession,
) -> AsyncGenerator[ProfileService, None]:
    async with ProfileService.new(session=db_session) as service:
        yield service
