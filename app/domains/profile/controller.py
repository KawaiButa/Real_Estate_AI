from typing import Any
import uuid
from litestar import Controller, Request, get, patch, post
from litestar.di import Provide
from database.models.property import Property
from domains.properties.service import PropertyService, provide_property_service
from database.models.user import User, UserSchema
from domains.auth.guard import GuardRole, role_guard
from domains.profile.dto import ProfileUpdateDTO
from domains.profile.service import ProfileService, provide_profile_service
from litestar.security.jwt import Token
from litestar.dto import DTOData
from litestar.plugins.sqlalchemy import SQLAlchemyDTO
from litestar.exceptions import ValidationException


class ProfileController(Controller):
    path = "profile"
    tags = ["Profile"]
    dependencies = {
        "profile_service": Provide(provide_profile_service),
        "property_service": Provide(provide_property_service),
    }

    @get(
        "/",
    )
    async def get_profile(
        self, profile_service: ProfileService, request: Request[User, Token, Any]
    ) -> UserSchema:
        return await profile_service.get_profile(user_id=request.user.id)

    @get(
        "/{user_id: uuid}",
        guards=[role_guard([GuardRole.ADMIN])],
        opt={"admin": GuardRole.ADMIN},
    )
    async def get_user_profile(
        self, user_id: uuid.UUID, profile_service: ProfileService
    ) -> UserSchema:
        return await profile_service.get_profile(user_id=user_id)

    @patch("/")
    async def update_profile(
        self,
        data: DTOData[User],
        profile_service: ProfileService,
        request: Request[User, Token, Any],
    ) -> UserSchema:
        return await profile_service.update_profile(data=data, user_id=request.user.id)

    @patch(
        "/{user_id: uuid}",
        dto=ProfileUpdateDTO,
        return_dto=SQLAlchemyDTO[User],
        guards=[role_guard([GuardRole.ADMIN])],
        opt={"admin": GuardRole.ADMIN},
    )
    async def update_profile(
        self, user_id: uuid.UUID, data: DTOData[User], profile_service: ProfileService
    ) -> UserSchema:
        return await profile_service.update_profile(data=data, user_id=user_id)

    @post("/favorite/{property_id: uuid}")
    async def toggle_favorite(
        self,
        property_id: uuid.UUID,
        profile_service: ProfileService,
        property_service: PropertyService,
        request: Request[User, Token, Any],
    ) -> Any:
        property = await property_service.get_one_or_none(
            Property.id.__eq__(property_id)
        )
        if not property:
            raise ValidationException(f"No property found with id {property_id}")
        profile_service.toggle_favorite(user_id=request.user.id, property=property)

    @post(
        "/{user_id: uuid}/favorite/{property_id: uuid}",
        guards=[role_guard([GuardRole.ADMIN])],
    )
    async def admin_toggle_favorite(
        self,
        user_id: uuid.UUID,
        property_id: uuid.UUID,
        profile_service: ProfileService,
        property_service: PropertyService,
    ) -> Any:
        property = await property_service.get_one_or_none(
            Property.id.__eq__(property_id)
        )
        if not property:
            raise ValidationException(f"No property found with id {property_id}")
        profile_service.toggle_favorite(user_id=user_id, property=property)
