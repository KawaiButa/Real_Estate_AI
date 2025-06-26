from typing import Annotated, Any
import uuid
from litestar import Controller, Request, Response, get, patch, post
from litestar.di import Provide
from database.utils import provide_pagination_params
from domains.auth.dtos import LoginReturnSchema
from database.models.property import Property, PropertySchema
from domains.properties.service import PropertyService, provide_property_service
from database.models.user import User, UserSchema
from domains.auth.guard import GuardRole, role_guard
from domains.profile.dto import (
    ProfileReturnDTO,
    UpdateUserDTO,
    UpdateUserSchema,
)
from domains.profile.service import ProfileService, provide_profile_service
from litestar.security.jwt import Token
from litestar.exceptions import ValidationException
from litestar.params import Body
from litestar.enums import RequestEncodingType
from advanced_alchemy.filters import LimitOffset
from litestar.pagination import OffsetPagination


class ProfileController(Controller):
    path = "profile"
    tags = ["Profile"]
    dependencies = {
        "profile_service": Provide(provide_profile_service),
        "property_service": Provide(provide_property_service),
    }

    @get(
        "/",
        return_dto=ProfileReturnDTO,
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

    @patch("/", dto=UpdateUserDTO, return_dto=None)
    async def update_profile(
        self,
        data: Annotated[
            UpdateUserSchema, Body(media_type=RequestEncodingType.MULTI_PART)
        ],
        profile_service: ProfileService,
        request: Request[User, Token, Any],
    ) -> UserSchema:
        return await profile_service.update_profile(data=data, user_id=request.user.id)

    @patch(
        "/{user_id: uuid}",
        dto=UpdateUserDTO,
        return_dto=None,
        guards=[role_guard([GuardRole.ADMIN])],
        opt={"admin": GuardRole.ADMIN},
    )
    async def update_profile_admin(
        self,
        user_id: uuid.UUID,
        data: Annotated[
            UpdateUserSchema, Body(media_type=RequestEncodingType.MULTI_PART)
        ],
        profile_service: ProfileService,
    ) -> UserSchema:
        return await profile_service.update_profile(data=data, user_id=user_id)

    @get(
        "/favorite",
        dependencies={
            "pagination": Provide(provide_pagination_params),
        },
    )
    async def get_favorites(
        self,
        profile_service: ProfileService,
        pagination: LimitOffset,
        request: Request[User, Token, Any],
    ) -> OffsetPagination[PropertySchema]:
        return await profile_service.get_favorites(request.user.id, pagination)

    @post("/favorite/{property_id: uuid}")
    async def toggle_favorite(
        self,
        property_id: uuid.UUID,
        profile_service: ProfileService,
        property_service: PropertyService,
        request: Request[User, Token, Any],
    ) -> Response[Any]:
        property = await property_service.get_one_or_none(
            Property.id.__eq__(property_id)
        )
        if not property:
            raise ValidationException(f"No property found with id {property_id}")
        await profile_service.toggle_favorite(
            user_id=request.user.id, property=property
        )
        return Response(
            content={"message": "Update favorite successfully"}, status_code=200
        )

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
    ) -> Response[Any]:
        property = await property_service.get_one_or_none(
            Property.id.__eq__(property_id)
        )
        if not property:
            raise ValidationException(f"No property found with id {property_id}")
        await profile_service.toggle_favorite(user_id=user_id, property=property)
        return Response(content="Update favorite successfully", status_code=200)

    @get("/refresh-token")
    async def refresh_token(
        self, profile_service: ProfileService, request: Request[User, Token, Any]
    ) -> LoginReturnSchema:
        return await profile_service.refresh_token(user_id=request.user.id)
