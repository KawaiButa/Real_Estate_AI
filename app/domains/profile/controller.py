from typing import Any
import uuid
from litestar import Controller, Request, get, patch
from litestar.di import Provide
from database.models.user import User
from domains.auth.guard import GuardRole, role_guard
from domains.profile.dto import ProfileUpdateDTO
from domains.profile.service import ProfileService, provide_profile_service
from litestar.security.jwt import Token
from litestar.dto import DTOData
from litestar.plugins.sqlalchemy import SQLAlchemyDTO


class ProfileController(Controller):
    path = "profile"
    tags = ["Profile"]
    dependencies = {"profile_service": Provide(provide_profile_service)}

    @get(
        "/",
    )
    async def get_profile(
        self, profile_service: ProfileService, request: Request[User, Token, Any]
    ) -> User:
        return await profile_service.get_profile(user_id=request.user.id)

    @get(
        "/{user_id: uuid}",
        guards=[role_guard([GuardRole.ADMIN])],
        opt={"admin": GuardRole.ADMIN},
    )
    async def get_user_profile(
        self, user_id: uuid.UUID, profile_service: ProfileService
    ) -> User:
        return await profile_service.get_profile(user_id=user_id)

    @patch("/")
    async def update_profile(
        self,
        data: DTOData[User],
        profile_service: ProfileService,
        request: Request[User, Token, Any],
    ) -> User:
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
    ) -> User:
        return await profile_service.update_profile(data=data, user_id=user_id)
