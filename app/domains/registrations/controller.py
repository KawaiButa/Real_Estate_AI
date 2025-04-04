# The `PartnerRegistrationController` class defines endpoints for managing partner registrations with
# role-based access control in a web application.
from datetime import datetime
from typing import Annotated, Any
import uuid
from litestar.exceptions import ValidationException
from litestar import Request, get, patch, post
from domains.supabase.service import SupabaseService, provide_supabase_service
from database.models.user import User
from domains.registrations.dtos import RegisterWriteDTO, RegisterWriteModel
from domains.registrations.services import (
    PartnerRegistrationService,
    provide_property_service,
)
from litestar.di import Provide
from database.models.partner_registration import PartnerRegistration, PartnerType
from domains.auth.guard import role_guard, GuardRole
from litestar.controller import Controller
from litestar.security.jwt import Token
from litestar.params import Body
from litestar.enums import RequestEncodingType
from litestar.exceptions import PermissionDeniedException
from litestar.dto import DTOData
from litestar.plugins.sqlalchemy import SQLAlchemyDTO


class PartnerRegistrationController(Controller):
    tags = ["Partner Registration"]
    path = "partner/registration"

    dependencies = {
        "partner_registration_service": Provide(provide_property_service),
    }

    @get(
        "/",
    )
    async def get_my_partner_registration(
        self,
        partner_registration_service: PartnerRegistrationService,
        request: Request[User, Token, Any],
    ) -> PartnerRegistration:
        return await partner_registration_service.get_registration_by_user_id(
            user_id=request.user.id
        )

    @get(
        "/users/{user_id: uuid}",
        guards=[role_guard([GuardRole.ADMIN])],
        opt={"admin": GuardRole.ADMIN},
    )
    async def get_user_partner_registration(
        self,
        user_id: uuid.UUID,
        partner_registration_service: PartnerRegistrationService,
    ) -> PartnerRegistration:
        return await partner_registration_service.get_registration_by_user_id(
            user_id=user_id
        )

    @post(
        "/",
        guards=[role_guard([GuardRole.CUSTOMER])],
        opt={"customer": GuardRole.CUSTOMER},
    )
    async def register_partner(
        self,
        data: Annotated[
            RegisterWriteModel, Body(media_type=RequestEncodingType.MULTI_PART)
        ],
        partner_registration_service: PartnerRegistrationService,
        request: Request[User, Token, Any],
    ) -> PartnerRegistration:
        if not data.profile_img and request.user.image_id:
            raise ValidationException("You have to set profile image or provide image")
        return await partner_registration_service.register(
            data=data, user_id=request.user.id
        )

    @patch(
        "/",
        dto=RegisterWriteDTO,
        return_dto=SQLAlchemyDTO[PartnerRegistration],
        guards=[role_guard([GuardRole.CUSTOMER])],
    )
    async def update_my_registration(
        self,
        data: DTOData[RegisterWriteModel],
        user_id: uuid.UUID,
        partner_registration_service: PartnerRegistrationService,
    ) -> PartnerRegistration:
        partner_registration = await partner_registration_service.update_registration(
            data=data.create_instance(updated_at=datetime.utcnow()),
            user_id=user_id,
        )
        return partner_registration

    @patch(
        "/{user_id: uuid}",
        dto=RegisterWriteDTO,
        return_dto=SQLAlchemyDTO[PartnerRegistration],
        guards=[role_guard([GuardRole.ADMIN])],
    )
    async def update_registration(
        self,
        data: DTOData[RegisterWriteModel],
        user_id: uuid.UUID,
        supabase_service: SupabaseService,
        partner_registration_service: PartnerRegistrationService,
    ) -> PartnerRegistration:
        partner_registration = await partner_registration_service.update_registration(
            data=data.create_instance(updated_at=datetime.utcnow()),
            user_id=user_id,
            supabase_service=supabase_service,
            is_admin=True,
        )
        return partner_registration

    @post(
        "/{user_id: uuid}/approve",
        guards=[role_guard([GuardRole.ADMIN])],
    )
    async def approve_registration(
        self,
        user_id: uuid.UUID,
        partner_registration_service: PartnerRegistrationService,
    ) -> PartnerRegistration:
        return await partner_registration_service.approve_registration(user_id=user_id)

    @post(
        "/{user_id: uuid}/reject",
        guards=[role_guard([GuardRole.ADMIN])],
    )
    async def reject_registration(
        self,
        user_id: uuid.UUID,
        partner_registration_service: PartnerRegistrationService,
    ) -> PartnerRegistration:
        return await partner_registration_service.reject_registration(user_id=user_id)
