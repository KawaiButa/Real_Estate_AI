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
        "supabase_service": Provide(provide_supabase_service),
    }

    @get(
        "/me",
        guards=[role_guard([GuardRole.PARTNER])],
        opt={"partner": GuardRole.PARTNER},
    )
    async def get_my_partner_registration(
        self,
        partner_registration_service: PartnerRegistrationService,
        request: Request[User, Token, Any],
    ) -> PartnerRegistration:
        return await partner_registration_service.get_registration_by_user_id(
            user_id=request.user.id
        )

    @get("/users/{user_id: uuid}", opt={"admin": GuardRole.ADMIN})
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
        dto=RegisterWriteDTO,
        guards=[role_guard([GuardRole.CUSTOMER])],
        opt={"customer": GuardRole.CUSTOMER},
    )
    async def register_partner(
        self,
        data: Annotated[
            RegisterWriteModel, Body(media_type=RequestEncodingType.MULTI_PART)
        ],
        partner_registration_service: PartnerRegistrationService,
        supabase_service: SupabaseService,
        request: Request[User, Token, Any],
    ) -> PartnerRegistration:
        if "partner" in [role["name"] for role in request.user.roles]:
            raise PermissionDeniedException("You are a partner already")
        if (
            data.type == PartnerType.ENTERPRISE
            and not data.business_registration_certificate_img
        ):
            raise ValidationException(
                "Business Registration Certification Image is required with partner type Enterprise"
            )
        # Generate a unique filename or use provided one
        profile_filename = f"{request.user.id}"
        if data.type == PartnerType.INDIVIDUAL:
            data.authorized_representative_name = None
            data.business_registration_certificate_url = None
            data.tax_id = None
        data.profile_url = await supabase_service.upload_file(
            data.profile_img, bucket_name="profile", name=profile_filename
        )
        if data.business_registration_certificate_img:
            data.business_registration_certificate_url = (
                await supabase_service.upload_file(
                    data.business_registration_certificate_img,
                    name=profile_filename,
                    bucket_name="business_registration",
                )
            )
        data.user_id = request.user.id
        partner_registration = await partner_registration_service.create(data=data)
        return partner_registration

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
        supabase_service: SupabaseService,
        partner_registration_service: PartnerRegistrationService,
    ) -> PartnerRegistration:
        partner_registration = await partner_registration_service.update_registration(
            data=data.create_instance(updated_at=datetime.utcnow()),
            user_id=user_id,
            supabase_service=supabase_service,
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
