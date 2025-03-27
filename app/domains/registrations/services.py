from collections.abc import AsyncGenerator
import uuid
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from domains.registrations.dtos import RegisterWriteModel
from domains.supabase.service import SupabaseService
from database.models.role import Role
from database.models.user import User
from database.models.partner_registration import PartnerRegistration, PartnerType
from sqlalchemy.ext.asyncio import AsyncSession
from litestar.exceptions import ValidationException
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from advanced_alchemy.utils.text import slugify
from sqlalchemy.orm import selectinload


class PartnerRegistrationRepository(SQLAlchemyAsyncRepository[PartnerRegistration]):
    model_type = PartnerRegistration


class PartnerRegistrationService(SQLAlchemyAsyncRepositoryService[PartnerRegistration]):
    repository_type = PartnerRegistrationRepository

    async def get_registration_by_user_id(
        self, user_id: uuid.UUID
    ) -> PartnerRegistration:
        registration = await self.get_one_or_none(
            PartnerRegistration.user_id.__eq__(user_id)
        )
        if not registration:
            raise ValidationException(
                f"The registration of user {user_id} cannot be found"
            )
        return registration

    async def update_registration(
        self,
        data: RegisterWriteModel,
        user_id: uuid.UUID | None,
        supabase_service: SupabaseService,
        is_admin: bool = False,
    ) -> PartnerRegistration:
        partner_registration = await self.get_one_or_none(
            PartnerRegistration.user_id.__eq__(user_id)
        )
        if not partner_registration:
            raise ValidationException(f"No registration found with user id {user_id}")
        if not is_admin and partner_registration.approved is not None:
            raise ValidationException(
                "This registration has already been validated and cannot be changed"
            )
        profile_filename = f"{user_id}"
        data.profile_url = await supabase_service.upload_file(
            data.profile_img,
            bucket_name="profile",
            name=profile_filename,
        )
        if data.type == PartnerType.INDIVIDUAL:
            data.authorized_representative_name = None
            data.business_registration_certificate_url = None
            data.tax_id = None
        if data.business_registration_certificate_img:
            data.business_registration_certificate_url = (
                await supabase_service.upload_file(
                    data.business_registration_certificate_img,
                    name=profile_filename,
                    bucket_name="business_registration",
                )
            )
        partner_registration = await self.update(
            data,
            item_id=partner_registration.id,
            auto_commit=True,
            auto_refresh=True,
        )
        return partner_registration

    async def approve_registration(self, user_id: uuid.UUID):
        partner_registration = await self.get_one_or_none(
            PartnerRegistration.user_id.__eq__(user_id)
        )
        if not partner_registration:
            raise ValidationException(
                f"No partner registration found with user id {user_id}"
            )
        partner_registration.approved = True
        partner_registration.user.roles.extend(
            [
                await Role.as_unique_async(
                    self.repository.session,
                    name="partner",
                    slug=slugify("partner"),
                )
            ]
        )
        partner_registration = await self.update(
            item_id=partner_registration.id,
            data=partner_registration,
            auto_commit=True,
            auto_refresh=True,
        )
        return partner_registration

    async def reject_registration(self, user_id: uuid.UUID):
        partner_registration = await self.get_one_or_none(
            PartnerRegistration.user_id.__eq__(user_id),
        )
        if not partner_registration:
            raise ValidationException(
                f"No partner registration found with user id {user_id}"
            )
        partner_registration.approved = False
        partner_registration = await self.update(
            item_id=partner_registration.id,
            data=partner_registration,
            auto_commit=True,
            auto_refresh=True,
        )
        return partner_registration


async def provide_property_service(
    db_session: AsyncSession,
) -> AsyncGenerator[PartnerRegistrationService, None]:
    
    async with PartnerRegistrationService.new(session=db_session) as service:
        yield service
