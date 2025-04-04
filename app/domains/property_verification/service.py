# File: src/app/domain/properties/services/verification.py
import base64
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta
from io import BytesIO
import os
import secrets
from uuid import UUID
from qrcode import QRCode
from database.models.property import Property
from domains.properties.service import PropertyService
from database.models.property_verification import PropertyVerification
from litestar.exceptions import NotAuthorizedException
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from litestar.stores.memory import MemoryStore
from litestar.exceptions import ValidationException
from sqlalchemy.ext.asyncio import AsyncSession

store = MemoryStore()


class VerificationRepository(SQLAlchemyAsyncRepository[PropertyVerification]):
    model_type = PropertyVerification


class VerificationService(SQLAlchemyAsyncRepositoryService[PropertyVerification]):
    repository_type = VerificationRepository

    async def generate_qr_code(self, property_id: UUID, user_id: UUID) -> str:
        """Generate a QR code for verification"""
        is_owner = await self._check_owner(property_id, user_id)
        qr_code_id = secrets.token_urlsafe(32)
        qr_code_data = f"{qr_code_id}"
        print("Code id:", qr_code_id)
        # Create a QR code image
        qr = QRCode(
            version=1,
            box_size=10,
            border=4,
        )
        qr.add_data(qr_code_data)
        qr.make(fit=True)

        img = qr.make_image(fill_color="black", back_color="white")
        img_io = BytesIO()
        img.save(img_io, "PNG", quality=70)
        img_io.seek(0)
        print("Property id: ", property_id)
        await store.set(str(property_id), qr_code_id, expires_in=600)
        return base64.b64encode(img_io.read()).decode()

    async def verify_code(
        self, property_id: UUID, user_id: UUID, code: str, validation_method: str
    ) -> PropertyVerification:
        qr_code_id = str(await store.get(str(property_id)))[2:-1]
        print("Code id:", qr_code_id)
        if code != qr_code_id:
            raise ValidationException("Wrong code or expired")
        store.delete(str(property_id))
        return await self.create(
            data={
                "user_id": user_id,
                "property_id": property_id,
                "verification_method": validation_method,
                "verification_code": qr_code_id,
            },
            auto_commit=True,
            auto_refresh=True,
        )

    async def generate_otp(self, property_id: UUID, user_id: UUID) -> str:
        otp = secrets.token_urlsafe(6)
        store.set(str(property_id), otp, expires_in=600)
        return otp

    async def _check_owner(self, property_id: UUID, user_id: UUID) -> bool:
        property_service = PropertyService(session=self.repository.session)
        property = await property_service.get_one_or_none(Property.id == property_id)
        if not property:
            ValidationException("No property found")
        return property.owner_id == user_id


async def provide_verification_service(
    db_session: AsyncSession,
) -> AsyncGenerator[VerificationService, None]:
    async with VerificationService.new(session=db_session) as service:
        yield service
