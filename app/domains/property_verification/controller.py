from typing import Any
from uuid import UUID
from litestar import Controller, MediaType, Request, get, post
from litestar.di import Provide
from litestar.security.jwt import Token
from database.models.property_verification import PropertyVerification
from database.models.user import User
from domains.property_verification.dtos import (
    VerificationConfirmDTO,
)
from litestar.exceptions import NotAuthorizedException
from domains.property_verification.service import (
    VerificationService,
    provide_verification_service,
)


class VerificationController(Controller):
    path = "/properties/{property_id:uuid}/verifications"
    dependencies = {
        "verification_service": Provide(provide_verification_service),
    }

    @get(
        "/qr-code",
        content_media_type="image/*",
        no_auth=True,
    )
    async def generate_qr_code(
        self,
        verification_service: VerificationService,
        property_id: UUID,
        request: Request[User, Token, Any]
    ) -> str:
        """Generate a QR code for property verification"""
        qr_code_data = await verification_service.generate_qr_code(property_id, user_id=request.user.id)
        return qr_code_data

    @get("/otp")
    async def generate_otp(
        self,
        verification_service: VerificationService,
        property_id: UUID,
        request: Request[User, Token, Any]
    ) -> dict:
        """Generate an OTP for property verification"""
        otp = await verification_service.generate_otp(property_id, user_id=request.user.id)
        return {"otp": otp}

    @post(
        "/verify",
    )
    async def verify_otp(
        self,
        verification_service: VerificationService,
        property_id: UUID,
        data: VerificationConfirmDTO,
        request: Request[User, Token, Any],
    ) -> PropertyVerification:
        """Verify an OTP"""
        return await verification_service.verify_code(
            property_id=property_id,
            user_id=request.user.id,
            validation_method=data.method,
            code=data.code,
        )
    
    @get('/allow')
    async def check_allow(self, property_id: UUID, verification_service: VerificationService, request: Request[User, Token, Any]) -> PropertyVerification:
        verification = await verification_service.get_one_or_none(PropertyVerification.user_id == request.user.id, PropertyVerification.property_id == property_id)
        if not verification:
            raise NotAuthorizedException("You are not allowed to review this property.")
        return verification