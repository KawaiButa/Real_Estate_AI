from datetime import date
from typing import Annotated, Any
from litestar import Controller, get, post
from litestar.di import Provide
from litestar.dto import DTOData
from domains.email.service import MailService, provide_mail_service
from domains.auth.dtos import (
    ForgotPasswordSchema,
    LoginDTO,
    LoginReturnSchema,
    RegisterDTO,
    RegisterReturnDTO,
    ResetPasswordSchema,
)
from litestar.enums import RequestEncodingType
from domains.auth.service import AuthService, provide_auth_service
from database.models.user import User
from litestar.plugins.pydantic import PydanticDTO
from litestar.exceptions import InternalServerException
from litestar.response import Template
from litestar.params import Body


class AuthController(Controller):

    path = "/auth"
    tags = ["Authorization"]
    dependencies = {
        "auth_service": Provide(provide_auth_service),
        "mail_service": Provide(provide_mail_service),
    }

    @post(
        "/register",
        dto=RegisterDTO,
        return_dto=RegisterReturnDTO,
    )
    async def register(self, data: DTOData[User], auth_service: AuthService) -> Any:
        return await auth_service.register(userData=data)

    @post(
        "/login",
        dto=LoginDTO,
        return_dto=None,
        status_code=200,
        response_description="Login successfully",
    )
    async def login(
        self, data: DTOData[User], auth_service: AuthService
    ) -> LoginReturnSchema:
        return await auth_service.login(data)

    @post("/forgot-password", dto=PydanticDTO[ForgotPasswordSchema])
    async def forgot_password(
        self,
        data: ForgotPasswordSchema,
        auth_service: AuthService,
        mail_service: MailService,
    ) -> None:

        result = await auth_service.forgot_password(
            email=data.email,
            mail_service=mail_service,
        )
        if not result:
            raise InternalServerException()
        return

    @get("/reset-password", include_in_schema=False)
    async def get_reset_password_template(
        self,
    ) -> Template:
        return Template(
            template_name="reset_password.html",
            context={
                "reset_url": f"/auth/reset-password",
                "current_year": date.today().year,
            },
        )

    @post("/reset-password", dto=PydanticDTO[ResetPasswordSchema], status_code=200)
    async def reset_password(
        self,
        data: Annotated[
            ResetPasswordSchema, Body(media_type=RequestEncodingType.URL_ENCODED)
        ],
        auth_service: AuthService,
    ) -> str:
        user = await auth_service.reset_password(
            token=data.token, new_password=data.new_password
        )
        return "Reset password successfully"
