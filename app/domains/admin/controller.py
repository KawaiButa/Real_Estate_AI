from typing import Annotated
from litestar import Controller, get, post
from litestar.response import Template
from litestar.di import Provide
from database.models.user import User
from domains.auth.service import AuthService
from domains.auth.dtos import LoginDTO, LoginReturnSchema
from domains.auth.provider import provide_auth_service
from litestar.enums import RequestEncodingType
from litestar.params import Body
from litestar.dto import DTOData
from litestar.exceptions import NotAuthorizedException
class AdminController(Controller):
    path="admin"
    
    dependencies = {
        "auth_service": Provide(provide_auth_service)
    }
    @get("/login")
    async def admin_login_template(self) -> Template:
        return Template("admin/login.html")
    @post(
        "/login",
        dto=LoginDTO,
        return_dto=None,
        status_code=200,
        response_description="Login successfully",
    )
    async def admin_login(self, data: Annotated[DTOData[User], Body(media_type=RequestEncodingType.URL_ENCODED)], auth_service: AuthService
    ) -> LoginReturnSchema:
        response = await auth_service.login(userData=data)
        if "admin" not in [role.name for role in response.user.roles]:
            raise NotAuthorizedException()
        return response