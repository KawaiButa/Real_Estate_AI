from pathlib import Path
from litestar.exceptions import ValidationException
from litestar import Litestar, MediaType, Request, Response, get
from litestar.types import ControllerRouterHandler
from litestar.plugins.sqlalchemy import SQLAlchemyPlugin
from domains.admin.controller import AdminController
from domains.profile.controller import ProfileController
from domains.registrations.controller import PartnerRegistrationController
from configs.sqlalchemy import provide_transaction, sqlalchemy_config
from domains.auth.controller import AuthController
from dotenv import load_dotenv
from configs import openapi
from domains.properties.controller import PropertyController
from security.oauth2 import oauth2_auth
from litestar.di import Provide
from litestar.exceptions.responses import create_exception_response
from litestar.plugins.structlog import StructlogPlugin, StructlogConfig
from litestar.plugins.prometheus import PrometheusConfig, PrometheusController
from litestar.static_files import create_static_files_router
from configs.template import template_config

load_dotenv()


def validation_exception_handler(
    request: Request, exc: ValidationException
) -> Response:
    if isinstance(exc.extra, list) and len(exc.extra) > 0 and exc.extra[0]["input"]:
        content = {
            "status_code": 400,
            "Message": "Bad request",
            "details": [
                {"type": error["type"], "field": error["loc"], "msg": error["msg"]}
                for error in exc.extra
            ],
        }
        return Response(
            media_type=MediaType.JSON,
            content=content,
            status_code=400,
        )
    else:
        return create_exception_response(request=request, exc=exc)


structlog_plugin = StructlogPlugin(config=StructlogConfig())
# ASSETS_DIR
ASSETS_DIR = Path("assets")


def on_startup():
    ASSETS_DIR.mkdir(exist_ok=True)


@get(path="/schema")
async def schema(request: Request) -> dict:
    schema = request.app.openapi_schema
    return schema.to_schema()


routes: list[ControllerRouterHandler] = [
    AuthController,
    PropertyController,
    PartnerRegistrationController,
    ProfileController,
    PrometheusController,
    AdminController,
    schema,
    create_static_files_router(
        path="/",
        directories=["assets"],
        opt={"some": True},
        include_in_schema=False,
        tags=["static"],
    ),
]
prometheus_config = PrometheusConfig(group_path=False)
app = Litestar(
    route_handlers=routes,
    openapi_config=openapi.config,
    dependencies={"transaction": Provide(provide_transaction, sync_to_thread=True)},
    on_app_init=[oauth2_auth.on_app_init],
    debug=True,
    exception_handlers={
        ValidationException: validation_exception_handler,
    },
    template_config=template_config,
    plugins=[SQLAlchemyPlugin(config=sqlalchemy_config)],
)
