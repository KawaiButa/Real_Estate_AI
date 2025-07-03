import os
from pathlib import Path
from litestar.exceptions import ValidationException
from litestar import Litestar, MediaType, Request, Response, get
from litestar.types import ControllerRouterHandler
from litestar.plugins.sqlalchemy import SQLAlchemyPlugin
from domains.tourview.controller import TourviewController
from domains.banner.controller import BannerController
from domains.review.controller import RatingController
from domains.property_verification.controller import VerificationController
from domains.property_types.controller import PropertyTypeController
from domains.user_action.controller import UserActionController
from seed.factories.image import ImageFactory
from seed.factories.user_action import UserActionFactory
from seed.factories.partner_registration import PartnerRegistrationFactory
from seed.factories.property import PropertyFactory
from seed.factories.address import AddressFactory
from seed.factories.user_search import UserSearchFactory
from seed.seed import Seeder
from domains.news.controller import ArticleController
from domains.chat_message.controller import ChatMessageController
from domains.chat_session.controller import ChatSessionController
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
from seed.factories.user import UserFactory
from seed.factories.article import ArticleFactory
from seed.factories.review import ReviewFactory
from seed.factories.banner import BannerFactory

load_dotenv()


def validation_exception_handler(
    request: Request, exc: ValidationException
) -> Response:
    if isinstance(exc.extra, list) and len(exc.extra) > 0 and "input" in exc.extra[0]:
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


@get(path="/schema", include_in_schema=False)
async def schema(request: Request) -> dict:
    schema = request.openapi_schema
    return schema.to_schema()


@get("/", opt={"no_auth": True}, include_in_schema=False)
async def helloWorld(request: Request) -> str:
    return "Hello world"


routes: list[ControllerRouterHandler] = [
    AuthController,
    ProfileController,
    PartnerRegistrationController,
    PropertyController,
    RatingController,
    BannerController,
    ArticleController,
    PrometheusController,
    AdminController,
    PropertyTypeController,
    VerificationController,
    UserActionController,
    ChatMessageController,
    ChatSessionController,
    TourviewController,
    schema,
    helloWorld,
    create_static_files_router(
        path="/",
        directories=["assets"],
        opt={"some": True},
        include_in_schema=False,
        tags=["static"],
    ),
]
prometheus_config = PrometheusConfig(group_path=False)
structlog_plugin = StructlogPlugin()
seeder = Seeder()


async def on_startUp() -> None:
    await seeder.seed_all(
        factory_classes=[
            # (AddressFactory, 1000),
            # (UserFactory, 20),
            # (PartnerRegistrationFactory, 20),
            # (PropertyFactory, 1000),
            # (ImageFactory, None),
            # (ArticleFactory, 200),
            # (ReviewFactory, 10000),
            # (BannerFactory, 10)
            # (UserActionFactory, 10000),
            # (UserSearchFactory, 10000),
        ]
    )
    return


app = Litestar(
    route_handlers=routes,
    openapi_config=openapi.config,
    dependencies={"transaction": Provide(provide_transaction, sync_to_thread=True)},
    on_app_init=[oauth2_auth.on_app_init],
    on_startup=[on_startUp],
    debug=os.environ.get("ENVIRONMENT") == "dev",
    exception_handlers={
        ValidationException: validation_exception_handler,
    },
    template_config=template_config,
    plugins=[SQLAlchemyPlugin(config=sqlalchemy_config)],
)
