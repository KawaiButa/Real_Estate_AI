"""OpenAPI Config."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from litestar.openapi.config import OpenAPIConfig
from litestar.openapi.plugins import (
    RedocRenderPlugin,
    ScalarRenderPlugin,
    StoplightRenderPlugin,
    SwaggerRenderPlugin,
)
from litestar.openapi.spec import Contact

from __metadata__ import __project__ as project
from __metadata__ import __version__ as version

__all__ = ["config"]
load_dotenv()

config = OpenAPIConfig(
    title=os.getenv("OPENAPI_TITLE", project),
    description=os.getenv(
        "OPENAPI_DESCRIPTION",
        "Litestar template for Railway",
    ),
    # servers=os.getenv("BASE_URL", "http://localhost:8000"),
    external_docs=os.getenv(  # type: ignore[arg-type]
        "OPENAPI_EXTERNAL_DOCS", "https://github.com/JacobCoffee/litestar-template/docs/"  # type: ignore[arg-type]
    ),
    version=version,
    contact=Contact(
        name=os.getenv("OPENAPI_CONTACT_NAME", "Administrator"),
        email=os.getenv("OPENAPI_CONTACT_EMAIL", "nh151700@gmail.com"),
    ),
    use_handler_docstrings=True,
    path=os.getenv("OPENAPI_PATH", "/api"),
    render_plugins=[
        SwaggerRenderPlugin(
            init_oauth={
                "clientId": "your-client-id",
                "appName": "your-app-name",
                "scopeSeparator": " ",
                "scopes": "openid profile",
                "useBasicAuthenticationWithAccessCodeGrant": True,
                "usePkceWithAuthorizationCodeGrant": True,
            }
        )
    ],
)
"""OpenAPI config for  """
