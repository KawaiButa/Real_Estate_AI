from typing import Callable
from litestar.exceptions import NotAuthorizedException
from litestar.connection import ASGIConnection
from litestar.handlers.base import BaseRouteHandler
from sqlalchemy import Enum


class GuardRole(str, Enum):
    ADMIN = "admin"
    PARTNER = "partner"
    CUSTOMER = "customer"


def login_guard(connection: ASGIConnection, _: BaseRouteHandler) -> None:
    if not connection.user:
        raise NotAuthorizedException("User is not logged in.")


def role_guard(
    allowed_roles: list[GuardRole],
) -> Callable[[ASGIConnection, BaseRouteHandler], None]:
    def guard(connection: ASGIConnection, _: BaseRouteHandler) -> None:
        if len(allowed_roles) == 0:
            return
        if not connection.user:
            raise NotAuthorizedException("User is not logged in.")
        print(connection.user.roles)
        if not any(
            role['name'] in [r.lower() for r in allowed_roles] for role in connection.user.roles
        ):
            raise NotAuthorizedException("You are not authorized to use this feature")
        
    return guard


def provide_role_guard(
    roles: list[GuardRole],
) -> Callable[[ASGIConnection, BaseRouteHandler], None]:
    return role_guard(roles)
