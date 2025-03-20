import ast
from os import environ
from typing import Any
from litestar.security.jwt import Token, JWTAuth
from litestar.connection import ASGIConnection
from database.models.user import User


async def retrieve_user_handler(
    token: Token,
    connection: ASGIConnection[Any, Any, Any, Any],
) -> User | None:
    return User(**ast.literal_eval(token.sub))


oauth2_auth = JWTAuth[User](
    retrieve_user_handler=retrieve_user_handler,
    token_secret=environ.get(
        "JWT_SECRET",
        "4141afedd0c6ee65ebf3a943dccf2639809778fbd72e68955bc5e67bb6a0ca44c1e742c288b980dd8e9d3a6acf3e0de79eda17229b7cb18d837e56a6be0e5ced3ecaa2914ca4e9cc013155071adee405a5ddb8029c592e5fd6f495d534e6fc59f3fafd48b124efa6a14e6e5a7c23c822b1b6045a3ded74d2039b2cd39f94994b5214c90e9a7be35349d86102be34e4092a5e13671de1be601f3a0632cb6af12d36e1f502c4a1ade7bacb24d4f0bfdf482d2c1eb41a96c3750210cbb9b86868816345b6f33364b3c1720c34c074613d1b5c4671cf598f672173006d26dafda82d8515047024fb52298ec926480b7e04a0861bb499a9af417c70ab9fcaf390ed7b",
    ),
    exclude=[
        "auth/",
        "schema",
        "api",
        "metrics",
        "admin/",
        "favicon.ico",
        "reset_password",
    ],
    exclude_opt_key="no_auth",
)
