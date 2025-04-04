from contextlib import asynccontextmanager
import os
from litestar.plugins.sqlalchemy import AsyncSessionConfig, SQLAlchemyAsyncConfig
from litestar.datastructures import State
from collections.abc import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.exc import IntegrityError
from litestar.exceptions import ClientException
from litestar.status_codes import HTTP_409_CONFLICT
from dotenv import load_dotenv
load_dotenv()
session_config = AsyncSessionConfig(expire_on_commit=False)
sqlalchemy_config = SQLAlchemyAsyncConfig(
    connection_string=os.environ.get("DB_URL"),
    session_config=session_config,
    create_all=False,
)  # Create 'async_session' dependency.

sessionmaker = async_sessionmaker(expire_on_commit=False)

@asynccontextmanager
async def provide_transaction(state: State) -> AsyncGenerator[AsyncSession, None]:
    async with sessionmaker(bind=getattr(state, "engine", None)) as session:
        try:
            async with session.begin():
                yield session
        except IntegrityError as exc:
            raise ClientException(
                status_code=HTTP_409_CONFLICT,
                detail=str(exc),
            ) from exc
