
from sqlalchemy.ext.asyncio import AsyncSession
from collections.abc import AsyncGenerator
from domains.auth.service import AuthService

async def provide_auth_service(
    db_session: AsyncSession,
) -> AsyncGenerator["AuthService", None]:
    async with AuthService.new(session=db_session) as service:
        yield service
