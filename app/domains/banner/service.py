from collections.abc import AsyncGenerator
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from database.models.banner import Banner
from sqlalchemy.ext.asyncio import AsyncSession


class BannerRepository(SQLAlchemyAsyncRepository[Banner]):
    model_type = Banner


class BannerService(SQLAlchemyAsyncRepositoryService[Banner]):
    repository_type = BannerRepository


async def provide_banner_service(
    db_session: AsyncSession,
) -> AsyncGenerator[BannerService, None]:
    async with BannerService.new(session=db_session) as service:
        yield service
