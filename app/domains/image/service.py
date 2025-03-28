from collections.abc import AsyncGenerator
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from database.models.image import Image
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService

from sqlalchemy.ext.asyncio import AsyncSession


class ImageRepository(SQLAlchemyAsyncRepository[Image]):
    model_type = Image


class ImageService(SQLAlchemyAsyncRepositoryService[Image]):
    repository_type = ImageRepository


async def provide_image_service(
    db_session: AsyncSession,
) -> AsyncGenerator[ImageService]:
    async with ImageService.new(session=db_session) as service:
        yield service
