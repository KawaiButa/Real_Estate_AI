from collections.abc import AsyncGenerator
from database.models.property_type import PropertyType
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from sqlalchemy.ext.asyncio import AsyncSession


class PropertyTypeRepository(SQLAlchemyAsyncRepository[PropertyType]):
    model_type = PropertyType


class PropertyTypeService(SQLAlchemyAsyncRepositoryService[PropertyType]):
    repository_type = PropertyTypeRepository


async def provide_property_type_service(
    db_session: AsyncSession,
) -> AsyncGenerator[PropertyTypeService, None]:
    async with PropertyTypeService.new(session=db_session) as service:
        yield service
