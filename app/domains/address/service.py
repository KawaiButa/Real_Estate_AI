from collections.abc import AsyncGenerator
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from database.models.address import Address
from database.models.image import Image
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService

from sqlalchemy.ext.asyncio import AsyncSession


class AddressRepository(SQLAlchemyAsyncRepository[Address]):
    model_type = Address


class AddressService(SQLAlchemyAsyncRepositoryService[Address]):
    repository_type = AddressRepository
