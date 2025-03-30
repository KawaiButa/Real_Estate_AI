from abc import ABC, abstractmethod
from typing import Type
from sqlalchemy.ext.asyncio import AsyncSession
from advanced_alchemy.repository import SQLAlchemyAsyncRepository


class BaseFactory(ABC):
    repository: Type[SQLAlchemyAsyncRepository]

    @abstractmethod
    async def seed(self, count: int) -> None:
        """Insert a specified number of rows into the table."""
        pass

    @abstractmethod
    async def drop_all(self) -> None:
        """Delete all rows from the table."""
        pass
