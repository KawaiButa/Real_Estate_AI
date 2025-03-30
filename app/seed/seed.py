from typing import List, Tuple, Type
from sqlalchemy.ext.asyncio import AsyncSession
from seed.factories.base import BaseFactory


class Seeder:
    async def seed_all(
        self,
        factory_classes: List[Tuple[Type[BaseFactory], int]],
    ) -> None:
        """
        Seed all models using their respective factories.

        :param factory_classes: List of tuples where each tuple contains a factory class and the count of records to seed.
        """
        for FactoryClass, count in factory_classes:
            print(f"Seeding {count} {FactoryClass}")
            factory = FactoryClass()
            await factory.drop_all()
            await factory.seed(count)
            print(f"Seeding {FactoryClass} complete!")
