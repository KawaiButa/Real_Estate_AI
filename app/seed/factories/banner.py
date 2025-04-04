from datetime import datetime, timezone
import uuid
from faker import Faker
from sqlalchemy import select
from database.models.banner import Banner
from seed.factories.base import BaseFactory
from domains.banner.service import BannerRepository
from configs.sqlalchemy import sqlalchemy_config

fake = Faker()


class BannerFactory(BaseFactory):
    repository = BannerRepository

    async def seed(self, count: int) -> None:
        async with sqlalchemy_config.get_session() as session:
            try:
                for _ in range(count):
                    banner = Banner(
                        id=str(uuid.uuid4()),
                        title=fake.sentence(nb_words=4),
                        url=fake.image_url(),
                        content=fake.text(max_nb_chars=200),
                    )
                    await self.repository(session=session).add(banner)
            except Exception as e:
                await session.rollback()
                print(f"Error during banner seeding: {e}")
                raise
            finally:
                await session.commit()

    async def drop_all(self) -> None:
        async with sqlalchemy_config.get_session() as session:
            await self.repository(session=session).delete_where(Banner.id.is_not(None))
            await session.commit()
