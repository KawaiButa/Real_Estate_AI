import random
import uuid
from faker import Faker
from sqlalchemy import select
from datetime import datetime, timezone
from domains.properties.service import VIETNAM_PROPERTY_CATEGORIES
from database.models.user import User
from database.models.user_search import UserSearch  
from seed.factories.base import BaseFactory
from domains.user_search.service import UserSearchRepository  # Adjust import path
from configs.sqlalchemy import sqlalchemy_config

fake = Faker()

class UserSearchFactory(BaseFactory):
    repository = UserSearchRepository

    async def seed(self, count: int) -> None:
        async with sqlalchemy_config.get_session() as session:
            try:
                # Get valid user IDs
                user_ids = list(
                    (await session.execute(select(User.id))).scalars().all()
                )
                if not user_ids:
                    raise Exception("No users found to assign user searches.")

                property_types = VIETNAM_PROPERTY_CATEGORIES

                for _ in range(count):
                    min_price = random.randint(100000, 1000000) * 10
                    max_price = min_price + random.randint(1000000, 10000000)

                    user_search = UserSearch(
                        id=str(uuid.uuid4()),
                        user_id=random.choice(user_ids),
                        search_query=fake.sentence(nb_words=10),
                        type=random.choice(property_types),
                        min_price=min_price,
                        max_price=max_price,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    )
                    await self.repository(session=session).add(user_search)

            except Exception as e:
                await session.rollback()
                print(f"Error during user search seeding: {e}")
                raise
            finally:
                await session.commit()

    async def drop_all(self) -> None:
        async with sqlalchemy_config.get_session() as session:
            await self.repository(session=session).delete_where(
                UserSearch.id.is_not(None)
            )
            await session.commit()
