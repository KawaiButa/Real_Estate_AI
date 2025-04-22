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

# Import the Property model for use in the many-to-many relationship.
from database.models.property import Property

fake = Faker()


class UserSearchFactory(BaseFactory):
    repository = UserSearchRepository

    async def seed(self, count: int) -> None:
        async with sqlalchemy_config.get_session() as session:
            try:
                # Fetch valid user IDs
                user_ids = list(
                    (await session.execute(select(User.id))).scalars().all()
                )
                if not user_ids:
                    raise Exception("No users found to assign user searches.")

                for _ in range(count):
                    type = random.choice(VIETNAM_PROPERTY_CATEGORIES)
                    min_price = random.randint(100000, 1000000) * 10
                    max_price = min_price + random.randint(1000000, 10000000)
                    available_properties = list(
                        (
                            await session.execute(
                                select(Property).where(
                                    Property.property_category == type,
                                    Property.price >= min_price,
                                    Property.price <= max_price,
                                )
                            )
                        )
                        .scalars()
                        .all()
                    )
                    user_search = UserSearch(
                        id=str(uuid.uuid4()),
                        user_id=random.choice(user_ids),
                        search_query=fake.sentence(nb_words=10),
                        type=type,
                        min_price=min_price,
                        max_price=max_price,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    )

                    # Randomly sample a subset of properties (if available) and assign to the relationship.
                    if available_properties:
                        # Ensure that we only select up to the number of available properties.
                        sample_count = random.randint(
                            1, min(5, len(available_properties))
                        )
                        user_search.properties = random.sample(
                            available_properties, sample_count
                        )

                    # Add the user search entry along with its relationships.
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
