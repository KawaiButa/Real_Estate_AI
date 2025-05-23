from datetime import datetime, timezone
import random
import uuid
from faker import Faker
from sqlalchemy import select
from database.models.review import Review
from database.models.user import User
from database.models.property import Property
from seed.factories.base import BaseFactory
from domains.review.service import RatingRepository  # adjust if your repo is elsewhere
from configs.sqlalchemy import sqlalchemy_config
from advanced_alchemy.utils.fixtures import open_fixture

fake = Faker()


class ReviewFactory(BaseFactory):
    repository = RatingRepository

    async def seed(self, count: int, fixture_path: str = "seed/fixtures") -> None:
        async with sqlalchemy_config.get_session() as session:
            try:
                # Get valid property IDs
                property_ids = (
                    (await session.execute(select(Property.id))).scalars().all()
                )
                if not property_ids:
                    raise Exception("No properties found to assign reviews.")
                user_ids = list(
                    (await session.execute(select(User.id))).scalars().all()
                )

                if fixture_path:
                    review_data = open_fixture(fixture_path, "reviews")
                    reviews = []
                    for item in review_data:
                        property_id = uuid.UUID(int=int(item["listing_id"]))
                        user_id = uuid.UUID(int=int(item["reviewer_id"]))
                        if property_id not in property_ids or user_id not in user_ids:
                            continue
                        reviews.append(
                            Review(
                                **{
                                    "id": uuid.UUID(int=int(item["id"])),
                                    "reviewer_id": user_id,
                                    "review_text": item["comments"],
                                    "created_at": datetime.strptime(
                                        item["date"], "%Y-%m-%d"
                                    ),
                                    "rating": random.randint(1, 5),
                                    "property_id": property_id,
                                }
                            )
                        )
                    await self.repository(session=session).add_many(reviews)
                    return
                for _ in range(count):
                    review = Review(
                        id=str(uuid.uuid4()),
                        property_id=random.choice(property_ids),
                        reviewer_id=random.choice(user_ids),
                        rating=random.randint(1, 5),
                        review_text=fake.paragraph(nb_sentences=3),
                        helpful_count=random.randint(0, 50),
                        featured=fake.boolean(),
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    )
                    await self.repository(session=session).add(review)
            except Exception as e:
                await session.rollback()
                print(f"Error during review seeding: {e}")
                raise
            finally:
                await session.commit()

    async def drop_all(self) -> None:
        async with sqlalchemy_config.get_session() as session:
            await self.repository(session=session).delete_where(Review.id.is_not(None))
            await session.commit()
