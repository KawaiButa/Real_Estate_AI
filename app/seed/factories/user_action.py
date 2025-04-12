import random
import uuid
from datetime import datetime, timezone
from sqlalchemy import select
from faker import Faker

from database.models.user import User
from database.models.property import Property
from database.models.user_action import UserAction  # Adjust import if needed
from seed.factories.base import BaseFactory
from domains.user_action.service import (
    UserActionRepository,
)  # Adjust if your repo is elsewhere
from configs.sqlalchemy import sqlalchemy_config

fake = Faker()


class UserActionFactory(BaseFactory):
    repository = UserActionRepository  # Adjust if your repository is elsewhere

    async def seed(self, count: int) -> None:
        async with sqlalchemy_config.get_session() as session:
            try:
                # Get valid property IDs
                property_ids = (
                    (await session.execute(select(Property.id))).scalars().all()
                )
                if not property_ids:
                    raise Exception("No properties found to assign user actions.")

                # Get valid user IDs
                user_ids = list(
                    (await session.execute(select(User.id))).scalars().all()
                )
                if not user_ids:
                    raise Exception("No users found to assign user actions.")

                # Define possible actions
                actions = ["view", "like", "unlike"]

                for _ in range(count):
                    user_action = UserAction(
                        id=str(
                            uuid.uuid4()
                        ), 
                        user_id=random.choice(user_ids),
                        property_id=random.choice(property_ids),
                        action=random.choice(actions),
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    )
                    await self.repository(session=session).add(user_action)
            except Exception as e:
                await session.rollback()
                print(f"Error during user action seeding: {e}")
                raise
            finally:
                await session.commit()

    async def drop_all(self) -> None:
        async with sqlalchemy_config.get_session() as session:
            await self.repository(session=session).delete_where(
                UserAction.id.is_not(None)
            )
            await session.commit()
