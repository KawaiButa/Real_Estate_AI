from datetime import datetime, timezone
import random
import uuid
from faker import Faker
from domains.address.service import AddressService
from database.models.address import Address
from database.models.role import Role
from domains.auth.repository import UserRepository
from seed.factories.base import BaseFactory
from database.models.user import User
from passlib.hash import bcrypt
from configs.sqlalchemy import sqlalchemy_config
from sqlalchemy import select
from advanced_alchemy.utils.fixtures import open_fixture

fake = Faker("vi_VN")


def generate_unique_email():
    unique_id = uuid.uuid4().hex
    return f"user_{unique_id}@example.com"


class UserFactory(BaseFactory):
    repository = UserRepository

    async def seed(self, count: int, fixture_path: str = "seed/fixtures") -> None:
        async with sqlalchemy_config.get_session() as session:
            try:
                customer = (
                    await session.execute(
                        select(Role).filter(Role.name == "customer"),
                    )
                ).scalar()
                partner = (
                    await session.execute(select(Role).filter(Role.name == "partner"))
                ).scalar()
                address_ids = (
                    (await session.execute(select(Address.id))).scalars().all()
                )
                hashed_password = bcrypt.hash("123456")
                if fixture_path:
                    user_data = open_fixture(fixture_path, "hosts")
                    user_data = [
                        {
                            "id": uuid.UUID(int=int(data["id"])),
                            "name": data["name"],
                            "created_at": (
                                datetime.strptime(data["since"], "%Y-%m-%d")
                                if len(data["since"]) > 0
                                else datetime.now(timezone.utc)
                            ),
                            "roles": [partner],
                            "address_id": random.choice(address_ids),
                            "verified": fake.boolean(),
                            "password": hashed_password,
                            "phone": fake.phone_number(),
                            "email": generate_unique_email(),
                            "reset_password_token": None,
                            "reset_password_expires": None,
                        }
                        for data in user_data
                    ]
                    await self.repository(session=session).add_many(
                        User(**data) for data in user_data
                    )
                    return
                for _ in range(count):
                    user = User(
                        id=str(uuid.uuid4()),
                        name=fake.name(),
                        email=generate_unique_email(),
                        phone=fake.phone_number(),
                        password=bcrypt.hash("123456"),
                        verified=fake.boolean(),
                        address_id=random.choice(address_ids),
                        image_id=None,
                        roles=(
                            [customer]
                            if random.choice([True, False])
                            else [customer, partner]
                        ),
                        reset_password_token=None,
                        reset_password_expires=None,
                        created_at=datetime.now(timezone.utc),
                        updated_at=datetime.now(timezone.utc),
                    )
                    await self.repository(session=session).add(user)
            except Exception as e:
                await session.rollback()
                print(f"Error during seeding: {e}")
                raise
            finally:
                await session.commit()

    async def drop_all(self) -> None:
        async with sqlalchemy_config.get_session() as session:
            await self.repository(session=session).delete_where(User.id.is_not(None))
            await session.commit()
