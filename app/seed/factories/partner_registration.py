import uuid
import random
from datetime import date, datetime, timezone
from faker import Faker
from sqlalchemy import select
from seed.factories.base import BaseFactory
from database.models.user import User
from database.models.partner_registration import PartnerRegistration, PartnerType
from database.models.role import Role
from configs.sqlalchemy import sqlalchemy_config
from domains.registrations.services import (
    PartnerRegistrationRepository,
)  # Assume this exists

fake = Faker("vi_VN")


class PartnerRegistrationFactory(BaseFactory):
    repository = PartnerRegistrationRepository

    async def seed(self, count: int = None) -> None:
        async with sqlalchemy_config.get_session() as session:
            try:
                result = await session.execute(
                    select(User)
                    .join(User.roles)
                    .filter(Role.name == "partner", User.partner_registration.is_(None))
                )
                partner_users = result.scalars().all()
                if count and count < len(partner_users):
                    partner_users = partner_users[0:count]
                for user in partner_users:
                    p_type = (
                        PartnerType.INDIVIDUAL
                        if random.choice([True, False])
                        else PartnerType.ENTERPRISE
                    )

                    if p_type == PartnerType.ENTERPRISE:
                        tax_id = fake.numerify(text="########")
                        authorized_representative_name = fake.name()
                    else:
                        tax_id = None
                        authorized_representative_name = None
                    dob = (
                        fake.date_of_birth(tzinfo=None)
                        if p_type == PartnerType.INDIVIDUAL
                        else fake.date_between(start_date="-50y", end_date="-20y")
                    )

                    partner_registration = PartnerRegistration(
                        user_id=user.id,
                        type=p_type,
                        date_of_birth=dob,
                        tax_id=tax_id,
                        authorized_representative_name=authorized_representative_name,
                        approved=True,
                    )
                    await self.repository(session=session).add(partner_registration)
            except:
                await session.rollback()
            finally:
                await session.commit()

    async def drop_all(self) -> None:
        async with sqlalchemy_config.get_session() as session:
            await self.repository(session=session).delete_where(
                PartnerRegistration.id.is_not(None)
            )
            await session.commit()
