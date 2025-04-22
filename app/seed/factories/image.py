import random
from faker import Faker
from sqlalchemy import select
from domains.auth.repository import UserRepository
from database.models.user import User
from seed.factories.base import BaseFactory
from database.models.image import Image
from database.models.partner_registration import PartnerRegistration
from database.models.property import Property
from configs.sqlalchemy import sqlalchemy_config
from domains.image.service import ImageRepository  # Adjust the import path if needed
from sqlalchemy.orm import noload
fake = Faker("vi_VN")


class ImageFactory(BaseFactory):
    repository = ImageRepository

    async def seed(self, count: int = None) -> None:
        async with sqlalchemy_config.get_session() as session:
            try:
                # Seed images for PartnerRegistration:
                partner_result = (await session.execute(select(PartnerRegistration))).unique()
                partner_regs = partner_result.scalars().all()

                for partner in partner_regs:
                    # Create a random number of images between 3 and 5
                    num_images = random.randint(3, 5)
                    for _ in range(num_images):
                        image = Image(
                            url=fake.image_url(),  # Generates a fake image URL
                            model_id=partner.id,
                            model_type="partner_registration",
                        )
                        await self.repository(session=session).add(image)

                # Seed images for Property:
                property_result = await session.execute(select(Property))
                properties = property_result.scalars().all()

                for prop in properties:
                    # Create a random number of images between 8 and 12
                    num_images = random.randint(8, 12)
                    for _ in range(num_images):
                        image = Image(
                            url=fake.image_url(),
                            model_id=prop.id,
                            model_type="property",
                        )
                        await self.repository(session=session).add(image)
                # Seed image for user
                user_repository = UserRepository(session=session)
                users = (await session.execute(select(User))).scalars().all()
                for user in users:
                    image = Image(
                        url=fake.image_url(),
                        model_id=None,
                        model_type=None,
                    )
                    image_id = image.id
                    image = await self.repository(session=session).add(
                        image, auto_refresh=True
                    )
                    user.image_id = image_id
                    await user_repository.update(user, attribute_names=["image_id"])
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.commit()

    async def drop_all(self) -> None:
        async with sqlalchemy_config.get_session() as session:
            await self.repository(session=session).delete_where(Image.id.is_not(None), load=[noload("*")])
            await session.commit()
