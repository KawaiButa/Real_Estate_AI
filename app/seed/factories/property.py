import random
from faker import Faker
from sqlalchemy import select
from database.models.role import Role
from domains.properties.service import (
    VIETNAM_PROPERTY_CATEGORIES,
    PropertyRepository,
    VIETNAM_TRANSACTION_TYPES,
)
from openai import OpenAI
from seed.factories.base import BaseFactory
from database.models.property import Property
from database.models.user import User
from database.models.address import Address
from configs.sqlalchemy import sqlalchemy_config

fake = Faker("vi_VN")
client = OpenAI()


def generate_title(property_category, transaction_type, city):
    """
    Generates a creative title for a property listing using ChatGPT.
    """
    prompt = (
        "You are a creative real estate marketer in Viet Nam. "
        "Generate a compelling and attractive title in Vietnamese for a property listing with the following details:\n"
        f"- Property Category: {property_category}\n"
        f"- Transaction Type: {transaction_type}\n"
        f"- City: {city}\n\n"
        "The title should be concise, engaging, and suitable for a high-end real estate listing."
    )

    try:
        response = client.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=50,
        )
        title = response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback in case of an error with the LLM call.
        title = f"{property_category} {transaction_type} tại {city}"
    return title


def generate_html_description(
    title, property_category, transaction_type, price, bedrooms, bathrooms, sqm
):
    """
    Generates an HTML description for a property by calling an LLM model.
    The LLM is instructed to act as a Master Marketer with extensive real estate experience.
    """
    prompt = (
        "You are a Master Marketer with extensive experience in Vietnamese real estate. "
        "Your task is to generate a detailed and enticing HTML description for a luxury real estate property in Vietnamese. "
        "Use creative and persuasive language that engages potential high-end buyers. "
        "The description should be formatted with proper HTML tags (e.g., <div>, <h2>, <p>) and include a vivid narrative highlighting the unique features, quality, and value proposition of the property. Also the HTML must include at least one image describing the real estate\n\n"
        "Here are the details of the property:\n"
        f"- Title: {title}\n"
        f"- Category: {property_category}\n"
        f"- Transaction Type: {transaction_type}\n"
        f"- Price: {price} USD\n"
        f"- Bedrooms: {bedrooms}\n"
        f"- Bathrooms: {bathrooms}\n"
        f"- Area: {sqm} m²\n\n"
        "Please generate a compelling, elegant, and well-structured HTML description that showcases this property as a top-tier real estate opportunity."
    )

    try:
        response = client.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=10000,
        )
        description = response.choices[0].message.content
    except Exception as e:
        # Fallback in case of an error with the LLM call.
        description = (
            f"<div class='property-description'>"
            f"<h2>{title}</h2>"
            f"<p>This {property_category.lower()} for {transaction_type.lower()} features "
            f"{bedrooms} bedrooms, {bathrooms} bathrooms, and spans {sqm} m². Priced at {price} USD.</p>"
            f"</div>"
        )
    return description


class PropertyFactory(BaseFactory):
    repository = PropertyRepository

    async def seed(self, count: int = 10) -> None:
        async with sqlalchemy_config.get_session() as session:
            try:
                user_ids = (
                    (
                        await session.execute(
                            select(User.id)
                            .join(User.roles)
                            .filter(Role.name == "partner")
                        )
                    )
                    .scalars()
                    .all()
                )
                addresses = list(
                    (await session.execute(select(Address))).scalars().all()
                )
                if len(addresses) < count:
                    raise Exception(
                        "Not enough addresses available to assign to properties."
                    )

                for _ in range(count):
                    if not user_ids:
                        raise Exception(
                            "No users available to assign as property owner."
                        )

                    owner_id = random.choice(user_ids)
                    address = random.choice(addresses)
                    address_id = address.id
                    addresses.remove(address)
                    property_category = random.choice(VIETNAM_PROPERTY_CATEGORIES)
                    transaction_type = random.choice(VIETNAM_TRANSACTION_TYPES)
                    title = generate_title(
                        property_category, transaction_type, address.city
                    )
                    price = round(random.uniform(50000, 500000), 2)
                    bedrooms = random.randint(1, 5)
                    bathrooms = random.choice([1.0, 1.5, 2.0, 2.5, 3.0])
                    sqm = random.randint(20, 500)
                    status = random.choice([True, False])
                    active = True
                    description = generate_html_description(
                        title,
                        property_category,
                        transaction_type,
                        price,
                        bedrooms,
                        bathrooms,
                        sqm,
                    )
                    property_obj = Property(
                        title=title,
                        property_category=property_category,
                        transaction_type=transaction_type,
                        price=price,
                        bedrooms=bedrooms,
                        bathrooms=bathrooms,
                        sqm=sqm,
                        status=status,
                        description=description,
                        active=active,
                        owner_id=owner_id,
                        address_id=address_id,
                    )
                    await self.repository(session=session).add(property_obj)
            except:
                await session.rollback()
                raise
            finally:
                await session.commit()

    async def drop_all(self) -> None:
        async with sqlalchemy_config.get_session() as session:
            await self.repository(session=session).delete_where(
                Property.id.is_not(None)
            )
            await session.commit()
