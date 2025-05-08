import random
import uuid
from faker import Faker
from sqlalchemy import select
from database.models.image import Image
from domains.address.service import AddressService
from domains.image.service import ImageService
from database.models.property_type import PropertyType
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
from advanced_alchemy.utils.fixtures import open_fixture
from database.models.address import Address
from configs.sqlalchemy import sqlalchemy_config
from advanced_alchemy.utils.text import slugify

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
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=5000,
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


def parse_price(raw_price: str) -> float:
    try:
        price_str = str(raw_price)
        cleaned = price_str[1:].replace(",", "") if len(price_str) > 0 else "0.0"
        return float(cleaned)
    except (ValueError, TypeError) as e:
        print(f"Failed to parse price '{raw_price}': {e}")
        return 0.0


class PropertyFactory(BaseFactory):
    repository = PropertyRepository

    async def seed(self, count: int = 10, fixture_path: str = "seed/fixtures") -> None:
        async with sqlalchemy_config.get_session() as session:
            try:
                if fixture_path:
                    address_service = AddressService(session=session)
                    image_service = ImageService(session=session)
                    property_data = open_fixture(fixture_path, "properties")
                    # property_data = property_data[:10]
                    property_data = [
                        {
                            "id": uuid.UUID(int=int(data["id"])),
                            "property_type_id": (
                                await PropertyType.as_unique_async(
                                    session=session,
                                    name=data["room_type"],
                                    slug=slugify(data["room_type"]),
                                )
                            ).id,
                            "description": data["description"],
                            "property_category": data["room_type"],
                            "bedrooms": int(
                                float(
                                    data["bedrooms"]
                                    if len(data["bedrooms"]) > 0
                                    else "0.0"
                                )
                            ),
                            "bathrooms": int(
                                float(
                                    data["bathrooms"]
                                    if len(data["bathrooms"]) > 0
                                    else "0.0"
                                )
                            ),
                            "transaction_type": random.choice(
                                VIETNAM_TRANSACTION_TYPES
                            ),
                            "title": data["name"],
                            "sqm": round(random.uniform(10.0, 1000.0), 2),
                            "status": random.choice([True, False]),
                            "average_rating": (
                                float(data["review_scores_rating"])
                                if len(data["review_scores_rating"]) > 0
                                else 0.0
                            ),
                            "price": parse_price(data["price"]) * 25000,
                            "owner_id": uuid.UUID(int=int(data["host_id"])),
                            "active": True,
                            "address_id": (
                                await address_service.create_address(
                                    latitude=float(data["latitude"]),
                                    longitude=float(data["longitude"]),
                                    street=data["neighbourhood_cleansed"],
                                    neighborhood=data["neighbourhood_cleansed"],
                                    city=data["neighbourhood_group_cleansed"],
                                )
                            ).id,
                            "image_urls": data["image_urls"],
                        }
                        for data in property_data
                    ]
                    property_list = []
                    image_list = []
                    for data in property_data:
                        id = data["id"]
                        image_urls = data["image_urls"]
                        data.pop("image_urls")
                        if not data["property_type_id"]:
                            print(data)
                            continue
                        property_list.append(Property(**data))
                        image_list.extend(
                            [
                                Image(
                                    **{
                                        "url": url,
                                        "model_id": id,
                                        "model_type": "property",
                                    }
                                )
                                for url in image_urls
                            ]
                        )
                    await self.repository(session=session).add_many(data=property_list)
                    await image_service.create_many(data=image_list)
                    return
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
                    property_type_id = (
                        await PropertyType.as_unique_async(
                            session=session,
                            name=property_category,
                            slug=slugify(property_category),
                        )
                    ).id
                    transaction_type = random.choice(VIETNAM_TRANSACTION_TYPES)
                    title = generate_title(
                        property_category, transaction_type, address.city
                    )
                    price = round(random.uniform(500000, 50000000), 0)
                    bedrooms = random.randint(1, 5)
                    bathrooms = random.randint(1, 5)
                    sqm = round(random.uniform(10.0, 1000.0), 2)
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
                        property_type_id=property_type_id,
                        sqm=sqm,
                        status=status,
                        description=description,
                        active=active,
                        owner_id=owner_id,
                        address_id=address_id,
                    )
                    await self.repository(session=session).add(property_obj)
            except Exception as e:
                print(str(e))
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
