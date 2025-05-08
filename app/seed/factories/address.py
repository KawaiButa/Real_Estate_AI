import uuid
import random
import json
from datetime import datetime
from geoalchemy2 import WKTElement
import pygeohash as pgh
from faker import Faker
from configs.sqlalchemy import sqlalchemy_config
from seed.factories.base import BaseFactory
from database.models.address import Address
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from openai import OpenAI

fake = Faker("vi_VN")
client = OpenAI()


class AddressRepository(SQLAlchemyAsyncRepository[Address]):
    model_type = Address


def generate_full_address() -> str:
    """
    Fallback method using Faker to generate a full address string.
    """
    street_address = fake.street_address()
    city = fake.city()
    state = fake.state() if hasattr(fake, "state") else ""
    postal_code = fake.postcode()
    country = fake.country()
    full_address = f"{street_address}, {city}, {state} {postal_code}, {country}"
    return full_address


def generate_precise_address() -> dict:
    """
    Generates a more precise address in Vietnam by calling ChatGPT.
    The LLM is expected to return a JSON string with the following keys:
      - street: The street address.
      - postal_code: The postal code.
      - neighborhood: A detailed full address.

    After obtaining the address details, the function overrides the city field
    using the vietnam-provinces package.
    If the call fails, it falls back to Faker.
    """
    prompt = (
        "You are an expert in New York addresses. Generate a precise and realistic address in New York state. "
        "Include details such as the street name, district/ward, and postal code. "
        "Return the address in the following JSON format without any additional text: "
        '{"street": "<street address>", "postal_code": "<postal code>", "neighborhood": "<full address>"}'
    )
    try:
        response = client.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Using GPT-3.5 Turbo for free generation
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150,
        )
        result = json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        result = {
            "street": fake.street_address(),
            "postal_code": fake.postcode(),
            "neighborhood": generate_full_address(),
        }
    return result


class AddressFactory(BaseFactory):
    repository = AddressRepository

    async def seed(self, count: int) -> None:
        async with sqlalchemy_config.get_session() as session:
            repository = self.repository(session=session)
            try:
                for _ in range(count):
                    precise_address = generate_precise_address()
                    # Generate random coordinates inside Vietnam.
                    latitude = random.uniform(8.55, 23.39)
                    longitude = random.uniform(102.17, 109.47)

                    address = Address(
                        id=str(uuid.uuid4()),
                        street=precise_address.get("street", fake.street_address()),
                        city=precise_address.get("city", fake.city()),
                        postal_code=precise_address.get("postal_code", fake.postcode()),
                        neighborhood=precise_address.get(
                            "neighborhood", generate_full_address()
                        ),
                        latitude=latitude,
                        longitude=longitude,
                        coordinates=WKTElement(
                            f"POINT({longitude} {latitude})", srid=4326
                        ),
                        geohash=pgh.encode(latitude, longitude),
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                    )
                    await repository.add(data=address, auto_commit=False)
            finally:
                await repository.session.commit()

    async def drop_all(self) -> None:
        async with sqlalchemy_config.get_session() as session:
            repository = self.repository(session=session)
            await repository.delete_where(Address.id.is_not(None))
            await repository.session.commit()
