from collections.abc import AsyncGenerator
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from database.models.address import Address
from database.models.image import Image
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from geoalchemy2 import Geometry, WKTElement
import pygeohash as pgh


from sqlalchemy.ext.asyncio import AsyncSession


class AddressRepository(SQLAlchemyAsyncRepository[Address]):
    model_type = Address


class AddressService(SQLAlchemyAsyncRepositoryService[Address]):
    repository_type = AddressRepository

    async def create_address(
        self,
        latitude: float,
        longitude: float,
        city: str,
        street: str,
        neighborhood: str | None = None,
        geohash: str | None = None,
        auto_commit: bool = False,
        auto_refresh: bool = False,
    ) -> Address:
        if not geohash:
            geohash = pgh.encode(latitude, longitude)
        return await self.create(
            data={
                "latitude": latitude,
                "longitude": longitude,
                "city": city,
                "street": street,
                "neighborhood": neighborhood,
                "coordinates": WKTElement(f"POINT({longitude} {latitude})", srid=4326),
                "geohash": geohash,
            },
            auto_commit=auto_commit,
            auto_refresh=auto_refresh,
        )
