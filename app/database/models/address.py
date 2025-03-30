from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional
from sqlalchemy import (
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column
from geoalchemy2 import Geography
from litestar.dto import dto_field
from database.models.base import (
    BaseModel,
    BaseSchema,
)  # Adjust the import per your project setup


@dataclass
class Address(BaseModel):
    __tablename__ = "addresses"
    street: Mapped[str] = mapped_column(String(255), nullable=False)
    city: Mapped[str] = mapped_column(String(100), nullable=False)
    postal_code: Mapped[str | None] = mapped_column(String(20), nullable=True)
    neighborhood: Mapped[str | None] = mapped_column(String(100), nullable=True)
    latitude: Mapped[float | None] = mapped_column(Numeric(12, 9, asdecimal=False), nullable=False)
    longitude: Mapped[float | None] = mapped_column(Numeric(12, 9, asdecimal=False), nullable=False)
    coordinates: Mapped[str | None] = mapped_column(
        Geography("POINT", srid=4326), nullable=True, info=dto_field(mark="private")
    )
    geohash: Mapped[str | None] = mapped_column(
        Text, nullable=True, default=None, info=dto_field(mark="private")
    )

class AddressSchema(BaseSchema):
    street: str
    city: str
    postal_code: Optional[str]
    neighborhood: Optional[str]
    latitude: float
    longitude: float
