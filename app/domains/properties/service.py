import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Optional
from uuid import UUID
import uuid
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from pydantic import BaseModel, ValidationInfo, field_validator
from sqlalchemy import Enum, asc, desc, func, select
from litestar.params import Parameter
from litestar.openapi.spec.example import Example
from litestar.exceptions import ValidationException, NotAuthorizedException
from sqlalchemy.orm import joinedload, selectinload
from database.models.property_type import PropertyType
from domains.address.service import AddressService
from database.models.user import User
from domains.properties.dtos import (
    CreatePropertyDTO,
    CreatePropertySchema,
    UpdatePropertySchema,
)
from domains.image.service import ImageService, provide_image_service
from domains.supabase.service import SupabaseService, provide_supabase_service
from database.models.address import Address
from database.models.property import Property
from sqlalchemy.ext.asyncio import AsyncSession
from advanced_alchemy.filters import LimitOffset
from litestar.pagination import OffsetPagination
from sqlalchemy.orm import noload
from advanced_alchemy.utils.text import slugify

# Vietnam-specific property constants
VIETNAM_PROPERTY_CATEGORIES = [
    "apartment",
    "villa",
    "townhouse",
    "commercial",
    "land",
    "residential",
]
VIETNAM_TRANSACTION_TYPES = ["rent", "sale"]
VIETNAM_PROPERTY_STATUSES = ["available", "occupied", "pending", "expired"]


class PropertyOrder(str, Enum):
    TITLE = "title"
    PRICE = "price"
    CREATED_AT = "created_at"


class PropertyRepository(SQLAlchemyAsyncRepository[Property]):
    model_type = Property


class PropertySearchParams(BaseModel):
    lat: Optional[float] = Parameter(
        None,
        title="Latitude",
        description="Latitude of search center (Vietnam coordinates)",
    )
    lng: Optional[float] = Parameter(
        None,
        title="Longitude",
        description="Longitude of search center (Vietnam coordinates)",
    )
    radius: Optional[float] = Parameter(
        10.0,
        title="Search Radius",
        description="Radius in kilometers for location-based search",
    )

    # Core property filters
    min_price: Optional[float] = Parameter(
        None,
        title="Minimum Price",
        description="Minimum price in VND million",
    )
    max_price: Optional[float] = Parameter(
        None,
        title="Maximum Price",
        description="Maximum price in VND million",
    )
    property_category: Optional[str] = Parameter(
        None,
        title="Property Type",
        description=f"Vietnam property types: {', '.join(VIETNAM_PROPERTY_CATEGORIES)}",
    )
    transaction_type: Optional[str] = Parameter(
        None,
        title="Transaction Type",
        description=f"Allowed values: {', '.join(VIETNAM_TRANSACTION_TYPES)}",
    )
    min_bedrooms: Optional[int] = Parameter(
        None,
        title="Minimum Bedrooms",
        gt=1,
    )
    min_bathrooms: Optional[int] = Parameter(
        None,
        title="Minimum Bathrooms",
        gt=1,
    )
    min_sqm: Optional[int] = Parameter(
        None,
        title="Minimum Area (sqm)",
        gt=20,
    )
    status: Optional[str] = Parameter(
        None, description=f"Property status: {', '.join(VIETNAM_PROPERTY_STATUSES)}"
    )
    created_after: Optional[datetime] = Parameter(
        None, description="Filter properties listed after this date"
    )

    # Vietnam-specific filters
    city: Optional[str] = Parameter(
        None, description="City name (Hanoi, Ho Chi Minh, Da Nang, etc)"
    )
    district: Optional[str] = Parameter(None, description="District within city")
    direction_facing: Optional[str] = Parameter(
        None, description="Preferred building direction (East, West, etc)"
    )
    order_by: Optional[str] = Parameter(
        None,
        title="Order By",
        description="Sorting field (e.g., price, created_at). Use '-' prefix for descending order",
        examples=[Example(value="price"), Example(value="created_at")],
    )
    order_direction: Optional[str] = Parameter(
        default="desc",
        query="orderDirection",
        title="Order Direction",
        description="Sorting order (asc or desc)",
    )

    @field_validator("transaction_type", "status", "property_category")
    def validate_enum_fields(cls, value, info: ValidationInfo):
        allowed_values = {
            "transaction_type": VIETNAM_TRANSACTION_TYPES,
            "status": VIETNAM_PROPERTY_STATUSES,
            "property_category": VIETNAM_PROPERTY_CATEGORIES,
        }.get(info.field_name)
        if value and value.lower() not in allowed_values:
            raise ValueError(
                f"Invalid {info.field_name}. Allowed values: {', '.join(allowed_values)}"
            )
        return value.lower() if value else value

    class Config:
        populate_by_name = True


class PropertyService(SQLAlchemyAsyncRepositoryService[Property]):
    repository_type = PropertyRepository
    default_radius = 10
    supabase_service: SupabaseService = provide_supabase_service(bucket_name="property")

    async def search(
        self,
        search_param: PropertySearchParams,
        pagination: LimitOffset,
    ) -> OffsetPagination[Property]:
        query = select(Property).options(
            joinedload(Property.address), joinedload(Property.owner)
        )

        if search_param.lat and search_param.lng:
            point = func.ST_SetSRID(
                func.ST_MakePoint(search_param.lng, search_param.lat), 4326
            )
            query = query.where(
                func.ST_DWithin(
                    Address.coordinates,
                    point,
                    search_param.radius * 1000,
                )
            )

        # Apply price range filter
        if search_param.min_price is not None:
            query = query.where(Property.price >= search_param.min_price)
        if search_param.max_price is not None:
            query = query.where(Property.price <= search_param.max_price)

        # Apply categorical filters
        if search_param.property_category:
            query = query.where(
                Property.property_category == search_param.property_category
            )
        if search_param.transaction_type:
            query = query.where(
                Property.transaction_type == search_param.transaction_type
            )
        if search_param.status:
            query = query.where(Property.status == search_param.status)

        # Apply numeric filters
        if search_param.min_bedrooms:
            query = query.where(Property.bedrooms >= search_param.min_bedrooms)
        if search_param.min_bathrooms:
            query = query.where(Property.bathrooms >= search_param.min_bathrooms)
        if search_param.min_sqm:
            query = query.where(Property.sqm >= search_param.min_sqm)

        # Apply Vietnam-specific filters
        if search_param.city:
            query = query.where(Property.address.city.ilike(f"%{search_param.city}%"))
        if search_param.district:
            query = query.where(
                Property.address.district.ilike(f"%{search_param.district}%")
            )

        # Apply date filter
        if search_param.created_after:
            query = query.where(Property.created_at >= search_param.created_after)
        if search_param.order_by == PropertyOrder.TITLE:
            order_field = Property.title
        elif search_param.order_by == PropertyOrder.PRICE:
            order_field = Property.price
        else:
            order_field = Property.created_at

        if search_param.order_direction.lower() == "asc":
            order_expression = asc(order_field)
        else:
            order_expression = desc(order_field)
        paginated_query = (
            query.offset(pagination.offset)
            .limit(pagination.limit)
            .order_by(order_expression)
        )
        # Execute queries concurrently
        items = (
            (await self.repository.session.execute(paginated_query))
            .scalars()
            .unique()
            .all(),
        )
        total = await self.count()

        return OffsetPagination(
            items=items[0],
            total=total,
            limit=pagination.limit,
            offset=pagination.offset,
        )

    async def create(self, data: CreatePropertySchema, user_id: uuid.UUID) -> Property:
        # Vietnam-specific validation
        if data.transaction_type not in VIETNAM_TRANSACTION_TYPES:
            raise ValidationException(
                f"Invalid transaction type. Allowed: {VIETNAM_TRANSACTION_TYPES}"
            )
        if data.property_category not in VIETNAM_PROPERTY_CATEGORIES:
            raise ValidationException(
                f"Invalid property category. Allowed: {VIETNAM_PROPERTY_CATEGORIES}"
            )
        data_dict = data.model_dump(
            exclude={
                "deleted_images",
                "image_list",
                "latitude",
                "longitude",
                "city",
                "street",
            }
        )
        address_service = AddressService(self.repository.session)
        data_dict["address_id"] = (
            await address_service.create_address(
                latitude=data.latitude,
                longitude=data.longitude,
                street=data.street,
                city=data.city,
                neighborhood=data.neighborhood,
                auto_commit=False,
            )
        ).id
        data_dict["property_type_id"] = (
            await PropertyType.as_unique_async(
                session=self.repository.session,
                name=data.property_category,
                slug=slugify(data.property_category),
            )
        ).id
        data_dict["owner_id"] = user_id
        property = await super().create(
            data=data_dict,
            auto_refresh=True,
            auto_commit=False,
        )
        image_services = ImageService(session=self.repository.session)
        images = await image_services.create_many(
            [
                {
                    "url": (await self.supabase_service.upload_file(image)),
                    "model_id": property.id,
                    "model_type": "property",
                }
                for image in data.image_list
            ],
            auto_commit=False,
        )
        property.images = images
        await self.repository.session.commit()
        return property

    async def update(
        self,
        item_id: UUID,
        data: UpdatePropertySchema,
        user_id: uuid.UUID | None = None,
    ) -> Property:
        property = await self.get_one_or_none(
            Property.id.__eq__(item_id),
        )
        if user_id and property.owner_id != user_id:
            raise NotAuthorizedException(f"You are not allowed to delete this property")
        if not property:
            raise ValidationException(f"There is not property with id {item_id}")
        restricted_fields = {"transaction_type", "owner_id", "created_at"}
        if any(field in data for field in restricted_fields):
            raise ValidationException("Cannot modify restricted fields")
        data_dict = data.model_dump(
            exclude={
                "deleted_images",
                "image_list",
                "latitude",
                "longitude",
                "city",
                "street",
            }
        )
        if data.latitude:
            data_dict["address"] = {
                "latitude": data.latitude,
                "longitude": data.longitude,
                "city": data.city,
                "street": data.street,
            }
        image_service = ImageService(session=self.repository.session)
        data_dict["images"] = property.images
        if data.deleted_images and len(data.deleted_images) > 0:
            data_dict["images"] = [
                image for image in property.images if image.id not in data.delete_images
            ]
            image_service.delete_many(data.deleted_images, auto_commit=False)
            asyncio.gather(
                *[self.supabase_service.delete_image(id) for id in data.deleted_images]
            )

        if data.image_list and len(data.image_list) > 0:
            data_dict["images"].extend(
                await image_service.create_many(
                    [
                        {
                            "url": (await self.supabase_service.upload_file(image)),
                            "model_id": property.id,
                            "model_type": "property",
                        }
                        for image in data.image_list
                    ],
                    auto_commit=True,
                )
            )
        return await super().update(
            item_id=item_id, data=data_dict, auto_refresh=True, auto_commit=True
        )

    async def update_activation(
        self, user_id: uuid.UUID, property_id: uuid.UUID, activate: bool | None
    ):
        property = await self.get_one_or_none(Property.id.__eq__(property_id))
        if not property:
            raise ValidationException(f"No property found with id {property_id}")
        if property is not user_id:
            raise ValidationException(f"This property is not belong to user {user_id}")
        if not activate:
            activate = not property.active
        property.active = activate
        property = await super().update(
            item_id=property_id, data=property, auto_commit=True, auto_refresh=True
        )
        return property


async def provide_property_service(
    db_session: AsyncSession,
) -> AsyncGenerator[PropertyService]:

    async with PropertyService.new(session=db_session) as service:
        yield service


async def query_params_extractor(
    lat: Optional[float] = Parameter(
        None,
        title="Latitude",
        description="Latitude of search center (Vietnam coordinates)",
    ),
    lng: Optional[float] = Parameter(
        None,
        title="Longitude",
        description="Longitude of search center (Vietnam coordinates)",
    ),
    radius: Optional[float] = Parameter(
        10.0,
        title="Search Radius",
        description="Radius in kilometers for location-based search",
    ),
    # Core property filters
    min_price: Optional[float] = Parameter(
        None,
        title="Minimum Price",
        description="Minimum price in VND million",
    ),
    max_price: Optional[float] = Parameter(
        None,
        title="Maximum Price",
        description="Maximum price in VND million",
    ),
    property_category: Optional[str] = Parameter(
        None,
        title="Property Type",
        description=f"Vietnam property types: {', '.join(VIETNAM_PROPERTY_CATEGORIES)}",
    ),
    transaction_type: Optional[str] = Parameter(
        None,
        title="Transaction Type",
        description=f"Allowed values: {', '.join(VIETNAM_TRANSACTION_TYPES)}",
    ),
    min_bedrooms: Optional[int] = Parameter(
        None,
        title="Minimum Bedrooms",
        gt=1,
    ),
    min_bathrooms: Optional[int] = Parameter(
        None,
        title="Minimum Bathrooms",
        gt=1,
    ),
    min_sqm: Optional[int] = Parameter(
        None,
        title="Minimum Area (sqm)",
        gt=20,
    ),
    status: Optional[str] = Parameter(
        None, description=f"Property status: {', '.join(VIETNAM_PROPERTY_STATUSES)}"
    ),
    created_after: Optional[datetime] = Parameter(
        None, description="Filter properties listed after this date"
    ),
    # Vietnam-specific filters
    city: Optional[str] = Parameter(
        None, description="City name (Hanoi, Ho Chi Minh, Da Nang, etc)"
    ),
    district: Optional[str] = Parameter(None, description="District within city"),
    direction_facing: Optional[str] = Parameter(
        None, description="Preferred building direction (East, West, etc)"
    ),
    order_by: Optional[str] = Parameter(
        default=PropertyOrder.CREATED_AT,
        query="orderBy",
        title="Order By Field",
        description="Field to order results by (title, price, created_at)",
    ),
    order_direction: Optional[str] = Parameter(
        default="desc",
        query="orderDirection",
        title="Order Direction",
        description="Sorting order (asc or desc)",
    ),
) -> PropertySearchParams:
    return PropertySearchParams(
        lat=lat,
        lng=lng,
        radius=radius,
        min_price=min_price,
        max_price=max_price,
        property_category=property_category,
        transaction_type=transaction_type,
        min_bedrooms=min_bedrooms,
        min_bathrooms=min_bathrooms,
        min_sqm=min_sqm,
        status=status,
        created_after=created_after,
        city=city,
        district=district,
        direction_facing=direction_facing,
        order_by=order_by,
        order_direction=order_direction,
    )
