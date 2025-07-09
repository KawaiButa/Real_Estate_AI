import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime
import os
from typing import Dict, List, Optional
from uuid import UUID
import uuid
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
import numpy as np
from pydantic import BaseModel, ValidationInfo, field_validator
import requests
from sqlalchemy import Enum, and_, asc, desc, exists, func, literal, select
from litestar.params import Parameter
from litestar.openapi.spec.example import Example
from litestar.exceptions import ValidationException, NotAuthorizedException
from sqlalchemy.orm import joinedload
from database.models.review import Review
from domains.user_action.service import UserActionRepository
from domains.user_search.service import UserSearchService
from database.models.property_type import PropertyType
from domains.address.service import AddressService
from database.models.user import User
from domains.properties.dtos import (
    CreatePropertySchema,
    UpdatePropertySchema,
)
from domains.image.service import ImageService
from domains.supabase.service import SupabaseService, provide_supabase_service
from database.models.address import Address
from database.models.property import Favorite, Property, PropertySchema
from sqlalchemy.ext.asyncio import AsyncSession
from advanced_alchemy.filters import LimitOffset
from litestar.pagination import OffsetPagination, CursorPagination
from sqlalchemy.orm import noload
from advanced_alchemy.utils.text import slugify
from litestar.stores.memory import MemoryStore
from configs.pinecone import property_index
from pinecone import Vector
from firebase_admin import credentials, messaging

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
store = MemoryStore()


class PropertyOrder(str, Enum):
    TITLE = "title"
    PRICE = "price"
    CREATED_AT = "created_at"


class PropertyRepository(SQLAlchemyAsyncRepository[Property]):
    model_type = Property


class PropertySearchParams(BaseModel):

    search: Optional[str] = Parameter(
        title="Search Query", description="Search query to check in the description"
    )
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
    has_review: Optional[bool] = Parameter(
        False,
        title="Toggle have review",
        description="Toggle to search for property that have review",
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
    recommended: Optional[bool] = Parameter(
        default=False,
        query="recommended",
        title="Toggle recommendation system",
        description="Whether using the recommender system or not",
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
        user_id: uuid.UUID | None = None,
    ) -> OffsetPagination[PropertySchema] | CursorPagination[str, Property]:
        """
        Route between recommendation (cursor-based) and normal search (offset-based).
        """
        if search_param.recommended and user_id:
            return await self._search_recommended(search_param, pagination, user_id)
        return await self._search_normal(search_param, pagination, user_id)

    async def get_relevant_property(
        self, property_id: UUID, pagination: LimitOffset | None = None
    ) -> OffsetPagination[PropertySchema]:
        result = await self.fetch_pinecone_document_by_id([property_id])
        if len(result) == 0:
            return OffsetPagination(
                items=[],
                offset=pagination.offset,
                limit=pagination.limit,
                total=0,
            )
        pine_res = property_index.query(
            vector=result[str(property_id)],
            top_k=1000,
        )
        ids = [m["id"] for m in pine_res["matches"]]
        property = self.to_schema(
            await self._fetch_properties_from_ids(ids, pagination=pagination),
            schema_type=PropertySchema,
        ).items
        return OffsetPagination(
            items=property,
            offset=pagination.offset,
            limit=pagination.limit,
            total=len(pine_res),
        )

    async def _search_recommended(
        self,
        search_param: PropertySearchParams,
        pagination: LimitOffset,
        user_id: uuid.UUID,
    ) -> OffsetPagination[PropertySchema]:
        # meta_filter = self._build_pinecone_filter(search_param)
        user_embedding = await self._compute_user_embedding(user_id)
        pine_res = property_index.query(
            vector=user_embedding,
            # filter=meta_filter,
            top_k=1000,
        )
        ids = [m["id"] for m in pine_res["matches"]]
        query = self._build_sql_query(search_param, user_id, ids)
        order_exp = self._build_order_expression(search_param)
        paginated = (
            query.order_by(order_exp).offset(pagination.offset).limit(pagination.limit)
        )
        items = await self.finalize_and_add_optional_field(paginated, recommended=True)

        total = await self.repository.session.scalar(
            select(func.count()).select_from(query.subquery())
        )
        await self._record_search(user_id, search_param, recommended=True)

        return OffsetPagination(
            items=items,
            offset=pagination.offset,
            limit=pagination.limit,
            total=total,
        )

    async def _search_normal(
        self,
        search_param: PropertySearchParams,
        pagination: LimitOffset,
        user_id: uuid.UUID | None,
    ) -> OffsetPagination[PropertySchema]:
        query = self._build_sql_query(search_param, user_id)
        order_exp = self._build_order_expression(search_param)
        paginated = (
            query.order_by(order_exp).offset(pagination.offset).limit(pagination.limit)
        )
        items = await self.finalize_and_add_optional_field(paginated)

        total = await self.repository.session.scalar(
            select(func.count()).select_from(query.subquery())
        )
        await self._record_search(user_id, search_param, recommended=True)

        return OffsetPagination(
            items=items,
            offset=pagination.offset,
            limit=pagination.limit,
            total=total,
        )

    async def finalize_and_add_optional_field(
        self, query, recommended: bool = False
    ) -> List[PropertySchema]:
        result = await self.repository.session.execute(query)
        items = result.unique().all()
        property_schemas = []
        for prop, is_favorited in items:
            data = prop.to_schema() if hasattr(prop, "to_schema") else prop.__dict__
            data["is_favorited"] = is_favorited
            data["recommended"] = recommended
            schema = PropertySchema.model_validate(data)
            property_schemas.append(schema)
        return property_schemas

    def _build_sql_query(
        self,
        search_param: PropertySearchParams,
        user_id: uuid.UUID | None,
        ids: List[uuid.UUID] = [],
    ):
        subquery = (
            select(literal(True))
            .where(
                and_(Favorite.user_id == user_id, Favorite.property_id == Property.id)
            )
            .correlate(Property)
            .exists()
        )
        query = (
            select(Property, subquery.label("is_favorited"))
            .options(
                joinedload(Property.address),
                joinedload(Property.owner),
                noload(Property.reviews),
            )
            .join(Property.owner)
            .where(User.id != user_id)
        )
        if len(ids) > 0:
            query = query.where(Property.id.in_(ids))

        # geo filter
        if search_param.lat is not None and search_param.lng is not None:
            query = query.join(Property.address)
            radius_meters = search_param.radius * 1000
            radius_degrees = radius_meters / 111320.0
            lat = search_param.lat
            lng = search_param.lng
            min_lat = lat - radius_degrees
            max_lat = lat + radius_degrees
            min_lng = lng - radius_degrees
            max_lng = lng + radius_degrees
            query = query.where(
                and_(
                    Address.latitude >= min_lat,
                    Address.latitude <= max_lat,
                    Address.longitude >= min_lng,
                    Address.longitude <= max_lng,
                )
            )
        # price filters
        if search_param.min_price is not None:
            query = query.where(Property.price >= search_param.min_price)
        if search_param.max_price is not None:
            query = query.where(Property.price <= search_param.max_price)
        # Have review
        if search_param.has_review:
            subquery = select(Review.id).where(Review.property_id == Property.id)
            query = query.where(exists(subquery))
        # categorical
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
        # numeric
        if search_param.min_bedrooms:
            query = query.where(Property.bedrooms >= search_param.min_bedrooms)
        if search_param.min_bathrooms:
            query = query.where(Property.bathrooms >= search_param.min_bathrooms)
        if search_param.min_sqm:
            query = query.where(Property.sqm >= search_param.min_sqm)
        # text and location
        if search_param.search:
            query = query.where(Property.description.contains(search_param.search))
            query = query.where(Address.city.ilike(f"%{search_param.city}%"))
        if search_param.district:
            query = query.where(Address.street.ilike(f"%{search_param.district}%"))
        # date
        if search_param.created_after:
            query = query.where(Property.created_at >= search_param.created_after)
        return query

    def _build_order_expression(self, search_param: PropertySearchParams):
        if search_param.order_by == PropertyOrder.TITLE:
            field = Property.title
        elif search_param.order_by == PropertyOrder.PRICE:
            field = Property.price
        else:
            field = Property.created_at
        return (
            asc(field) if search_param.order_direction.lower() == "asc" else desc(field)
        )

    def _build_pinecone_filter(self, search_param: PropertySearchParams) -> dict:
        """
        Translate SQL filters into Pinecone metadata filter syntax.
        """
        mf: dict = {}
        if search_param.property_category:
            mf["property_category"] = {"$eq": search_param.property_category}
        if search_param.transaction_type:
            mf["transaction_type"] = {"$eq": search_param.transaction_type}
        if search_param.status:
            mf["status"] = {"$eq": search_param.status}
        if search_param.min_price is not None:
            mf.setdefault("price", {})["$gte"] = search_param.min_price
        if search_param.max_price is not None:
            mf.setdefault("price", {})["$lte"] = search_param.max_price
        if search_param.min_bedrooms:
            mf.setdefault("bedrooms", {})["$gte"] = search_param.min_bedrooms
        if search_param.min_bathrooms:
            mf.setdefault("bathrooms", {})["$gte"] = search_param.min_bathrooms
        if search_param.min_sqm:
            mf.setdefault("sqm", {})["$gte"] = search_param.min_sqm
        # if search_param.city:
        #     mf["city"] = {"$eq": f"%{search_param.city}%"}
        return mf

    async def _fetch_properties_from_ids(
        self, ids: list[str], pagination: LimitOffset | None = None
    ) -> list[Property]:
        query = (
            select(Property)
            .options(joinedload(Property.address), joinedload(Property.owner))
            .where(Property.id.in_(ids))
        )
        if pagination:
            query = query.offset(pagination.offset).limit(pagination.limit)
        result = await self.repository.session.execute(query)
        props = result.scalars().all()
        return props

    async def _record_search(
        self,
        user_id: uuid.UUID | None,
        search_param: PropertySearchParams,
        recommended: bool = False,
    ):
        data = {
            "user_id": user_id,
            "search_query": search_param.search,
            "type": search_param.property_category,
            "min_price": search_param.min_price,
            "max_price": search_param.max_price,
            "recommended": recommended,
        }
        svc = UserSearchService(session=self.repository.session)
        await svc.create(data, auto_commit=True)

    async def fetch_pinecone_document_by_id(
        self, doc_ids: list[uuid.UUID]
    ) -> Dict[str, Vector]:
        """
        Fetch a single vector document (and its metadata) by ID from Pinecone.
        """
        response = property_index.fetch(ids=[str(idx) for idx in doc_ids])
        vectors = response.vectors
        return vectors

    async def _compute_user_embedding(self, user_id: uuid.UUID) -> list[float]:
        user_action_repository = UserActionRepository(session=self.repository.session)
        property_id_list = await user_action_repository.get_relevant_properties(
            user_id=user_id
        )
        if len(property_id_list) == 0:
            return next(iter(property_index.fetch(["0"]).vectors.values())).values
        result = await self.fetch_pinecone_document_by_id(property_id_list)
        vectors = [value.values for value in result.values()]
        mean_vector = np.mean(vectors, axis=0).tolist()
        return mean_vector

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

    async def update_property(
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

    def send_property_price_update(fcm_token: str, property_data: dict):
        """
        Send a notification about a property price change
        :param fcm_token: User's FCM token
        :param property_data: Dict with keys like title, price, image_url, etc.
        """

        title = f"Price Update: {property_data['title']}"
        body = f"The price has changed to ${property_data['new_price']:,}"

        message = messaging.Message(
            notification=messaging.Notification(
                title=title,
                body=body,
                image=property_data.get("image_url"),
            ),
            data={
                "property_id": str(property_data["id"]),
                "title": property_data["title"],
                "new_price": str(property_data["new_price"]),
                "image_url": property_data.get("image_url", ""),
            },
            token=fcm_token,
        )
        response = messaging.send(message)
        print("Notification sent:", response)

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

    async def count_by_city(self, type: Optional[str]) -> dict:
        query = (
            select(Address.city, func.count(Address.id).label("property_count"))
            .join(Property, Property.address_id == Address.id)
            .group_by(Address.city)
        )
        if type:
            query = query.where(Property.property_category == type)
        result = (await self.repository.session.execute(query)).fetchall()
        data = [{"city": row[0], "count": row[1]} for row in result]
        return [
            {**data_point, "url": await fetch_city_image(data_point["city"])}
            for data_point in data
        ]


async def provide_property_service(
    db_session: AsyncSession,
) -> AsyncGenerator[PropertyService]:

    async with PropertyService.new(session=db_session) as service:
        yield service


async def query_params_extractor(
    search: Optional[str] = Parameter(
        title="Search Query", description="Search query to check in the description"
    ),
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
        default=10.0,
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
    has_review: Optional[bool] = Parameter(
        False,
        title="Toggle have review",
        description="Toggle to search for property that have review",
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
    recommended: Optional[bool] = Parameter(
        default=False,
        query="recommended",
        title="Toggle recommendation system",
        description="Whether using the recommender system or not",
    ),
) -> PropertySearchParams:
    return PropertySearchParams(
        search=search,
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
        has_review=has_review,
        district=district,
        direction_facing=direction_facing,
        order_by=order_by,
        order_direction=order_direction,
        recommended=recommended,
    )


async def fetch_city_image(city_name: str) -> str:
    stored_data = await store.get(f"city_{city_name.replace(' ', '_')}")
    if stored_data:
        return str(stored_data)[2:-1]
    url = "https://api.pexels.com/v1/search"
    params = {
        "query": city_name,
    }
    headers = {
        "Authorization": os.getenv("PEXELS_API_KEY"),
    }
    response = requests.get(url, params=params, headers=headers)
    data = response.json()
    if data.get("results"):
        result = data["results"][0]["urls"]["regular"]
        city_name = city_name.replace(" ", "_")
        store.set(f"city_{city_name}", result)
        return result
    return None
