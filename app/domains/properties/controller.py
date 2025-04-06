from typing import Annotated, Any, Optional
import uuid
from litestar import Controller, Request, delete, get, patch, post
from litestar.di import Provide
from database.utils import provide_pagination_params
from domains.properties.service import (
    PropertySearchParams,
    PropertyService,
    provide_property_service,
    query_params_extractor,
)
from litestar.params import Parameter
from database.models.property import Property, PropertySchema
from domains.properties.dtos import (
    CreatePropertyDTO,
    CreatePropertyReturnDTO,
    CreatePropertySchema,
    PropertySearchReturnDTO,
    UpdatePropertyDTO,
    UpdatePropertySchema,
    UpdateStatusModel,
)
from litestar.dto import DTOData
from litestar.security.jwt.token import Token
from database.models.user import User
from litestar.plugins.sqlalchemy import SQLAlchemyDTO
from litestar.exceptions import ValidationException, NotAuthorizedException
from litestar.status_codes import HTTP_200_OK, HTTP_201_CREATED
from advanced_alchemy.filters import LimitOffset
from litestar.pagination import OffsetPagination
from domains.auth.guard import GuardRole, login_guard, role_guard
from litestar.params import Body
from litestar.enums import RequestEncodingType
from sqlalchemy.orm import lazyload


class PropertyController(Controller):
    path = "property"
    tags = ["Property"]

    dependencies = {"property_service": Provide(provide_property_service)}

    @get(
        "/",
        dependencies={
            "params": Provide(query_params_extractor),
            "pagination": Provide(provide_pagination_params),
        },
        no_auth=True,
        description="Get property",
    )
    async def get_properties(
        self,
        params: PropertySearchParams,
        property_service: PropertyService,
        pagination: LimitOffset,
    ) -> OffsetPagination[Property]:
        return await property_service.search(search_param=params, pagination=pagination)

    @post(
        "/",
        dto=CreatePropertyDTO,
        return_dto=None,
        status_code=HTTP_201_CREATED,
        guards=[
            role_guard([GuardRole.ADMIN, GuardRole.PARTNER]),
        ],
    )
    async def create_property(
        self,
        data: Annotated[
            CreatePropertySchema, Body(media_type=RequestEncodingType.MULTI_PART)
        ],
        property_service: PropertyService,
        request: Request[User, Token, Any],
    ) -> PropertySchema:
        return property_service.to_schema(
            await property_service.create(data, user_id=request.user.id),
            schema_type=PropertySchema,
        )

    @patch(
        "/{property_id: uuid}",
        dto=UpdatePropertyDTO,
        return_dto=None,
        status_code=HTTP_200_OK,
        guards=[
            role_guard([GuardRole.ADMIN, GuardRole.PARTNER]),
        ],
    )
    async def update_property(
        self,
        property_id: uuid.UUID,
        data: Annotated[
            UpdatePropertySchema, Body(media_type=RequestEncodingType.MULTI_PART)
        ],
        property_service: PropertyService,
        request: Request[User, Token, Any],
    ) -> PropertySchema:
        if not "admin" in [role["name"] for role in request.user.roles]:
            user_id = request.user.id
        else:
            user_id = None
        return property_service.to_schema(
            await property_service.update_property(property_id, data=data, user_id=user_id),
            schema_type=PropertySchema,
        )

    @delete(
        "/{property_id: uuid}",
        guards=[
            role_guard([GuardRole.ADMIN, GuardRole.PARTNER]),
        ],
    )
    async def delete_property(
        self,
        property_id: uuid.UUID,
        property_service: PropertyService,
        request: Request[User, Token, Any],
    ) -> None:
        property_service.repository.merge_loader_options = False
        property = await property_service.get_one_or_none(
            Property.id.__eq__(property_id), load=[lazyload("*")]
        )
        if not property:
            raise ValidationException(f"There is no property with id {property_id}")

        if not (
            "admin" in [role["name"] for role in request.user.roles]
            or property.owner_id == request.user.id
        ):
            raise NotAuthorizedException(f"You are not allowed to delete this property")

        await property_service.delete(
            item_id=property_id, auto_commit=True, load=[lazyload("*")]
        )
        return

    @post(
        "/{property_id: uuid}/status",
        guards=[role_guard([GuardRole.ADMIN, GuardRole.PARTNER])],
        opt={"admin": GuardRole.ADMIN, "partner": GuardRole.PARTNER},
    )
    async def active_property(
        self,
        property_id: uuid.UUID,
        data: UpdateStatusModel,
        property_service: PropertyService,
        request: Request[User, Token, Any],
    ) -> Property:
        if "admin" in [role.name for role in request.user.roles]:
            user_id = None
        else:
            user_id = request.user.id
        return await property_service.update_activation(
            property_id=property_id, activate=data.active, user_id=user_id
        )
    @get("/count", no_auth=True)
    async def count_by_city(self,property_service: PropertyService,  type: Optional[str]) -> Any:
        return await property_service.count_by_city(type=type)