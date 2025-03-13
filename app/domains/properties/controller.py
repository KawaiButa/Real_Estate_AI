from typing import Any
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
from database.models.property import Property
from domains.properties.dtos import (
    CreatePropertyDTO,
    CreatePropertyReturnDTO,
    PropertySearchReturnDTO,
    UpdatePropertyDTO,
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
        return_dto=CreatePropertyReturnDTO,
        status_code=HTTP_201_CREATED,
        guards=[
            role_guard([GuardRole.ADMIN, GuardRole.PARTNER]),
        ],
    )
    async def create_property(
        self,
        data: DTOData[Property],
        property_service: PropertyService,
    ) -> Property:
        return property_service.create(data)

    @patch(
        "/{property_id: uuid}",
        dto=UpdatePropertyDTO,
        return_dto=SQLAlchemyDTO[Property],
        status_code=HTTP_200_OK,
        guards=[
            role_guard([GuardRole.ADMIN, GuardRole.PARTNER]),
        ],
    )
    async def update_property(
        self,
        property_id: uuid.UUID,
        data: DTOData[Property],
        property_service: PropertyService,
        request: Request[User, Token, Any],
    ) -> Property:
        property = await property_service.get_one_or_none(
            Property.id.__eq__(property_id)
        )
        if not property:
            raise ValidationException(f"There is not property with id {property_id}")
        if (
            "admin" in [role.name for role in request.user.roles]
            or property.owner_id is not request.user.id
        ):
            raise NotAuthorizedException(f"You are not allowed to update this property")
        property = data.update_instance(property)
        return await property_service.update(property_id, property)

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
        property = await property_service.get_one_or_none(
            Property.id.__eq__(property_id)
        )
        if not property:
            raise ValidationException(f"There is not property with id {property_id}")
        if (
            "admin" in [role.name for role in request.user.roles]
            or property.owner_id is not request.user.id
        ):
            raise NotAuthorizedException(f"You are not allowed to update this property")
        return await property_service.delete(item_id=property_id, auto_commit=True)

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
