from typing import Annotated, Any
import uuid
from litestar import Controller, Request, get, post, delete
from litestar.di import Provide

from database.models.tourview import Tourview
from database.models.user import User
from domains.tourview.dtos import CreatePanoramaDTO
from domains.tourview.service import TourviewService, provide_tourview_service
from litestar.exceptions import (
    ValidationException,
    NotAuthorizedException,
    InternalServerException,
)
from litestar.security.jwt import Token
from litestar.params import Body
from litestar.enums import RequestEncodingType


class TourviewController(Controller):
    path = "/properties/{property_id:uuid}/tourviews"
    tags = ["Tourview"]

    dependencies = {"tourview_service": Provide(provide_tourview_service)}

    @get("")
    async def get_panorama(
        self, property_id: uuid.UUID, tourview_service: TourviewService
    ) -> Tourview:
        tourview = await tourview_service.get_one_or_none(
            Tourview.property_id == property_id
        )
        if not tourview:
            raise ValidationException(f"Property not found. Id: {property_id}")
        return tourview

    @post("")
    async def create_panorama(
        self,
        data: Annotated[
            CreatePanoramaDTO, Body(media_type=RequestEncodingType.MULTI_PART)
        ],
        property_id: uuid.UUID,
        tourview_service: TourviewService,
    ) -> Tourview:
        tourview = await tourview_service.create_tourview(property_id, data)
        return tourview

    @delete("/{tourview_id: uuid}")
    async def delete_panorama(
        self,
        tourview_id: uuid.UUID,
        tourview_service: TourviewService,
        request: Request[User, Token, Any],
    ) -> None:
        try:
            tourview = await tourview_service.get_one_or_none(
                Tourview.id == tourview_id
            )
            if not tourview:
                raise ValidationException(f"Tourview not found. Id: {tourview_id}")
            if tourview.property.owner.id != request.user.id:
                raise NotAuthorizedException(
                    f"User do not have authorization to make this action."
                )
            tourview_service.delete(item_id=tourview_id)
        except:
            raise InternalServerException(
                "There is issue when deleting tourview. Please try again later"
            )
