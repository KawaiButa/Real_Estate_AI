from litestar.background_tasks import BackgroundTask
from typing import Annotated, Any
from uuid import UUID
from litestar import Controller, Request, Response, post, get
from litestar.di import Provide
from litestar.pagination import OffsetPagination
from litestar.security.jwt.token import Token
from litestar.params import Body
from litestar.enums import RequestEncodingType
from database.utils import provide_pagination_params
from database.models.review import Review, ReviewResponse, ReviewSchema
from database.models.user import User
from domains.review.dtos import (
    ReviewCreateDTO,
    ReviewFilterDTO,
    ReviewResponseDTO,
    provide_filter_param,
)
from domains.review.service import RatingService, provide_review_service
from advanced_alchemy.filters import LimitOffset


class RatingController(Controller):
    path = "/properties/{property_id:uuid}/ratings"
    dependencies = {
        "rating_service": Provide(provide_review_service),
    }
    tags = ["Review"]

    @post()
    async def create_review(
        self,
        rating_service: RatingService,
        data: Annotated[
            ReviewCreateDTO, Body(media_type=RequestEncodingType.MULTI_PART)
        ],
        property_id: UUID,
        request: Request[User, Token, Any],
    ) -> Response[Review]:
        review = await rating_service.create_review(data, property_id, request.user.id)
        return Response(
            content=rating_service.to_schema(review, schema_type=ReviewSchema),
            background=BackgroundTask(
                rating_service._update_property_rating, property_id
            ),
        )

    @get(
        dependencies={
            "pagination": Provide(provide_pagination_params),
            "filters": Provide(provide_filter_param, sync_to_thread=True),
        },
        no_auth=True,
    )
    async def list_reviews(
        self,
        property_id: UUID,
        pagination: LimitOffset,
        filters: ReviewFilterDTO,
        rating_service: RatingService,
    ) -> OffsetPagination[ReviewSchema]:
        return await rating_service.list_paginated(
            property_id=property_id, pagination=pagination, filters=filters
        )

    @post("/{rating_id:uuid}/owner-response", status_code=201)
    async def add_owner_response(
        self,
        rating_service: RatingService,
        property_id: UUID,
        rating_id: UUID,
        data: Annotated[
            ReviewResponseDTO, Body(media_type=RequestEncodingType.MULTI_PART)
        ],
        request: Request[User, Token, Any],
    ) -> ReviewResponse:
        return await rating_service.add_response(
            rating_id, property_id, request.user.id, data.response_text
        )

    @post("/{rating_id: uuid}/toggle-helpful", status_code=200)
    async def toggle_helpful_vote(
        self,
        rating_service: RatingService,
        rating_id: UUID,
        request: Request[User, Token, Any],
    ) -> str:
        return await rating_service.toggle_helpful_vote(
            rating_id=rating_id, user_id=request.user.id
        )
