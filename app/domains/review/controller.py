# File: src/app/domain/properties/controllers/ratings.py
from typing import Any
from uuid import UUID
from litestar import Controller, Request, post, get
from litestar.di import Provide
from litestar.pagination import OffsetPagination
from litestar.security.jwt.token import Token

from database.utils import provide_pagination_params
from database.models.review import Review, ReviewResponse
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
        data: ReviewCreateDTO,
        request: Request[User, Token, Any],
    ) -> Review:
        return await rating_service.create_review(data, request.user.id)

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
    ) -> OffsetPagination[Review]:
        return await rating_service.list_paginated(
            property_id=property_id, pagination=pagination, filters=filters
        )

    @post("/{rating_id:uuid}/owner-response", status_code=201)
    async def add_owner_response(
        self,
        rating_service: RatingService,
        property_id: UUID,
        rating_id: UUID,
        data: ReviewResponseDTO,
        request: Request[User, Token, Any],
    ) -> ReviewResponse:
        return await rating_service.add_response(
            rating_id, property_id, request.user.id, data.response_text
        )
