from collections.abc import AsyncGenerator
from uuid import UUID
from litestar.exceptions import NotAuthorizedException, ValidationException
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from sqlalchemy import select, func, and_, update
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from domains.property_verification.service import VerificationService
from database.models.property_verification import PropertyVerification
from database.models.review import HelpfulVote, Review, ReviewResponse
from domains.properties.service import PropertyService
from domains.review.dtos import ReviewCreateDTO, ReviewFilterDTO
from litestar.pagination import OffsetPagination
from advanced_alchemy.filters import LimitOffset


class RatingRepository(SQLAlchemyAsyncRepository):
    model_type = Review

    async def get_property_rating_summary(self, property_id: UUID) -> dict:
        stmt = select(
            func.avg(Review.rating).label("average"),
        ).where(Review.property_id == property_id)
        result = await self.session.execute(stmt)
        return result.one()

    async def get_verified_reviews(self, property_id: UUID) -> list[Review]:
        stmt = (
            select(Review)
            .options(joinedload(Review.media))
            .where(
                and_(
                    Review.property_id == property_id,
                    Review.is_verified == True,
                    Review.status == "approved",
                )
            )
            .order_by(Review.created_at.desc())
        )
        result = await self.session.execute(stmt)
        return result.scalars().all()

    async def log_helpful_vote(self, review_id: UUID, user_id: UUID) -> None:
        async with self.session.begin():
            # Prevent duplicate votes
            existing = await self.session.execute(
                select(HelpfulVote).where(
                    and_(
                        HelpfulVote.review_id == review_id,
                        HelpfulVote.voter_id == user_id,
                    )
                )
            )
            if not existing.scalar_one_or_none():
                vote = HelpfulVote(review_id=review_id, voter_id=user_id)
                self.session.add(vote)
                await self.session.execute(
                    update(Review)
                    .where(Review.id == review_id)
                    .values(helpful_count=Review.helpful_count + 1)
                )


class RatingService(SQLAlchemyAsyncRepositoryService[Review]):
    repository_type = RatingRepository

    async def create_review(self, data: ReviewCreateDTO, user_id: UUID) -> Review:
        verification_service = VerificationService(session=self.repository.session)
        verification = await verification_service.get_one_or_none(
            PropertyVerification.property_id == data.property_id,
            PropertyVerification.user_id == user_id,
        )
        if not verification:
            raise NotAuthorizedException("Property verification required")

        # Fraud detection checks
        await self._fraud_checks(user_id, data.property_id)
        data = data.model_dump()
        data["reviewer_id"] = user_id
        # Create review
        review = await self.create(data=data, auto_refresh=True)

        await self._update_property_rating(data.property_id)
        await self.repository.session.commit()
        return review

    async def _fraud_checks(self, user_id: UUID, property_id: UUID) -> None:
        """Perform multiple fraud detection checks"""
        # Check for existing review
        existing = await self.get_one_or_none(
            Review.reviewer_id == user_id, Review.property_id == property_id
        )
        if existing:
            raise ValidationException("Multiple reviews not allowed")

    async def _update_property_rating(self, property_id: UUID) -> None:
        """Recalculate weighted average rating"""
        summary = await self.repository.get_property_rating_summary(property_id)

        property_service = PropertyService(session=self.repository.session)
        await property_service.update(
            data={"average_rating": summary["average"]}, item_id=True
        )

    async def list_paginated(
        self,
        property_id: UUID,
        filters: ReviewFilterDTO,
        pagination: LimitOffset,
    ) -> OffsetPagination[Review]:
        query_filters = [Review.property_id == property_id]

        if filters.min_rating:
            query_filters.append(Review.rating >= filters.min_rating)
        if filters.has_media:
            query_filters.append(Review.media.any())

        order_by = {
            "recent": Review.created_at.desc(),
            "helpful": Review.helpful_count.desc(),
            "rating": Review.rating.desc(),
        }[filters.sort_by]
        items = await self.list_and_count(*query_filters, uniquify=True)
        return OffsetPagination(
            items=items[0],
            total=items[1],
            limit=pagination.limit,
            offset=pagination.offset,
        )

    async def add_response(
        self, review_id: UUID, property_id: UUID, user_id: UUID, content: str
    ) -> ReviewResponse:
        pass


async def provide_review_service(
    db_session: AsyncSession,
) -> AsyncGenerator[RatingService, None]:

    async with RatingService.new(session=db_session) as service:
        yield service
