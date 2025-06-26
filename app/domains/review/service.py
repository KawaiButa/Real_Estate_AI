from collections.abc import AsyncGenerator
from uuid import UUID
from litestar.exceptions import (
    NotAuthorizedException,
    ValidationException,
    InternalServerException,
)
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from sqlalchemy import literal, select, func, and_, update
from sqlalchemy.orm import joinedload
from sqlalchemy.ext.asyncio import AsyncSession
from database.models.property import Property
from domains.image.service import ImageService
from domains.property_verification.service import VerificationService
from database.models.property_verification import PropertyVerification
from database.models.review import HelpfulVote, Review, ReviewResponse, ReviewSchema
from domains.properties.service import PropertyService
from domains.review.dtos import ReviewCreateDTO, ReviewFilterDTO
from litestar.pagination import OffsetPagination
from advanced_alchemy.filters import LimitOffset
from litestar.background_tasks import BackgroundTask


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


class ReviewResponseRepository(SQLAlchemyAsyncRepository[ReviewResponse]):
    model_type = ReviewResponse


class ReviewResponseService(SQLAlchemyAsyncRepositoryService[ReviewResponse]):
    repository_type = ReviewResponseRepository


class HelpfulVoteRepository(SQLAlchemyAsyncRepository[HelpfulVote]):
    model_type = HelpfulVote


class HelpfulVoteService(SQLAlchemyAsyncRepositoryService[HelpfulVote]):
    repository_type = HelpfulVoteRepository


class RatingService(SQLAlchemyAsyncRepositoryService[Review]):
    repository_type = RatingRepository

    async def create_review(
        self, data: ReviewCreateDTO, property_id: UUID, user_id: UUID
    ) -> Review:
        verification_service = VerificationService(session=self.repository.session)
        verification = await verification_service.get_one_or_none(
            PropertyVerification.property_id == property_id,
            PropertyVerification.user_id == user_id,
        )
        if not verification:
            raise NotAuthorizedException("Property verification required")

        await self._fraud_checks(user_id, property_id)
        data = data.model_dump()
        data["reviewer_id"] = user_id
        review = await self.create(data=data, auto_refresh=True)
        image_services = ImageService(session=self.repository.session)
        images = await image_services.create_many(
            [
                {
                    "url": (await self.supabase_service.upload_file(image)),
                    "model_id": review.id,
                    "model_type": "review",
                }
                for image in data.image_list
            ],
            auto_commit=False,
        )
        review.images = images
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
    ) -> OffsetPagination[ReviewSchema]:
        query_filters = [Review.property_id == property_id]

        if filters.min_rating:
            query_filters.append(Review.rating >= filters.min_rating)
        if filters.has_media:
            query_filters.append(Review.images.any())

        order_by = {
            "recent": Review.created_at.desc(),
            "helpful": Review.helpful_count.desc(),
            "rating": Review.rating.desc(),
        }[filters.sort_by]
        if filters.user_id:
            vote_exists = (
                select(literal(True))
                .where(
                    (HelpfulVote.review_id == Review.id)
                    & (HelpfulVote.voter_id == filters.user_id)
                )
                .exists()
            )

            stmt = (
                select(
                    Review,
                    vote_exists.label("has_voted"),
                )
                .where(*query_filters)
                .order_by(order_by)
                .offset(pagination.offset)
                .limit(pagination.limit)
            )

            result = await self.repository.session.execute(stmt)
            items = result.scalars().unique().all()
        else:
            stmt = (
                select(Review)
                .where(*query_filters)
                .order_by(order_by)
                .offset(pagination.offset)
                .limit(pagination.limit)
            )

            result = await self.repository.session.execute(stmt)
            items = result.scalars().unique().all()
        count_stmt = select(func.count()).select_from(Review).where(*query_filters)
        total = await self.repository.session.scalar(count_stmt)
        return OffsetPagination(
            items=self.to_schema(items, schema_type=ReviewSchema).items,
            total=total,
            limit=pagination.limit,
            offset=pagination.offset,
        )

    async def add_response(
        self, review_id: UUID, property_id: UUID, user_id: UUID, content: str
    ) -> ReviewResponse:
        property_service = PropertyService(session=self.repository.session)
        property = await property_service.get_one_or_none(Property.id == property_id)
        if not property:
            raise ValidationException("Property not found")
        if property.owner_id != user_id:
            raise NotAuthorizedException("You are not owner of this property.")
        review_response_service = ReviewResponseService(session=self.repository.session)
        return await review_response_service.create(
            {
                "review_id": review_id,
                "response_text": content,
            },
            auto_commit=True,
            auto_refresh=True,
        )

    async def toggle_helpful_vote(self, rating_id: UUID, user_id: UUID) -> HelpfulVote:
        session = self.repository.session
        try:
            existing = await HelpfulVoteService(session).get_one_or_none(
                HelpfulVote.review_id == rating_id,
                HelpfulVote.voter_id == user_id,
            )
            if existing:
                await HelpfulVoteService(session).delete(existing.id)
                await session.execute(
                    update(Review)
                    .where(Review.id == rating_id)
                    .values(helpful_vote_count=Review.helpful_vote_count - 1)
                )
                await session.commit()
                return existing

            vote = await HelpfulVoteService(session).create(
                {
                    "review_id": rating_id,
                    "voter_id": user_id,
                },
                auto_commit=False,
                auto_refresh=False,
            )
            await session.execute(
                update(Review)
                .where(Review.id == rating_id)
                .values(helpful_vote_count=Review.helpful_vote_count + 1)
            )
            await session.commit()
            await session.refresh(vote.review)
            return vote
        except Exception as e:
            print(e)
            session.rollback()
            raise InternalServerException(
                "Unknown server error. Please try again later"
            )


async def provide_review_service(
    db_session: AsyncSession,
) -> AsyncGenerator[RatingService, None]:

    async with RatingService.new(session=db_session) as service:
        yield service
