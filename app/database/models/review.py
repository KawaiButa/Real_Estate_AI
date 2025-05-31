from typing import TYPE_CHECKING, Optional
from sqlalchemy import (
    ForeignKey,
    Integer,
    String,
    Boolean,
    CheckConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship, Mapped, mapped_column, foreign
import uuid
from database.models.image import Image, ImageSchema
from database.models.base import BaseModel, BaseSchema

if TYPE_CHECKING:
    from database.models.user import UserSchema
    from database.models.user import User


class Review(BaseModel):
    __tablename__ = "reviews"
    __table_args__ = (
        CheckConstraint("rating >= 1 AND rating <= 5", name="rating_range_check"),
    )

    property_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("properties.id", ondelete="CASCADE"), index=True
    )
    reviewer_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    rating: Mapped[int] = mapped_column(Integer, nullable=False)
    review_text: Mapped[str] = mapped_column(String(2000))
    helpful_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    featured: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    # Relationships
    reviewer: Mapped["User"] = relationship("User", lazy="selectin")
    responses: Mapped[list["ReviewResponse"]] = relationship(
        "ReviewResponse", lazy="joined"
    )
    images: Mapped[list[Image]] = relationship(
        "Image",
        primaryjoin=(
            "and_(foreign(Image.model_id) == Review.id, "
            "Image.model_type == 'review')"
        ),
        foreign_keys=[foreign(Image.model_id)],
        remote_side=[Image.model_id],
        backref="review",
        lazy="selectin",
    )
    helpful_votes: Mapped[list["HelpfulVote"]] = relationship(
        "HelpfulVote", lazy="joined"
    )


class HelpfulVoteSchema(BaseSchema):
    review_id: uuid.UUID
    voter_id: uuid.UUID


class ReviewResponseSchema(BaseSchema):
    review_id: uuid.UUID
    response_text: str


class ReviewSchema(BaseSchema):
    property_id: uuid.UUID
    reviewer_id: uuid.UUID
    reviewer: Optional["UserSchema"]
    review_text: str
    rating: int
    helpful_votes: list[HelpfulVoteSchema] = []
    responses: list[ReviewResponseSchema] = []
    has_voted: Optional[bool] = None
    images: list[ImageSchema]


class ReviewMedia(BaseModel):
    __tablename__ = "review_medias"

    review_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("reviews.id", ondelete="CASCADE")
    )
    media_url: Mapped[str] = mapped_column(String(500))
    media_type: Mapped[str] = mapped_column(String(20))


class ReviewResponse(BaseModel):
    __tablename__ = "review_responses"

    review_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("reviews.id", ondelete="CASCADE")
    )
    response_text: Mapped[str] = mapped_column(String(2000))


class HelpfulVote(BaseModel):
    __tablename__ = "helpful_votes"

    review_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("reviews.id", ondelete="CASCADE")
    )
    voter_id: Mapped[uuid.UUID] = mapped_column(
        UUID, ForeignKey("users.id", ondelete="CASCADE")
    )
