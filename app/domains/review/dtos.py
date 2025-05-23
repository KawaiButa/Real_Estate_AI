from dataclasses import dataclass
from typing import Optional
from litestar.datastructures import UploadFile
from pydantic import BaseModel, ConfigDict, Field, field_validator
from uuid import UUID
from litestar.params import Parameter


class ReviewCreateDTO(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    rating: int = Field(..., ge=1, le=5)
    review_text: str = Field(..., min_length=50, max_length=2000)
    image_list: Optional[list[UploadFile]]

    @field_validator("image_list")
    def validate_media(cls, v):
        if len(v) > 5:
            raise ValueError("Maximum 5 media attachments allowed")
        return v


class ReviewUpdateDTO(BaseModel):
    rating: int | None = Field(None, ge=1, le=5)
    review_text: str | None = Field(None, min_length=50, max_length=2000)
    media_urls: list[str] | None = None


class ReviewResponseDTO(BaseModel):
    response_text: str = Field(..., min_length=20, max_length=1000)


@dataclass
class ReviewFilterDTO:
    min_rating: int | None = Field(None, ge=1, le=5)
    has_media: bool = False
    sort_by: str = Field("recent", pattern="^(recent|helpful|rating)$")


def provide_filter_param(
    min_rating: Optional[float] = Parameter(
        ge=0.0, le=5.0, default=0.0, query="minRating"
    ),
    has_media: Optional[bool] = Parameter(default=None, query="hasMedia"),
    sort_by: Optional[str] = Parameter(default="recent", query="sortBy"),
) -> ReviewFilterDTO:
    return ReviewFilterDTO(min_rating, has_media, sort_by)
