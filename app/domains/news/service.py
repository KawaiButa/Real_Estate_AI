from collections.abc import AsyncGenerator
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService
from sqlalchemy import asc, desc, select
from database.models.tag import Tag
from database.models.article import Article
from sqlalchemy.ext.asyncio import AsyncSession
from advanced_alchemy.filters import LimitOffset
from litestar.pagination import OffsetPagination
from enum import Enum
from datetime import datetime
from typing import Optional
from litestar.params import Parameter
from litestar.openapi.spec.example import Example
from pydantic import BaseModel, field_validator, ValidationInfo
from sqlalchemy.orm import selectinload


class ArticleRepository(SQLAlchemyAsyncRepository[Article]):
    model_type = Article


class ArticleOrder(str, Enum):
    TITLE = "title"
    PUBLISH_DATE = "publish_date"
    AUTHOR = "author"


class ArticleSearchParams(BaseModel):
    title: Optional[str] = Parameter(
        None,
        title="Title",
        description="Search articles by title keyword",
        examples=[Example(value="Breaking News")],
    )
    author: Optional[str] = Parameter(
        None,
        title="Author",
        description="Filter articles by author",
        examples=[Example(value="John Doe")],
    )
    tag: Optional[str] = Parameter(
        None,
        title="Tag",
        description="Filter articles by tag",
        examples=[Example(value="tech")],
    )
    published_after: Optional[datetime] = Parameter(
        None,
        title="Published After",
        description="Filter articles published after this date",
        examples=[Example(value="2022-01-01T00:00:00Z")],
    )
    published_before: Optional[datetime] = Parameter(
        None,
        title="Published Before",
        description="Filter articles published before this date",
        examples=[Example(value="2023-01-01T00:00:00Z")],
    )
    order_by: Optional[ArticleOrder] = Parameter(
        default=ArticleOrder.PUBLISH_DATE,
        title="Order By",
        description="Field to order the articles by",
        examples=[Example(value="publish_date")],
    )
    order_direction: Optional[str] = Parameter(
        default="desc",
        title="Order Direction",
        description="Sorting order (asc or desc)",
        examples=[Example(value="asc")],
    )

    @field_validator("order_direction")
    def validate_order_direction(cls, value, info: ValidationInfo):
        if value and value.lower() not in {"asc", "desc"}:
            raise ValueError("order_direction must be either 'asc' or 'desc'")
        return value.lower() if value else value


class ArticleService(SQLAlchemyAsyncRepositoryService[Article]):
    repository_type = ArticleRepository

    async def get_articles(
        self,
        search_param: ArticleSearchParams,
        pagination: LimitOffset,
    ) -> OffsetPagination[Article]:
        # Base query with related tags preloaded
        query = select(Article).options(selectinload(Article.tags))

        # Apply text filter for title
        if search_param.title:
            query = query.where(Article.title.ilike(f"%{search_param.title}%"))

        # Apply filter for author
        if search_param.author:
            query = query.where(Article.author.ilike(f"%{search_param.author}%"))

        # Apply tag filter by joining the Tag table
        if search_param.tag:
            query = query.join(Article.tags).where(
                Tag.name.ilike(f"%{search_param.tag}%")
            )

        # Apply publication date range filters
        if search_param.published_after:
            query = query.where(Article.publish_date >= search_param.published_after)
        if search_param.published_before:
            query = query.where(Article.publish_date <= search_param.published_before)

        # Determine ordering field based on search parameters
        if search_param.order_by == ArticleOrder.TITLE:
            order_field = Article.title
        elif search_param.order_by == ArticleOrder.AUTHOR:
            order_field = Article.author
        else:
            order_field = Article.publish_date

        # Apply ordering direction
        if search_param.order_direction.lower() == "asc":
            order_expression = asc(order_field)
        else:
            order_expression = desc(order_field)

        paginated_query = (
            query.offset(pagination.offset)
            .limit(pagination.limit)
            .order_by(order_expression)
        )

        # Execute the query and retrieve results
        items = (
            (await self.repository.session.execute(paginated_query))
            .scalars()
            .unique()
            .all(),
        )
        total = await self.count()  # Adjust this method if you need a filtered count

        return OffsetPagination(
            items=items[0],
            total=total,
            limit=pagination.limit,
            offset=pagination.offset,
        )


async def provide_article_service(
    db_session: AsyncSession,
) -> AsyncGenerator[ArticleService, None]:
    async with ArticleService.new(session=db_session) as service:
        yield service


# Optional extractor function to integrate with your endpoint query parameters
async def article_query_params_extractor(
    title: Optional[str] = Parameter(
        None,
        title="Title",
        description="Article title search keyword",
        examples=[Example(value="Breaking News")],
    ),
    author: Optional[str] = Parameter(
        None,
        title="Author",
        description="Filter articles by author",
        examples=[Example(value="Jane Doe")],
    ),
    tag: Optional[str] = Parameter(
        None,
        title="Tag",
        description="Filter articles by tag",
        examples=[Example(value="tech")],
    ),
    published_after: Optional[datetime] = Parameter(
        None,
        title="Published After",
        description="Filter articles published after this date",
        examples=[Example(value="2022-01-01T00:00:00Z")],
    ),
    published_before: Optional[datetime] = Parameter(
        None,
        title="Published Before",
        description="Filter articles published before this date",
        examples=[Example(value="2023-01-01T00:00:00Z")],
    ),
    order_by: Optional[ArticleOrder] = Parameter(
        default=ArticleOrder.PUBLISH_DATE,
        title="Order By",
        description="Field to order articles by",
        examples=[Example(value="publish_date")],
    ),
    order_direction: Optional[str] = Parameter(
        default="desc",
        title="Order Direction",
        description="Sorting order (asc or desc)",
        examples=[Example(value="asc")],
    ),
) -> ArticleSearchParams:
    return ArticleSearchParams(
        title=title,
        author=author,
        tag=tag,
        published_after=published_after,
        published_before=published_before,
        order_by=order_by,
        order_direction=order_direction,
    )
