from litestar import Controller, get
from litestar.di import Provide
from database.models.article import Article
from database.utils import provide_pagination_params
from domains.news.service import (
    ArticleSearchParams,
    article_query_params_extractor,
    provide_article_service,
    ArticleService,
)
from advanced_alchemy.filters import LimitOffset
from litestar.pagination import OffsetPagination


class ArticleController(Controller):
    path = "/news"

    dependencies = {"article_service": Provide(provide_article_service)}

    @get(
        "/",
        dependencies={
            "params": Provide(article_query_params_extractor),
            "pagination": Provide(provide_pagination_params),
        },
        no_auth=True,
    )
    async def get_news(
        self,
        article_service: ArticleService,
        params: ArticleSearchParams,
        pagination: LimitOffset,
    ) -> OffsetPagination[Article]:
        return await article_service.get_articles(
            search_param=params, pagination=pagination
        )
