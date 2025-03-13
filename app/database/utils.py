from litestar.params import Parameter
from advanced_alchemy.filters import LimitOffset
# Pagination dependency
async def provide_pagination_params(
    page: int = Parameter(ge=1, default=1, query="page"),
    page_size: int = Parameter(
        ge=1, 
        le=100, 
        default=20, 
        query="pageSize",
        description="Number of items per page (max 100)"
    ),
) -> LimitOffset:
    """Dependency to inject pagination parameters"""
    return LimitOffset(page_size, (page - 1) * page_size)