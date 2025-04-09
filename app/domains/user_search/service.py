from database.models.user_search import UserSearch
from database.models.user_action import UserAction
from advanced_alchemy.repository import SQLAlchemyAsyncRepository
from advanced_alchemy.service import SQLAlchemyAsyncRepositoryService

class UserSearchRepository(SQLAlchemyAsyncRepository[UserSearch]):
    model_type = UserSearch
class UserSearchService(SQLAlchemyAsyncRepositoryService[UserSearch]):
    repository_type = UserSearchRepository