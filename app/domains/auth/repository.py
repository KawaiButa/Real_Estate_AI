from advanced_alchemy.repository import (
    SQLAlchemyAsyncRepository,
)
from configs.sqlalchemy import provide_transaction
from database.models.user import User
from litestar.datastructures import State


class UserRepository(SQLAlchemyAsyncRepository[User]):
    """User SQLAlchemy Repository."""
    model_type = User

async def provide_users_repo(state: State) -> UserRepository:
    async with provide_transaction(state=state) as db_session:
        return UserRepository(session=db_session)
