from litestar.plugins.sqlalchemy import SQLAlchemyDTO, SQLAlchemyDTOConfig
from database.models.user import User


class ProfileUpdateDTO(SQLAlchemyDTO[User]):
    config = SQLAlchemyDTOConfig()
