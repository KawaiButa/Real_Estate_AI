from litestar import Controller, post
from litestar.di import Provide

from database.models.user_action import UserAction
from domains.user_action.dtos import CreateUserActionDTO
from domains.user_action.service import (
    UserActionService,
    provide_user_action_service,
)


class UserActionController(Controller):
    path = "actions"
    tags = ["actions"]

    dependencies = {"user_action_service": Provide(provide_user_action_service)}

    @post(no_auth=True)
    async def createAction(
        self, body: CreateUserActionDTO, user_action_service: UserActionService
    ) -> UserAction:
        return await user_action_service.create(
            body.to_dict(), auto_commit=True, auto_refresh=True
        )
