from datetime import datetime, timedelta, timezone
import hashlib
import secrets
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from collections.abc import AsyncGenerator
from typing import Any
from passlib.hash import bcrypt
from advanced_alchemy.utils.text import slugify

from advanced_alchemy.service import (
    SQLAlchemyAsyncRepositoryService,
)
from litestar.exceptions import ValidationException
from domains.email.service import MailService
from domains.auth.repository import UserRepository
from database.models.user import User, UserSchema
from litestar.dto import DTOData
from domains.auth.dtos import LoginReturnSchema, RegisterReturnModel
from security.oauth2 import oauth2_auth
from database.models.role import Role


class AuthService(SQLAlchemyAsyncRepositoryService[User]):
    repository_type = UserRepository
    default_role = "customer"

    async def register(
        self,
        userData: DTOData[User],
    ) -> RegisterReturnModel:
        user = userData.create_instance()
        existedUser = await self.get_one_or_none(User.email.__eq__(user.email))
        if existedUser:
            raise ValidationException("The email is assigned with another account")
        user.password = bcrypt.hash(user.password)
        user.roles.extend(
            [
                await Role.as_unique_async(
                    self.repository.session,
                    name=self.default_role,
                    slug=slugify(self.default_role),
                )
            ]
        )
        user = await self.create(user, auto_commit=True, auto_refresh=True)
        return oauth2_auth.login(
            identifier=str(
                {
                    "id": str(user.id),
                    "name": user.name,
                    "roles": [role.name for role in user.roles],
                }
            ),
            response_body=self.to_schema(data=user, schema_type=UserSchema),
        )

    async def login(self, userData: DTOData[User]) -> LoginReturnSchema:
        try:
            data = userData.create_instance()
            user = await self.get_one_or_none(
                User.email.__eq__(data.email),
            )

            if not user or not bcrypt.verify(data.password, user.password):
                raise ValidationException("Invalid credentials")
            if data.device_token:
                user = await self.update(data={
                    "device_token": data.device_token,
                    
                }, item_id=user.id)
            return LoginReturnSchema(
                token=oauth2_auth.create_token(
                    identifier=str(
                        {
                            "id": str(user.id),
                            "name": user.name,
                            "roles": [
                                {
                                    "id": str(user.id),
                                    "name": role.name,
                                }
                                for role in user.roles
                            ],
                        }
                    ),
                ),
                user=self.to_schema(data=user, schema_type=UserSchema),
            )
        except Exception as e:
            await self.repository.session.rollback()
            raise e
        finally:
            await self.repository.session.commit()

    async def update_role(
        self,
        user_id: uuid.UUID,
        roles: list[str],
        auto_commit: bool = False,
        auto_refresh: bool = False,
    ) -> User:
        user = await self.get_one_or_none(User.id.__eq__(user_id))
        if not user:
            raise ValidationException("No user found")
        user.roles.extend(
            [
                await Role.as_unique_async(
                    self.repository.session,
                    name=role,
                    slug=slugify(self.default_role),
                )
                for role in roles
            ]
        )
        user = await self.update(
            user=user, auto_commit=auto_commit, auto_refresh=auto_refresh
        )
        return user

    async def generate_and_save_reset_token(self, user: User) -> str:
        if not user:
            raise ValueError("User not found")

        raw_token = secrets.token_urlsafe(32)
        hashed_token = hashlib.sha256(raw_token.encode()).hexdigest()

        # Set token expiration time (e.g., 1 hour from now).
        expiration = datetime.now(timezone.utc) + timedelta(minutes=15)
        user.reset_password_token = hashed_token
        user.reset_password_expires = expiration
        user = await self.update(
            user, item_id=user.id, auto_commit=True, auto_refresh=True
        )
        return raw_token

    async def forgot_password(self, mail_service: MailService, email: str) -> bool:
        user = await self.get_one_or_none(User.email.__eq__(email))
        if not user:
            raise ValidationException(f"No account linked with email {email}")
        token = await self.generate_and_save_reset_token(user=user)
        await mail_service.send_forget_password_email(user=user, token=token)
        return True

    async def reset_password(self, token: str, new_password: str) -> User:
        hashed_token = hashlib.sha256(token.encode()).hexdigest()
        user = await self.get_one_or_none(
            User.reset_password_token.__eq__(hashed_token)
        )
        if not user or datetime.now(timezone.utc) > user.reset_password_expires:
            raise ValidationException("The token is incorrect or expired")
        user.password = bcrypt.hash(new_password)
        user.reset_password_expires = None
        user.reset_password_token = None
        user = await self.update(user, auto_commit=True, auto_refresh=True)
        return user


async def provide_auth_service(
    db_session: AsyncSession,
) -> AsyncGenerator[AuthService, None]:
    async with AuthService.new(session=db_session) as service:
        yield service
