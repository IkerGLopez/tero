from datetime import datetime, timezone
import logging
from typing import Optional

from sqlalchemy.orm import selectinload
from sqlmodel import col, select, and_, delete, or_
from sqlmodel.ext.asyncio.session import AsyncSession

from ..core.repos import attr, scalar
from ..teams.domain import TeamRole
from .domain import User


logger = logging.getLogger(__name__)


class UserRepository:

    def __init__(self, db: AsyncSession):
        self._db = db

    async def find_by_id(self, user_id: int) -> Optional[User]:
        stmt = select(User).where(and_(
            User.id == user_id,
            or_(col(User.deleted_at).is_(None), col(User.deleted_at) > datetime.now(timezone.utc))
        )).options(
            selectinload(attr(User.team_roles)).selectinload(attr(TeamRole.team))
        )
        result = await self._db.exec(stmt)
        return result.one_or_none()

    async def find_by_username(self, username: str) -> Optional[User]:
        stmt = (
            select(User)
            .where(and_(
                User.username == username,
                or_(col(User.deleted_at).is_(None), col(User.deleted_at) > datetime.now(timezone.utc))
            ))
            .options(
                selectinload(attr(User.team_roles)).selectinload(attr(TeamRole.team))
            )
        )
        result = await self._db.exec(stmt)
        return result.one_or_none()

    async def find_by_username_include_deleted(self, username: str) -> Optional[User]:
        stmt = (
            select(User)
            .where(User.username == username)
            .options(
                selectinload(attr(User.team_roles)).selectinload(attr(TeamRole.team))
            )
        )
        result = await self._db.exec(stmt)
        return result.one_or_none()

    async def create_user(self, user: User) -> User:
        self._db.add(user)
        await self._db.commit()
        await self._db.refresh(user, ['id', 'team_roles'])
        return user

    async def find_all_users(self) -> list[User]:
        stmt = select(User).where(or_(col(User.deleted_at).is_(None), col(User.deleted_at) > datetime.now(timezone.utc)))
        result = await self._db.exec(stmt)
        return list(result.all())

    async def update_user(self, user: User) -> User:
        self._db.add(user)
        await self._db.commit()
        await self._db.refresh(user, ['id', 'team_roles'])
        return user

    async def delete_user(self, user_id: int):
        stmt = delete(User).where(col(User.id) == user_id)
        await self._db.exec(scalar(stmt))
        await self._db.commit()
