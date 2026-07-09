import abc
from datetime import datetime, timezone
from typing import Optional, List, cast

from sqlmodel import SQLModel, Field, Relationship

from ..teams.domain import PublicTeamRole, Team, TeamRole, GLOBAL_TEAM_ID, TeamRoleStatus, Role


class BaseUser(SQLModel, abc.ABC):
    id: int = Field(primary_key=True, default=None)
    username: str = Field(index=True, max_length=50)
    name: Optional[str] = Field(max_length=100, default=None, index=True)


class User(BaseUser, table=True):
    monthly_usd_limit: int
    monthly_hours: int = Field(default=160)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    deleted_at: Optional[datetime] = Field(default=None)
    team_roles: List[TeamRole] = Relationship()

    def is_active(self) -> bool:
        if self.deleted_at is None:
            return True
        deleted_at = self.deleted_at
        # Records inserted directly into the database may lack timezone info,
        # for that reason we set it to the current timezone if it's not set
        if deleted_at.tzinfo is None:
            deleted_at = deleted_at.replace(tzinfo=timezone.utc)
        return deleted_at > datetime.now(timezone.utc)

    def is_member_of(self, team_id: int) -> bool:
        return team_id == GLOBAL_TEAM_ID or any(cast(Team, tr.team).id == team_id and tr.status == TeamRoleStatus.ACCEPTED for tr in self.team_roles)

    def is_global_owner(self) -> bool:
        return any(
            tr.role == Role.TEAM_OWNER
            and tr.team_id == GLOBAL_TEAM_ID
            and tr.status == TeamRoleStatus.ACCEPTED
            for tr in self.team_roles
        )


class UserListItem(BaseUser):

    @staticmethod
    def from_user(user: Optional[User]) -> Optional['UserListItem']:
        return UserListItem.model_validate(user) if user else None


class UserProfile(SQLModel):
    teams: List[PublicTeamRole]

    @staticmethod
    def from_user(user: User) -> 'UserProfile':
        return UserProfile(
            teams=[
                PublicTeamRole(
                    id=tr.team.id,
                    name=tr.team.name,
                    role=tr.role,
                    status=tr.status
                )
                for tr in user.team_roles if tr.team and tr.status in [TeamRoleStatus.PENDING, TeamRoleStatus.ACCEPTED]
            ]
        )
