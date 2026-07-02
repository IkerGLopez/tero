"""increase_recursion_limit

Revision ID: a7808316b52b
Revises: b3f7a91d2c4e
Create Date: 2026-06-25 14:03:07.419949

"""
import sqlalchemy as sa
import sqlmodel
from typing import Sequence, Union
from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'a7808316b52b'
down_revision: Union[str, None] = 'b3f7a91d2c4e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE agent DROP CONSTRAINT IF EXISTS agent_recursion_limit_range"
    )
    op.create_check_constraint(
        "agent_recursion_limit_range",
        "agent",
        "recursion_limit >= 20 AND recursion_limit <= 130",
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE agent DROP CONSTRAINT IF EXISTS agent_recursion_limit_range"
    )
    op.create_check_constraint(
        "agent_recursion_limit_range",
        "agent",
        "recursion_limit >= 20 AND recursion_limit <= 100",
    )
