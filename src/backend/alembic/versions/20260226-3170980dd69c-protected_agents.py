"""is_protected_agents

Revision ID: 3170980dd69c
Revises: 8e9d8ca4dd0c
Create Date: 2026-02-26 13:37:10.458412

"""
import sqlalchemy as sa
from typing import Sequence, Union
from alembic import op


# revision identifiers, used by Alembic.
revision: str = '3170980dd69c'
down_revision: Union[str, None] = '8e9d8ca4dd0c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('agent', sa.Column('is_protected', sa.Boolean(), nullable=False, server_default=sa.false()))


def downgrade() -> None:
    op.drop_column('agent', 'is_protected')
