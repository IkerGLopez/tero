"""add-error-code-to-test-case-result

Revision ID: e1f2a3b4c5d6
Revises: d5e6f7a8b9c0
Create Date: 2026-04-22

"""

from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op


revision: str = 'e1f2a3b4c5d6'
down_revision: Union[str, None] = 'd5e6f7a8b9c0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('test_case_result', sa.Column('error_code', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('test_case_result', 'error_code')
