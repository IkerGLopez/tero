"""add-error-reason-to-file

Revision ID: 9c4e2a7f1b8d
Revises: c3d4e5f6a1b2
Create Date: 2026-09-16

The File model gained an ``error_reason`` column (bulk-indexing error
diagnosis) but no migration was created for it, so every environment
built from the migration chain was missing the column and any file
INSERT failed with UndefinedColumn.
"""

from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op


revision: str = '9c4e2a7f1b8d'
down_revision: Union[str, None] = 'c3d4e5f6a1b2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('file', sa.Column('error_reason', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('file', 'error_reason')
