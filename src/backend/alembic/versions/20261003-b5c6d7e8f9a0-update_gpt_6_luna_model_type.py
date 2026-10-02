"""update_gpt_6_luna_model_type

Revision ID: b5c6d7e8f9a0
Revises: a3b4c5d6e7f8
Create Date: 2026-10-03 00:00:00.000000

NOTE: GPT-6 Luna rejects non-default temperature (fixed-temperature reasoning
family), so it is classified as REASONING like the rest of its family.
"""
from typing import Sequence, Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b5c6d7e8f9a0'
down_revision: Union[str, None] = 'a3b4c5d6e7f8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("UPDATE llm_model SET model_type = 'REASONING' WHERE id = 'gpt-6-luna'")


def downgrade() -> None:
    op.execute("UPDATE llm_model SET model_type = 'CHAT' WHERE id = 'gpt-6-luna'")
