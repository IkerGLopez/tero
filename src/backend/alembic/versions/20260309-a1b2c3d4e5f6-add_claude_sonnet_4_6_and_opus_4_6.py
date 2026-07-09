"""add-claude-sonnet-4-6-and-opus-4-6

Revision ID: a1b2c3d4e5f6
Revises: 3170980dd69c
Create Date: 2026-03-09

"""
from typing import Sequence, Union
from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '3170980dd69c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO llm_model (id, name, model_type, description, model_vendor, token_limit, output_token_limit, prompt_1k_token_usd, completion_1k_token_usd)
        VALUES
        ('claude-sonnet-4-6', 'Claude Sonnet 4.6', 'CHAT', 'This is an improved version of Claude Sonnet 4 with better reasoning. Good for complex analysis, coding, and creative writing. Similar pricing to Claude Sonnet 4.', 'ANTHROPIC', 200000, 64000, 0.003, 0.015),
        ('claude-opus-4-6', 'Claude Opus 4.6', 'CHAT', 'This is a more capable model than Claude Sonnet 4 for highly complex tasks. Best for demanding reasoning and analysis. Higher cost than Claude Sonnet models.', 'ANTHROPIC', 200000, 64000, 0.005, 0.025)
    """)


def downgrade() -> None:
    op.execute("DELETE FROM llm_model WHERE id IN ('claude-sonnet-4-6', 'claude-opus-4-6')")
