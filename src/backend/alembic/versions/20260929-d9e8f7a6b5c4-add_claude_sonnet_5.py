"""add_claude_sonnet_5

Revision ID: d9e8f7a6b5c4
Revises: 9c4e2a7f1b8d
Create Date: 2026-09-29 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'd9e8f7a6b5c4'
down_revision: Union[str, None] = '9c4e2a7f1b8d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO llm_model (id, name, description, token_limit, output_token_limit, prompt_1k_token_usd, completion_1k_token_usd, model_type, model_vendor) VALUES
        ('claude-sonnet-5', 'Claude Sonnet 5', 'The most capable Sonnet model from Anthropic, built for coding, agents, and professional work at scale.', 1000000, 128000, 0.002, 0.010, 'CHAT', 'ANTHROPIC')
        ON CONFLICT (id) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("DELETE FROM llm_model WHERE id = 'claude-sonnet-5'")
