"""add_claude_haiku_4_5

Revision ID: b3f7a91d2c4e
Revises: 8e9d8ca4dd0c
Create Date: 2026-06-02 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b3f7a91d2c4e'
down_revision: Union[str, None] = '8e9d8ca4dd0c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO llm_model (id, name, description, token_limit, output_token_limit, prompt_1k_token_usd, completion_1k_token_usd, model_type, model_vendor) VALUES
        ('claude-haiku-4-5', 'Claude Haiku 4.5', 'Fast and affordable Anthropic model optimised for speed. Good for simple tasks and high-throughput use cases.', 200000, 8192, 0.0008, 0.004, 'CHAT', 'ANTHROPIC')
        ON CONFLICT (id) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("DELETE FROM llm_model WHERE id = 'claude-haiku-4-5'")
