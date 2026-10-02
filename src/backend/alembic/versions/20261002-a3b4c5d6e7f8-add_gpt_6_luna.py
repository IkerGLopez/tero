"""add_gpt_6_luna

Revision ID: a3b4c5d6e7f8
Revises: d9e8f7a6b5c4
Create Date: 2026-10-02 00:00:00.000000

NOTE: 1,050,000-token context window (max input ~922,000 tokens, max output
128,000 tokens). Pricing mirrors the eval judge table ($0.10/$0.50 per MTok).
"""
from typing import Sequence, Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'a3b4c5d6e7f8'
down_revision: Union[str, None] = 'd9e8f7a6b5c4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO llm_model (id, name, description, token_limit, output_token_limit, prompt_1k_token_usd, completion_1k_token_usd, model_type, model_vendor) VALUES
        ('gpt-6-luna', 'GPT-6 Luna', 'Fast and cost-efficient GPT-6 model for high-volume tasks such as reranking and evaluation.', 1050000, 128000, 0.0001, 0.0005, 'CHAT', 'OPENAI')
        ON CONFLICT (id) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("DELETE FROM llm_model WHERE id = 'gpt-6-luna'")
