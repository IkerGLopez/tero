"""add-gpt-5-4-and-codex

Revision ID: c4d5e6f7a8b9
Revises: b8c4d2e6f1a3
Create Date: 2026-03-13

"""

from typing import Sequence, Union
from alembic import op


revision: str = 'c4d5e6f7a8b9'
down_revision: Union[str, None] = 'b8c4d2e6f1a3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO llm_model (id, name, model_type, description, model_vendor, token_limit, output_token_limit, prompt_1k_token_usd, completion_1k_token_usd)
        VALUES
        ('gpt-5.1-codex-max', 'GPT-5.1 Codex Max', 'REASONING', 'This is the most capable Codex model for long-running agentic coding. Best for complex codebases and multi-step tasks. Similar pricing to GPT-5.1 Codex.', 'OPENAI', 400000, 128000, 0.00125, 0.01),
        ('gpt-5.4', 'GPT-5.4', 'REASONING', 'This is the most capable reasoning model from OpenAI for professional work. Best for demanding analysis and complex agentic tasks. Higher cost than GPT-5.', 'OPENAI', 1050000, 128000, 0.0025, 0.015)
    """)


def downgrade() -> None:
    op.execute("DELETE FROM llm_model WHERE id IN ('gpt-5.1-codex-max', 'gpt-5.4')")
