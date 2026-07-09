"""add_gpt_5_4_nano

Revision ID: f7a8b9c0d1e2
Revises: a7808316b52b
Create Date: 2026-07-09 00:00:00.000000

NOTE: All numeric values (pricing, token limits) are provisional (TO_BE_CONFIRMED).
Estimated from sibling model gpt-5-nano until official values are available.
"""
from typing import Sequence, Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'f7a8b9c0d1e2'
down_revision: Union[str, None] = 'a7808316b52b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO llm_model (id, name, description, token_limit, output_token_limit, prompt_1k_token_usd, completion_1k_token_usd, model_type, model_vendor)
        VALUES ('gpt-5.4-nano', 'GPT-5.4 Nano', 'Lightweight OpenAI reasoning model. Token limits are provisional (TO_BE_CONFIRMED).', 400000, 128000, 0.0002, 0.00125, 'REASONING', 'OPENAI')
        ON CONFLICT (id) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("DELETE FROM llm_model WHERE id = 'gpt-5.4-nano'")
