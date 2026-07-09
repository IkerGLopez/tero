"""add_gemini_3_5_flash

Revision ID: b2c3d4e5f6a1
Revises: f7a8b9c0d1e2
Create Date: 2026-07-08 00:00:00.000000

"""
from typing import Sequence, Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a1'
down_revision: Union[str, None] = 'f7a8b9c0d1e2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO llm_model (id, name, description, token_limit, output_token_limit, prompt_1k_token_usd, completion_1k_token_usd, model_type, model_vendor)
        VALUES ('gemini-3.5-flash', 'Gemini 3.5 Flash', 'Fast and efficient Google model with 1M context window.', 1048576, 65536, 0.0015, 0.009, 'CHAT', 'GOOGLE')
        ON CONFLICT (id) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("DELETE FROM llm_model WHERE id = 'gemini-3.5-flash'")
