"""add_gemini_3_1_pro_preview

Revision ID: c3d4e5f6a1b2
Revises: b2c3d4e5f6a1
Create Date: 2026-07-08 00:00:00.000000

NOTE: Pricing is tiered by prompt length.
Stored values reflect the <= 200K token tier (input $2.00/1M, output $12.00/1M).
For prompts > 200K tokens: input $4.00/1M, output $18.00/1M.
"""
from typing import Sequence, Union
from alembic import op

# revision identifiers, used by Alembic.
revision: str = 'c3d4e5f6a1b2'
down_revision: Union[str, None] = 'b2c3d4e5f6a1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        INSERT INTO llm_model (id, name, description, token_limit, output_token_limit, prompt_1k_token_usd, completion_1k_token_usd, model_type, model_vendor)
        VALUES ('gemini-3.1-pro', 'Gemini 3.1 Pro', 'Advanced Google model with 1M context. Tiered pricing: input $2.00/1M (<=200K) or $4.00/1M (>200K); output $12.00/1M (<=200K) or $18.00/1M (>200K). Stored price reflects <=200K tier.', 1048576, 65536, 0.002, 0.012, 'CHAT', 'GOOGLE')
        ON CONFLICT (id) DO NOTHING
    """)


def downgrade() -> None:
    op.execute("DELETE FROM llm_model WHERE id = 'gemini-3.1-pro'")
