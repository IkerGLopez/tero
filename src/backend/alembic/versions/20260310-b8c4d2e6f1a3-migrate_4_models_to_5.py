"""migrate_4_models_to_5

Revision ID: b8c4d2e6f1a3
Revises: a1b2c3d4e5f6
Create Date: 2026-03-10

"""
from typing import Sequence, Union
from alembic import op


revision: str = 'b8c4d2e6f1a3'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        UPDATE agent
        SET team_id = NULL
        WHERE model_id IN ('gpt-4o', 'gpt-4o-mini')
          AND user_id IS NULL
          AND team_id = 1
    """)

    op.execute("""
        UPDATE agent
        SET model_id = CASE model_id
            WHEN 'gpt-4o' THEN 'gpt-5'
            WHEN 'gpt-4.1' THEN 'gpt-5'
            WHEN 'o4-mini' THEN 'gpt-5-mini'
            WHEN 'gpt-4o-mini' THEN 'gpt-5-mini'
            WHEN 'gpt-4.1-nano' THEN 'gpt-5-nano'
        END,
        reasoning_effort = 'MEDIUM'
        WHERE model_id IN ('gpt-4o', 'gpt-4.1', 'o4-mini', 'gpt-4o-mini', 'gpt-4.1-nano')
    """)

    op.execute("""
        UPDATE evaluator
        SET model_id = CASE model_id
            WHEN 'gpt-4o' THEN 'gpt-5'
            WHEN 'gpt-4.1' THEN 'gpt-5'
            WHEN 'o4-mini' THEN 'gpt-5-mini'
            WHEN 'gpt-4o-mini' THEN 'gpt-5-mini'
            WHEN 'gpt-4.1-nano' THEN 'gpt-5-nano'
        END
        WHERE model_id IN ('gpt-4o', 'gpt-4.1', 'o4-mini', 'gpt-4o-mini', 'gpt-4.1-nano')
    """)

    op.execute("""
        DELETE FROM llm_model
        WHERE id IN ('gpt-4o', 'gpt-4o-mini', 'gpt-4.1', 'gpt-4.1-nano', 'o4-mini')
    """)

    op.execute("""
        UPDATE llm_model
        SET description = CASE id
            WHEN 'gemini-2.5-pro' THEN 'This is an advanced reasoning model, outperforming GPT-5 Mini with a larger context while being more affordable.'
            WHEN 'gemini-2.5-flash' THEN 'This is a fast and efficient model, comparable to GPT-5 Nano, optimized for speed while maintaining high quality responses.'
            WHEN 'gpt-5' THEN 'This is the best reasoning model for coding and complex agentic tasks from OpenAI. Aligned with GPT-5 Mini for most use cases.'
            WHEN 'gpt-5-mini' THEN 'This is a new reasoning model with a good balance between cost and intelligence. Suited for most agent tasks.'
            WHEN 'gpt-5-nano' THEN 'This is a new reasoning model for simpler tasks. Lower cost than GPT-5 Mini.'
        END
        WHERE id IN ('gemini-2.5-pro', 'gemini-2.5-flash', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano')
    """)


def downgrade() -> None:
    op.execute("""
        UPDATE llm_model
        SET description = CASE id
            WHEN 'gemini-2.5-pro' THEN 'This is an advanced reasoning model, outperforming GPT-4o with a larger context while being more affordable.'
            WHEN 'gemini-2.5-flash' THEN 'This is a fast and efficient model, comparable to GPT-4.1 Nano, optimized for speed while maintaining high quality responses.'
            WHEN 'gpt-5' THEN 'This is the best reasoning model for coding and complex agentic tasks from OpenAI. It will replace GPT-4o and GPT-4.1 in the short term.'
            WHEN 'gpt-5-mini' THEN 'This is a new reasoning model that will replace GPT-4o Mini in the short term. It has a good balance between cost and intelligence.'
            WHEN 'gpt-5-nano' THEN 'This is a new reasoning model that will replace GPT-4.1 Nano in the short term.'
        END
        WHERE id IN ('gemini-2.5-pro', 'gemini-2.5-flash', 'gpt-5', 'gpt-5-mini', 'gpt-5-nano')
    """)

    op.execute("""
        INSERT INTO llm_model (id, name, model_type, description, model_vendor, token_limit, output_token_limit, prompt_1k_token_usd, completion_1k_token_usd)
        VALUES
        ('gpt-4o', 'GPT-4o', 'CHAT', 'This is the best model for complex tasks. This is more powerful than GPT-4o Mini, but slower and more expensive.', 'OPENAI', 128000, 4096, 0.005, 0.015),
        ('gpt-4o-mini', 'GPT-4o Mini', 'CHAT', 'This is a good model for simple tasks like summaries, simple questions, etc.', 'OPENAI', 128000, 16384, 0.00015, 0.0006),
        ('gpt-4.1', 'GPT-4.1', 'CHAT', 'This is a new model that will replace GPT-4o in the short term.', 'OPENAI', 1047576, 32768, 0.002, 0.008),
        ('gpt-4.1-nano', 'GPT-4.1 Nano', 'CHAT', 'This is a new model that will replace GPT-4o Mini in the short term.', 'OPENAI', 1047576, 32768, 0.0001, 0.0004),
        ('o4-mini', 'O4 Mini', 'REASONING', 'This is a reasoning model that is good for coding, math and some complex tasks.', 'OPENAI', 200000, 100000, 0.0011, 0.0044)
    """)
