"""increase_recursion_limit

Revision ID: a7808316b52b
Revises: b3f7a91d2c4e
Create Date: 2026-06-25 14:03:07.419949

"""
import sqlalchemy as sa
import sqlmodel
from typing import Sequence, Union
from alembic import op


# revision identifiers, used by Alembic.
revision: str = 'a7808316b52b'
down_revision: Union[str, None] = 'b3f7a91d2c4e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE agent DROP CONSTRAINT IF EXISTS agent_recursion_limit_range"
    )
    op.create_check_constraint(
        "agent_recursion_limit_range",
        "agent",
        "recursion_limit >= 20 AND recursion_limit <= 130",
    )


def downgrade() -> None:
    op.execute(
        "ALTER TABLE agent DROP CONSTRAINT IF EXISTS agent_recursion_limit_range"
    )
    # Cap rows that exceed the old max (100) before recreating the constraint
    connection = op.get_bind()
    rows = connection.execute(
        sa.text("SELECT id, recursion_limit FROM agent WHERE recursion_limit > 100")
    ).fetchall()
    if rows:
        # Log each row being capped via RAISE NOTICE (must use DO block)
        ids = ", ".join(str(r[0]) for r in rows)
        limits = ", ".join(str(r[1]) for r in rows)
        op.execute(
            f"DO $$ BEGIN RAISE NOTICE 'Capping agent(s) {ids}: recursion_limit {limits} → 100'; END $$;"
        )
        op.execute("UPDATE agent SET recursion_limit = 100 WHERE recursion_limit > 100")
    op.create_check_constraint(
        "agent_recursion_limit_range",
        "agent",
        "recursion_limit >= 20 AND recursion_limit <= 100",
    )
