"""
Tests for DB connection pool configuration (fix-db-pool-saturation).

REQ-001: Settings accept DB pool environment variables.
REQ-002: Engine creation uses configured pool values.
"""

import asyncio
import importlib
import os

import pytest
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession

from tero.core.env import Settings


# Base env vars required for Settings to instantiate without a .env file.
# These cover every non-optional field in the Settings class.
_MINIMAL_ENV: dict[str, str] = {
    'SECRET_ENCRYPTION_KEY': 'dGVzdC1rZXktZm9yLXRlc3RpbmctcHVycG9zZXM=',
    'FRONTEND_URL': 'http://localhost:5173',
    'OPENID_URL': 'http://localhost:8080',
    'OPENID_CLIENT_ID': 'test-client',
    'OPENID_SCOPE': 'openid',
    'CONTACT_EMAIL': 'test@test.com',
    'TEMPERATURES': 'PRECISE:0',
    'MONTHLY_USD_LIMIT_DEFAULT': '10',
    'INTERNAL_GENERATOR_MODEL': 'gpt-4o',
    'INTERNAL_GENERATOR_TEMPERATURE': '0.7',
    'AGENT_BASIC_MODELS': 'gpt-4o',
    'DEFAULT_AGENT_NAME': 'Test',
    'EMBEDDING_MODEL': 'text-embedding-3-small',
    'EMBEDDING_COST_PER_1K_TOKENS': '0.0001',
    'TRANSCRIPTION_MODEL': 'whisper',
    'AWS_REGION': 'us-east-1',
    'AWS_MODEL_ID_MAPPING': 'm1:d1',
    'GOOGLE_MODEL_ID_MAPPING': 'm1:d1',
    'OPENAI_MODEL_ID_MAPPING': 'm1:d1',
    'DOCS_TOOL_CHUNK_SIZE': '1000',
    'DOCS_TOOL_CHUNK_OVERLAP': '100',
    'DOCS_TOOL_RETRIEVE_TOP': '5',
    'DOCS_TOOL_DESCRIPTION_CHUNK_SIZE': '10000',
    'DOCS_TOOL_DESCRIPTION_CHUNK_OVERLAP': '100',
    'TOOL_OAUTH_TOKEN_TTL_MINUTES': '43200',
    'TOOL_OAUTH_STATE_TTL_MINUTES': '10',
    'MCP_TOOL_OAUTH_CLIENT_REGISTRATION_TTL_MINUTES': '259200',
    'WEB_TOOL_TAVILY_COST_PER_1K_CREDITS_USD': '1.0',
    'WEB_TOOL_GOOGLE_COST_PER_1K_SEARCHES_USD': '1.0',
    'BROWSER_TOOL_PLAYWRIGHT_MCP_URL': 'http://localhost:8931',
    'BROWSER_TOOL_PLAYWRIGHT_OUTPUT_DIR': 'var/output',
    # Fields with mode='before' validators that need string values
    'AZURE_MODEL_DEPLOYMENTS': 'dummy:dummy',
    'VLLM_MODEL_ID_MAPPING': '',
    'ALLOWED_USERS': '',
    'AZURE_ENDPOINTS': '',
    'AZURE_API_KEYS': '',
    'VLLM_URLS': '',
    'VLLM_API_KEYS': '',
}


@pytest.fixture
def base_env(monkeypatch):
    """Set up all required env vars so Settings() can be instantiated."""
    for key, value in _MINIMAL_ENV.items():
        monkeypatch.setenv(key, value)
    # Remove pool vars to test defaults
    for pool_key in ('DB_POOL_SIZE', 'DB_MAX_OVERFLOW', 'DB_POOL_TIMEOUT'):
        monkeypatch.delenv(pool_key, raising=False)


# ---------------------------------------------------------------------------
# REQ-001: Settings accept DB pool environment variables
# ---------------------------------------------------------------------------


def test_settings_defaults(base_env):
    """1.1 Pool fields default to 20/30/60 when env vars are absent.

    REQ-001 Scenario: Pool vars default when absent.
    """
    settings = Settings(db_url='postgresql+psycopg://user:pass@localhost/test', internal_generator_reasoning_effort='medium')
    assert settings.db_pool_size == 20
    assert settings.db_max_overflow == 30
    assert settings.db_pool_timeout == 60


def test_settings_override_from_env(base_env, monkeypatch):
    """1.2 Pool fields load from environment variables when present.

    REQ-001 Scenario: Pool vars loaded from .env.
    """
    monkeypatch.setenv('DB_POOL_SIZE', '25')
    monkeypatch.setenv('DB_MAX_OVERFLOW', '40')
    monkeypatch.setenv('DB_POOL_TIMEOUT', '45')

    settings = Settings(db_url='postgresql+psycopg://user:pass@localhost/test', internal_generator_reasoning_effort='medium')
    assert settings.db_pool_size == 25
    assert settings.db_max_overflow == 40
    assert settings.db_pool_timeout == 45


def test_pool_size_must_be_at_least_1(base_env):
    """db_pool_size of 0 or negative raises ValueError."""
    with pytest.raises(ValueError, match='db_pool_size must be >= 1'):
        Settings(db_url='postgresql+psycopg://user:pass@localhost/test', db_pool_size=0, internal_generator_reasoning_effort='medium')
    with pytest.raises(ValueError, match='db_pool_size must be >= 1'):
        Settings(db_url='postgresql+psycopg://user:pass@localhost/test', db_pool_size=-1, internal_generator_reasoning_effort='medium')


def test_max_overflow_must_be_non_negative(base_env):
    """db_max_overflow below 0 raises ValueError."""
    with pytest.raises(ValueError, match='db_max_overflow must be >= 0'):
        Settings(db_url='postgresql+psycopg://user:pass@localhost/test', db_max_overflow=-1, internal_generator_reasoning_effort='medium')


def test_pool_timeout_must_be_at_least_1(base_env):
    """db_pool_timeout of 0 or negative raises ValueError."""
    with pytest.raises(ValueError, match='db_pool_timeout must be >= 1'):
        Settings(db_url='postgresql+psycopg://user:pass@localhost/test', db_pool_timeout=0, internal_generator_reasoning_effort='medium')
    with pytest.raises(ValueError, match='db_pool_timeout must be >= 1'):
        Settings(db_url='postgresql+psycopg://user:pass@localhost/test', db_pool_timeout=-5, internal_generator_reasoning_effort='medium')


# ---------------------------------------------------------------------------
# REQ-002: Engine creation uses configured pool values
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_engine_receives_pool_params(base_env, postgres_container):
    """1.3 Engine creation uses pool_size/max_overflow/pool_timeout from Settings.

    REQ-002 Scenario: Pool config passes through to engine.
    """
    url = postgres_container.get_connection_url()
    settings = Settings(db_url=url, db_pool_size=5, db_max_overflow=2, db_pool_timeout=45, internal_generator_reasoning_effort='medium')

    engine = create_async_engine(
        url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
    )

    # Accessing private pool attrs — no public API for configured
    # pool_size/max_overflow/timeout. Monitored in test suite.
    assert engine.pool._pool.maxsize == 5, 'pool_size not wired correctly'
    assert engine.pool._max_overflow == 2, 'max_overflow not wired correctly'
    assert engine.pool._timeout == 45, 'pool_timeout not wired correctly'

    await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_sessions_exhaustion(base_env, postgres_container):
    """1.4 Running N+1 sessions over pool capacity triggers exhaustion.

    REQ-002 Scenario: Over-capacity sessions raise timeout/pool-exhaustion error.
    Uses pool_size=5 + max_overflow=3 = 8 total, runs 9 concurrent sessions.
    Holders occupy all 8 slots via Event barrier. The 9th session (N+1)
    is launched after confirming all slots are filled and must raise an error.
    """
    url = postgres_container.get_connection_url()
    settings = Settings(db_url=url, db_pool_size=5, db_max_overflow=3, db_pool_timeout=2, internal_generator_reasoning_effort='medium')

    engine = create_async_engine(
        url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
    )

    release = asyncio.Event()
    acquired = asyncio.Event()
    total_capacity = settings.db_pool_size + settings.db_max_overflow  # 8
    acquired_count = 0

    async def hold_session():
        nonlocal acquired_count
        async with AsyncSession(engine, expire_on_commit=False) as session:
            await session.exec_driver_sql('SELECT 1')
            acquired_count += 1
            if acquired_count >= total_capacity:
                acquired.set()
            await release.wait()     # block until released — keeps connection open

    # Launch holder tasks as background tasks (not in gather — Fix 1)
    holder_tasks = [asyncio.create_task(hold_session()) for _ in range(total_capacity)]

    try:
        # Wait for all pool capacity slots to fill (Fix 3: acquired event used here)
        await asyncio.wait_for(acquired.wait(), timeout=10)

        # N+1th session — must raise timeout/pool-exhaustion error
        with pytest.raises(Exception):
            async with AsyncSession(engine, expire_on_commit=False) as session:
                await session.exec_driver_sql('SELECT 1')
    finally:
        release.set()  # unblock held sessions before cleanup
        await asyncio.gather(*holder_tasks, return_exceptions=True)
        await engine.dispose()


@pytest.mark.asyncio
async def test_concurrent_sessions_production_defaults(base_env, postgres_container):
    """1.5 15 concurrent sessions complete with production defaults (20/30/60).

    REQ-002 Scenario: Concurrent queries do not exhaust pool under new defaults.
    Per spec: pool configured with defaults (20/30/60) and 15 concurrent
    get_db() sessions must all complete within pool_timeout with zero errors.
    """
    url = postgres_container.get_connection_url()
    settings = Settings(db_url=url, internal_generator_reasoning_effort='medium')  # uses defaults: 20/30/60

    engine = create_async_engine(
        url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout,
    )

    async def do_select():
        async with AsyncSession(engine, expire_on_commit=False) as session:
            await session.exec_driver_sql('SELECT 1')

    # 15 sessions within capacity (20 pool + 30 overflow = 50)
    results = await asyncio.gather(*[do_select() for _ in range(15)], return_exceptions=True)

    errors = [r for r in results if isinstance(r, Exception)]
    assert len(errors) == 0, f'Got {len(errors)} pool errors with production defaults: {errors}'

    await engine.dispose()


@pytest.mark.asyncio
async def test_repos_engine_pool_config(base_env, postgres_container, monkeypatch):
    """1.6 repos.engine is created with pool kwargs from env Settings.

    If repos.py drops pool kwargs, no other test catches it because
    integration tests create their own engine from a custom Settings instance.
    This test imports tero.core.repos after controlling env vars to verify
    the module-level engine has the correct pool configuration.
    """
    url = postgres_container.get_connection_url()
    monkeypatch.setenv('DB_URL', url)
    monkeypatch.setenv('DB_POOL_SIZE', '7')
    monkeypatch.setenv('DB_MAX_OVERFLOW', '3')
    monkeypatch.setenv('DB_POOL_TIMEOUT', '42')

    # Force fresh imports so repos.py reads our env vars.
    # env.py creates the Settings singleton at module level; it must be
    # reloaded first so repos.py picks up the monkeypatched env vars.
    import tero.core.env as env_module
    import tero.core.repos as repos_module
    importlib.reload(env_module)
    importlib.reload(repos_module)

    engine = repos_module.engine

    try:
        # Accessing private pool attrs — no public API for configured
        # pool_size/max_overflow/timeout. Monitored in test suite.
        assert engine.pool._pool.maxsize == 7, (
            f'repos.engine pool_size expected 7, got {engine.pool._pool.maxsize}'
        )
        assert engine.pool._max_overflow == 3, (
            f'repos.engine max_overflow expected 3, got {engine.pool._max_overflow}'
        )
        assert engine.pool._timeout == 42, (
            f'repos.engine pool_timeout expected 42, got {engine.pool._timeout}'
        )
    finally:
        await engine.dispose()
