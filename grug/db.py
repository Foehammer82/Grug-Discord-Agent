"""
Database configuration and initialization for the Grug application.

This module manages all database-related functionality including:
- PostgreSQL connection setup and management
- SQLAlchemy async engine and session configuration
- LangGraph memory persistence (store and checkpointer)
- Database schema initialization and migrations
- Platform-specific event loop policies

The module provides singletons for database access and an initialization
function that should be called during application startup. It supports
the agent's memory persistence needs through LangGraph's PostgreSQL adapters.

Dependencies:
- SQLAlchemy for ORM and async database access
- Alembic for database migrations
- psycopg for PostgreSQL driver functionality
- LangGraph for agent memory persistence
"""

import asyncio
import subprocess  # nosec B404
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres import AsyncPostgresStore
from loguru import logger
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from grug.settings import settings

# Set the event loop policy for Windows
# NOTE: https://youtrack.jetbrains.com/issue/PY-57667/Asyncio-support-for-the-debugger-EXPERIMENTAL-FEATURE
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# Database engine singleton
sqa_async_engine = create_async_engine(
    url=settings.postgres_dsn,
    echo=False,
    future=True,
    pool_size=10,
    max_overflow=20,
)

# Database session factory singleton
sqa_async_session_factory = async_sessionmaker(bind=sqa_async_engine, class_=AsyncSession, expire_on_commit=False)


@asynccontextmanager
async def langgraph_memory() -> AsyncGenerator[tuple[AsyncPostgresStore, AsyncPostgresSaver], Any]:
    """Creates and yields LangGraph memory components for PostgreSQL.

    Sets up a PostgreSQL connection pool, ensures the 'genai' schema exists,
    and configures the LangGraph store and checkpointer components for memory persistence.
    The connection pool is automatically closed when the context manager exits.

    Yields:
        tuple[AsyncPostgresStore, AsyncPostgresSaver]: A tuple containing:
            - store: For long-term memory persistence
            - checkpointer: For short-term memory persistence

    Raises:
        PostgresError: If there are issues connecting to the database
    """
    # Create the connection pool
    async_connection_pool = AsyncConnectionPool(
        conninfo=settings.postgres_dsn.replace("+psycopg", ""),
        open=False,
        max_size=20,
        kwargs={
            "autocommit": True,
            "prepare_threshold": 0,
            "row_factory": dict_row,
            "options": "-c search_path=genai",
        },
    )
    try:
        await async_connection_pool.open()

        # Create the db schema for the scheduler
        async with async_connection_pool.connection() as conn:
            await conn.execute("CREATE SCHEMA IF NOT EXISTS genai")

        # Configure `store` and `checkpointer` for long-term and short-term memory
        # (Ref: https://langchain-ai.github.io/langgraphjs/concepts/memory/#what-is-memory)
        langgraph_db_memory_store = AsyncPostgresStore(async_connection_pool)
        await langgraph_db_memory_store.setup()
        langgraph_db_checkpointer = AsyncPostgresSaver(async_connection_pool)
        await langgraph_db_checkpointer.setup()

        yield langgraph_db_memory_store, langgraph_db_checkpointer

    # Close the connection pool after use
    finally:
        await async_connection_pool.close()


def run_db_migrations():
    result = subprocess.run(  # nosec B607, B603
        ["alembic", "upgrade", "head"],
        cwd=settings.root_dir.absolute(),
        capture_output=True,
        text=True,
    )
    logger.info(result.stdout)
    logger.info(result.stderr)
    logger.info("Database initialized [alembic upgrade head].")
