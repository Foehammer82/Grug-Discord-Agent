"""Scheduler for the Grug bot."""

from apscheduler import AsyncScheduler
from apscheduler.datastores.sqlalchemy import SQLAlchemyDataStore
from apscheduler.eventbrokers.asyncpg import AsyncpgEventBroker
from pydantic import PostgresDsn
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from grug.settings import settings

# TODO: deprecated! as soon as ApScheduler releases past 4.0.0a5 we can switch to psycopg for the event broker.
scheduler = AsyncScheduler(
    data_store=SQLAlchemyDataStore(
        engine_or_url=settings.postgres_dsn,
        schema="apscheduler",
    ),
    event_broker=AsyncpgEventBroker.from_async_sqla_engine(
        engine=create_async_engine(
            url=str(
                PostgresDsn.build(
                    scheme="postgresql+asyncpg",
                    host=settings.postgres_host,
                    port=settings.postgres_port,
                    username=settings.postgres_user,
                    password=settings.postgres_password.get_secret_value(),
                    path=settings.postgres_db,
                )
            ),
            echo=False,
            future=True,
        ),
    ),
)
_scheduler_is_started = False


async def start_scheduler(discord_bot_startup_timeout: int = 15):
    # Create the db schema for the scheduler
    async with create_async_engine(url=settings.postgres_dsn, echo=False, future=True).begin() as conn:
        await conn.execute(text("CREATE SCHEMA IF NOT EXISTS apscheduler"))

    # start the scheduler
    async with scheduler:
        await scheduler.run_until_stopped()
