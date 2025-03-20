import contextlib

import anyio
from loguru import logger

from grug.db import run_db_migrations
from grug.discord_client import discord_client
from grug.scheduler import start_scheduler
from grug.settings import settings

# TODO: evaluate llm caching: https://python.langchain.com/api_reference/community/cache.html


# noinspection PyTypeChecker
async def main():
    """Main application entrypoint."""
    if not settings.discord_token:
        raise ValueError("`DISCORD_TOKEN` env variable is required to run the Grug Discord Agent.")
    if not settings.openai_api_key:
        raise ValueError("`OPENAI_API_KEY` env variable is required to run the Grug Discord Agent.")

    logger.info("Starting Grug...")

    run_db_migrations()

    async with anyio.create_task_group() as tg:
        tg.start_soon(discord_client.start, settings.discord_token.get_secret_value())
        tg.start_soon(start_scheduler)

    logger.info("Grug has shut down...")


def run_main():
    with contextlib.suppress(KeyboardInterrupt):
        anyio.run(main)

    logger.info("Shutting down Grug...")


if __name__ == "__main__":
    run_main()
