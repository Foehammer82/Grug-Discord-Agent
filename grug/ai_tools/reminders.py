import json
from datetime import UTC, datetime
from typing import Literal, Optional

from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.date import DateTrigger
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import BaseModel, Field, computed_field

from grug.scheduler import scheduler
from grug.settings import settings


class ScheduleResponse(BaseModel):
    """Response model for schedule processing."""

    schedule_prompt: str = Field(..., description="The original schedule request.")
    schedule_type: Literal["cron", "datetime"] | None = Field(None, description="The type of schedule format.")
    schedule_value: str | None = Field(None, description="The formatted schedule value.")
    clarification: Optional[str] = Field(
        None, description="Clarification or feedback on the schedule request required from the user."
    )

    @computed_field
    @property
    def schedule(self) -> str | datetime:
        """Returns the schedule in the appropriate format."""
        # Validation
        if self.clarification:
            raise ValueError(f"Clarification needed: {self.clarification}")
        if not self.schedule_value:
            raise ValueError("No schedule value provided.")

        # Return the schedule in the appropriate format
        if self.schedule_type == "cron":
            return self.schedule_value
        else:
            return datetime.fromisoformat(self.schedule_value)


def get_schedule(schedule_prompt: str) -> ScheduleResponse:
    model = ChatOpenAI(
        model_name="gpt-4o-mini",
        temperature=0,
        max_tokens=None,
        max_retries=2,
        openai_api_key=settings.openai_api_key,
    )

    return model.with_structured_output(ScheduleResponse).invoke(
        [
            SystemMessage(
                json.dumps(
                    [
                        "You will be given a schedule request.",
                        "Identify the type of schedule (e.g., cron, datetime) and convert it to the appropriate format.",
                        "If the request is ambiguous, ask for clarification.",
                        "If it's a recurring schedule, convert to a cron string (e.g., '0 12 * * 1-5' for weekdays at noon).",
                        "If it's a specific or relative time, convert to ISO datetime format.",
                        f"The current UTC datetime in ISO format is {datetime.now(tz=UTC).isoformat()}",
                    ]
                )
            ),
            HumanMessage(schedule_prompt),
        ]
    )


async def send_reminder(
    message: str,
    discord_channel_id: str,
    discord_user_id: str,
    discord_guild_id: str | None,
) -> None:
    from grug.discord_client import discord_client

    # TODO: figure out how to get the original message id so we can reference it in the reminder message
    if not discord_guild_id:
        await discord_client.get_user(int(discord_user_id)).send(message)
        print(f"Sent reminder to user {discord_user_id}: {message}")
    else:
        channel = discord_client.get_guild(int(discord_guild_id)).get_channel(int(discord_channel_id))
        await channel.send(f"Reminder for <@{discord_user_id}>: {message}")


@tool(parse_docstring=True)
async def set_reminder(
    config: RunnableConfig,
    schedule_prompt: str,
    message: str | None = None,
) -> str:
    """
    Sets a reminder for the user.

    Args:
        config (RunnableConfig): The configuration for the reminder.
        schedule_prompt (str): Prompt for when to schedule the reminder for.  This can be a date, time, a relative time (e.g. "in 2 hours"), or a recurring time (e.g. "every day at 3pm").
        message (str): The message to send when the reminder is triggered.  If not provided, a default message will be used.
    """
    logger.info(f"Scheduling reminder with prompt: {schedule_prompt} and message: {message}")

    discord_channel_id = config["metadata"].get("thread_id")
    user_id = config["metadata"].get("user_id")
    if "-" in user_id:
        discord_guild_id = user_id.split("-")[0]
        discord_user_id = user_id.split("-")[1]
    else:
        discord_guild_id = None
        discord_user_id = user_id

    reminder_schedule = get_schedule(schedule_prompt).schedule
    if isinstance(reminder_schedule, str):
        trigger = CronTrigger.from_crontab(reminder_schedule)
    elif isinstance(reminder_schedule, datetime):
        trigger = DateTrigger(run_time=reminder_schedule)
    else:
        raise ValueError("Invalid schedule format.")

    schedule_id = await scheduler.add_schedule(
        func_or_task_id=send_reminder,
        trigger=trigger,
        kwargs={
            "message": message or "This is your reminder!",
            "discord_channel_id": discord_channel_id,
            "discord_user_id": discord_user_id,
            "discord_guild_id": discord_guild_id,
        },
    )

    # TODO: put the reminder into memory store so that it can be looked up later when the user wants to adjust it
    #       or look it up.
    return f"Reminder scheduled with id: {schedule_id}"
