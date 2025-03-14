"""Discord bot interface for the Grug assistant server."""

import logging

import discord.utils
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.graph import CompiledGraph
from loguru import logger

from grug.ai_agent import get_react_agent
from grug.db import langgraph_memory
from grug.discord_voice_client import DiscordVoiceClient
from grug.settings import settings


class DiscordClient(discord.Client):
    """
    Custom Discord client for the Grug assistant server.

    Attributes:
        react_agent (CompiledGraph | None): The ReAct agent used for generating responses.
    """

    react_agent: CompiledGraph | None = None

    def __init__(self):
        """
        Initialize the Discord client with the required intents and logging setup.
        """
        # Define Discord Intents required for the bot session
        intents = discord.Intents.default()
        intents.members = True  # TODO: link to justification for intent

        super().__init__(intents=intents)
        discord.utils.setup_logging(handler=InterceptLogHandler())

    def get_bot_invite_url(self) -> str | None:
        """
        Generate the bot invite URL.

        Returns:
            The invite URL if the bot user is available, otherwise None.
        """
        return (
            f"https://discord.com/api/oauth2/authorize?client_id={self.user.id}&permissions=8&scope=bot"
            if self.user
            else None
        )

    async def on_ready(self):
        """
        Event handler for when the bot is ready.

        Documentation: https://discordpy.readthedocs.io/en/stable/api.html#discord.on_ready
        """
        if not self.react_agent:
            raise ValueError("ReAct agent not Initialized!")

        logger.info(f"Logged in as {self.user} (ID: {self.user.id})")
        logger.info(f"Discord bot invite URL: {self.get_bot_invite_url()}")

    async def on_message(
        self,
        message: discord.Message,
    ):
        """Processes incoming Discord messages and determines how to respond.

        This event handler is automatically called by the Discord.py framework for each
        message received. It processes mentions, direct messages, and replies, invoking
        the ReAct agent to generate appropriate responses or silently updating conversation
        history for context.

        Args:
            message: The Discord message object containing all message data including
                content, author, channel, and other metadata.

        Raises:
            ValueError: If the ReAct agent hasn't been initialized before receiving messages.
        """
        if not self.react_agent:
            raise ValueError("ReAct agent not Initialized!")

        # TODO: make a tool that can search chat history for a given channel.  proposed approach is to have a task
        #       that runs after a pause in the conversation, possibly nightly, that will summarize the last chat and
        #       store that to a vector store.  This will allow the bot to search the chat history for a given channel
        #       and return the most relevant messages.

        # ignore messages from self and all bots
        if message.author == self.user or message.author.bot:
            return

        # get the agent config based on the current message
        agent_config = {
            "configurable": {
                "thread_id": str(message.channel.id),
                "user_id": f"{str(message.guild.id) + '-' if message.guild else ''}{message.author.id}",
            }
        }

        # Respond if message is @mention or DM
        channel_is_text_or_thread = isinstance(message.channel, discord.TextChannel) or isinstance(
            message.channel, discord.Thread
        )
        is_direct_message = isinstance(message.channel, discord.DMChannel)
        is_at_mentioned_message = channel_is_text_or_thread and self.user in message.mentions
        if is_direct_message or is_at_mentioned_message:
            async with message.channel.typing():
                messages: list[BaseMessage] = []

                # Handle replies
                if message_replied_to := message.reference.resolved.content if message.reference else None:
                    messages.append(
                        SystemMessage(
                            f'You previously sent the following message: "{message_replied_to}", assume that that '
                            "you are responding to a reply to that message."
                        )
                    )

                # Add the message that the user sent
                messages.append(HumanMessage(message.content))

                # Invoke the ReAct agent with the messages
                final_state = await self.react_agent.ainvoke(
                    input={"messages": messages},
                    config=agent_config,
                )

                # Send the response to the channel
                await message.channel.send(
                    content=final_state["messages"][-1].content,
                    reference=message if channel_is_text_or_thread else None,
                )

        # Otherwise, add the message to the conversation history without requesting a response
        else:
            await self.react_agent.aupdate_state(
                config=agent_config,
                values={"messages": [HumanMessage(message.content)]},
            )

    async def start(self, token: str, *, reconnect: bool = True) -> None:
        """Start the Discord bot and initialize required components.

        This method initializes the ReAct agent with persistent memory,
        sets up the voice client if enabled, and connects to the Discord API.
        It also handles cleanup during shutdown.

        Args:
            token: The Discord authentication token used to connect to the API.
            reconnect: Whether to automatically attempt reconnecting on connection failures.
                Defaults to True.

        Raises:
            ValueError: If the Discord token is not set in environment variables.
        """
        if not settings.discord_token:
            raise ValueError("Discord bot token not set!")

        # Initialize the ReAct agent
        async with langgraph_memory() as (store, checkpointer):
            self.react_agent = get_react_agent(
                checkpointer=checkpointer,
                store=store,
            )

            # Initialize the Discord voice client if enabled
            if settings.discord_enable_voice_client:
                DiscordVoiceClient(
                    discord_client=self,
                    react_agent=self.react_agent,
                )

            # Start the Discord client
            try:
                await self.login(token)
                await self.connect(reconnect=reconnect)

            # Handle Application Shutdown
            finally:
                # Disconnect from all voice channels
                logger.info("Disconnecting from all voice channels...")
                for vc in self.voice_clients:
                    await vc.disconnect(force=True)

                # Close the Discord client
                logger.info("Closing the Discord client...")
                await self.close()


class InterceptLogHandler(logging.Handler):
    """
    Default log handler from examples in loguru documentation.
    See https://loguru.readthedocs.io/en/stable/overview.html#entirely-compatible-with-standard-logging
    """

    def emit(self, record: logging.LogRecord):
        """Intercept standard logging records."""
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            if frame.f_back:
                frame = frame.f_back
                depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())
