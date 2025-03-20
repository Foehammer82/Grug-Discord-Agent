"""Discord bot interface for the Grug assistant server."""

import array
import asyncio
import audioop
import concurrent.futures
import logging
import time
from collections import defaultdict, deque
from datetime import UTC, datetime
from typing import Any, Awaitable, Deque, Final, Optional, TypedDict, TypeVar

import discord.utils
import speech_recognition
from discord import FFmpegPCMAudio
from discord.ext import voice_recv
from discord.ext.voice_recv import AudioSink, SilencePacket, VoiceData, VoiceRecvClient
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.graph import CompiledGraph
from loguru import logger
from pydantic import BaseModel
from rapidfuzz import fuzz
from speech_recognition.recognizers.whisper_api import openai as speech_recognition_openai
from tembo_pgmq_python import async_queue
from tembo_pgmq_python import queue as sync_queue

from grug.ai_agent import get_react_agent
from grug.ai_tts_client import get_tts
from grug.db import langgraph_memory
from grug.settings import settings


class _RespondingTo(BaseModel):
    user_id: int
    last_message_timestamp: datetime


class _StreamData(TypedDict):
    stopper: Optional[Any]
    recognizer: speech_recognition.Recognizer
    buffer: array.array[int]


class _DiscordSRAudioSource(speech_recognition.AudioSource):
    little_endian: Final[bool] = True
    SAMPLE_RATE: Final[int] = 48_000
    SAMPLE_WIDTH: Final[int] = 2
    CHANNELS: Final[int] = 2
    CHUNK: Final[int] = 960

    # noinspection PyMissingConstructor
    def __init__(self, buffer: array.array[int], read_timeout: int = 10):
        self.read_timeout = read_timeout
        self.buffer = buffer
        self._entered: bool = False

    @property
    def stream(self):
        return self

    def __enter__(self):
        if self._entered:
            logger.warning("Already entered sr audio source")
        self._entered = True
        return self

    def __exit__(self, *exc) -> None:
        self._entered = False
        if any(exc):
            logger.exception("Error closing sr audio source")

    def read(self, size: int) -> bytes:
        for _ in range(self.read_timeout):
            if len(self.buffer) < size * self.CHANNELS:
                time.sleep(0.01)
            else:
                break
        else:
            if len(self.buffer) <= 100:
                return b""

        chunk_size = size * self.CHANNELS
        audio_chunk = self.buffer[:chunk_size].tobytes()
        del self.buffer[: min(chunk_size, len(audio_chunk))]
        audio_chunk = audioop.tomono(audio_chunk, 2, 1, 1)
        return audio_chunk

    def close(self) -> None:
        self.buffer.clear()


class _SpeechRecognitionSink(AudioSink):
    """
    Speech recognition sink for Discord voice channels.

    source: https://github.com/imayhaveborkedit/discord-ext-voice-recv/blob/main/discord/ext/voice_recv/extras/speechrecognition.py
    """

    _stream_data: defaultdict[int, _StreamData] = defaultdict(
        lambda: _StreamData(stopper=None, recognizer=speech_recognition.Recognizer(), buffer=array.array("B"))
    )

    def __init__(self, discord_channel: discord.VoiceChannel):
        super().__init__(None)
        self.discord_channel: discord.VoiceChannel = discord_channel

        self.queue = sync_queue.PGMQueue(
            host=settings.postgres_host,
            port=settings.postgres_port,
            username=settings.postgres_user,
            password=settings.postgres_password.get_secret_value(),
            database=settings.postgres_db,
        )

        # Create a queue for the voice channel if it doesn't exist.
        if str(self.discord_channel.id) not in self.queue.list_queues():
            self.queue.create_queue(str(self.discord_channel.id))

    def _await(self, coro: Awaitable[TypeVar]) -> concurrent.futures.Future[TypeVar]:
        assert self.client is not None
        return asyncio.run_coroutine_threadsafe(coro, self.client.loop)

    def wants_opus(self) -> bool:
        return False

    def write(self, user: Optional[discord.User], data: VoiceData) -> None:
        # Ignore silence packets and packets from users we don't have data for
        if isinstance(data.packet, SilencePacket) or user is None:
            return

        sdata = self._stream_data[user.id]
        sdata["buffer"].extend(data.pcm)

        if not sdata["stopper"]:
            sdata["stopper"] = sdata["recognizer"].listen_in_background(
                source=_DiscordSRAudioSource(sdata["buffer"]),
                callback=self.background_listener(user),
                phrase_time_limit=10,
            )

    def background_listener(self, user: discord.User):
        def callback(_recognizer: speech_recognition.Recognizer, _audio: speech_recognition.AudioData):
            # Don't process empty audio data or audio data that is too small
            if _audio.frame_data == b"" or len(bytes(_audio.frame_data)) < 10000:
                return None

            # Get the text from the audio data
            text_output = None
            try:
                text_output = speech_recognition_openai.recognize(_recognizer, _audio)
            except speech_recognition.UnknownValueError:
                logger.debug("Bad speech chunk")

            # WEIRDEST BUG EVER: for some reason whisper keeps getting the word "you" from the recognizer, so
            #                    we'll just ignore any text segments that are just "you"
            if text_output and text_output.lower() != "you":
                self.queue.send(
                    str(self.discord_channel.id),
                    {
                        "user_id": user.id,
                        "message_timestamp": datetime.now(tz=UTC).isoformat(),
                        "message": text_output,
                    },
                )

        return callback

    def cleanup(self) -> None:
        for user_id in tuple(self._stream_data.keys()):
            self._drop(user_id)

    def _drop(self, user_id: int) -> None:
        if user_id in self._stream_data:
            data = self._stream_data.pop(user_id)
            stopper = data.get("stopper")
            if stopper:
                stopper()

            buffer = data.get("buffer")
            if buffer:
                # arrays don't have a clear function
                del buffer[:]


class DiscordVoiceClient:
    def __init__(self, discord_client: discord.Client, react_agent: CompiledGraph):
        if not react_agent:
            raise ValueError("ReAct agent not Initialized")

        self.discord_client = discord_client
        self.react_agent = react_agent

        # Initialize the background voice responder tasks set to keep track of running tasks
        self.background_voice_responder_tasks: set = set()

        # Register the on_voice_state_update event
        self.discord_client.event(self.on_voice_state_update)

    async def get_bot_introduction_text(self, voice_channel: discord.VoiceState) -> str:
        """Get the bot introduction text for a voice channel."""
        final_state = await self.react_agent.ainvoke(
            {
                "messages": [
                    SystemMessage(
                        content=(
                            "- When introducing yourself, give a quick summary of who you are. \n"
                            f"- Make sure to let the user know that you are listening in {voice_channel.channel.name}"
                            f" voice channel on the {voice_channel.channel.guild.name} server. \n"
                        )
                    ),
                    HumanMessage(content="Introduce yourself!"),
                ]
            },
            config={
                "configurable": {
                    "thread_id": str(voice_channel.channel.id),
                }
            },
        )
        return final_state["messages"][-1].content

    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ):
        # TODO: store this in the DB so that it can be configured for each server
        bot_voice_channel_id = 1049728769541283883

        # Ignore bot users
        if member.bot:
            return

        # If the user joined the bot voice channel
        if after.channel is not None and after.channel.id == bot_voice_channel_id and before.channel is None:
            logger.info(f"{member.display_name} joined {after.channel.name}")

            # Notify the user that the bot is listening
            await after.channel.send(
                content=(
                    f"{await self.get_bot_introduction_text(after)}\n\n"
                    f'*You can talk to me by saying "Hey, {settings.ai_name.title()}"*'
                ),
            )

            # If the bot is not currently in the voice channel, connect to the voice channel
            if self.discord_client.user not in after.channel.members:
                logger.info(f"Connecting to {after.channel.name}")
                voice_channel = await after.channel.connect(cls=voice_recv.VoiceRecvClient)
                voice_channel.listen(_SpeechRecognitionSink(discord_channel=after.channel))

                # Start the voice responder agent
                voice_responder_task = asyncio.create_task(self._listen_to_voice_channel(voice_channel))
                self.background_voice_responder_tasks.add(voice_responder_task)
                voice_responder_task.add_done_callback(self.background_voice_responder_tasks.discard)

        # If the user left the bot voice channel
        elif before.channel.id == bot_voice_channel_id and before.channel is not None:
            logger.info(f"{member.display_name} left {before.channel.name}")

            # If there are no members in the voice channel and the bot is in the voice channel, disconnect from
            # the voice channel
            if len(before.channel.members) <= 1 and self.discord_client.user in before.channel.members:
                logger.info(f"No members in {before.channel.name}, disconnecting...")
                voice_channel = next(
                    (vc for vc in self.discord_client.voice_clients if vc.channel == before.channel), None
                )
                await voice_channel.disconnect(force=True)

    async def _listen_to_voice_channel(self, voice_channel: VoiceRecvClient):
        """A looping task that listens for messages in a voice channel and responds to them."""

        # Initialize the queue
        queue = async_queue.PGMQueue(
            host=settings.postgres_host,
            port=settings.postgres_port,
            username=settings.postgres_user,
            password=settings.postgres_password.get_secret_value(),
            database=settings.postgres_db,
        )
        await queue.init()
        await queue.purge(str(voice_channel.channel.id))  # Start with a fresh queue when the bot joins

        while voice_channel.is_connected():
            if not self.react_agent:
                raise ValueError("ReAct agent not Initialized!")

            responding_to: Optional[_RespondingTo] = None
            message_buffer: Deque = deque(maxlen=100)
            poll_interval_seconds = 0.1
            end_of_statement_seconds = 1
            while True:
                # Read messages in batches off the queue
                for message in await queue.read_batch(
                    queue=str(voice_channel.channel.id),
                    vt=30,
                    batch_size=5,
                ):
                    # Delete the message from the queue
                    await queue.delete(str(voice_channel.channel.id), message.msg_id)

                    # if currently responding to a message, add the user messages to the buffer
                    if responding_to and message.message.get("user_id") == responding_to.user_id:
                        message_buffer.append(message.message.get("message"))

                    # Check if the bot was called by name
                    elif (
                        fuzz.partial_ratio(
                            s1=f"hey, {settings.ai_name.lower()}",
                            s2=message.message.get("message").lower(),
                        )
                        > 80
                    ):
                        logger.info(f"Bot was called by name by {message.message.get('user_id')}")

                        # Play the boop sound effect when the bot is called by name
                        voice_channel.play(
                            FFmpegPCMAudio((settings.root_dir / "assets/sound_effects/boop.wav").as_posix())
                        )

                        message_buffer.clear()
                        message_buffer.append(message.message.get("message"))
                        responding_to = _RespondingTo(
                            user_id=message.message.get("user_id"),
                            last_message_timestamp=message.enqueued_at,
                        )

                # Check to see if the bot should respond to its summons
                if (
                    responding_to
                    and (datetime.now(tz=UTC) - responding_to.last_message_timestamp).seconds > end_of_statement_seconds
                ):
                    # Respond to the message
                    final_state = await self.react_agent.ainvoke(
                        {
                            "messages": [
                                SystemMessage(
                                    content=(
                                        "- you are are responding to a message sent by a user in voice chat. \n"
                                        "- remember that the speach to text is not perfect, so there may be some errors in the text. \n"
                                        "- Do your best to assume what the users meant to say, but DO NOT try to make sense of gibberish. \n"
                                        "- If you are unsure what the user said, ask them to clarify. \n"
                                        "- DO NOT correct the user about your name or who you are, assume they misspoke and ignore it. \n"
                                    )
                                ),
                                HumanMessage(content=" ".join(list(message_buffer))),
                            ]
                        },
                        config={
                            "configurable": {
                                "thread_id": str(voice_channel.channel.id),
                                "user_id": f"{str(voice_channel.guild.id)}-{responding_to.user_id}",
                            }
                        },
                    )

                    response_text = final_state["messages"][-1].content
                    await voice_channel.channel.send(content=response_text)
                    voice_channel.play(FFmpegPCMAudio(get_tts(response_text).as_posix()))

                    logger.info(f"Responded to {responding_to.user_id} for request: {' '.join(list(message_buffer))}")

                    # Reset the responding_to object
                    responding_to = None

                # Wait for the poll interval
                await asyncio.sleep(poll_interval_seconds)

        logger.info(f"Voice channel {voice_channel.channel.name} disconnected, stopping voice responder...")


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
