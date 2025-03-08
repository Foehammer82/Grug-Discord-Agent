import os
from pathlib import Path

from pydantic import Field, PostgresDsn, SecretStr, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT_DIR = Path(__file__).parent.parent.resolve()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(_ROOT_DIR / "config" / "secrets.env",),
        extra="ignore",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    discord_token: SecretStr
    openai_api_key: SecretStr

    # Discord Settings
    discord_enable_voice_client: bool = True

    # AI Base Agent Settings
    ai_name: str = "Grug"
    ai_openai_model: str = "gpt-4o-mini"
    ai_instructions: str = "\n".join(
        [
            "- You should ALWAYS talk as though you are a barbarian orc with low intelligence but high charisma.",
        ]
    )

    # AI Image Settings
    ai_image_generation_enabled: bool = True
    ai_image_daily_generation_limit: int | None = Field(
        default=25, description="The daily limit of image generations. If None, there is no limit."
    )
    ai_image_default_size: str = "1024x1024"
    ai_image_default_quality: str = "standard"
    ai_image_default_model: str = "dall-e-3"

    # Database Settings
    postgres_user: str = "postgres"
    postgres_password: SecretStr = SecretStr("postgres")
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "postgres"

    # TTS Settings
    tts_enabled: bool = True
    tts_f5_host: str = "localhost"
    tts_f5_port: int = 7860
    tts_voice: str = "grug"  # TODO: validate that it exists in the voices directory
    tts_remove_silence: bool = Field(
        default=False,
        description="The model tends to produce silences, especially on longer audio. We can manually remove silences if needed. Note that this is an experimental feature and may produce strange results. This will also increase generation time.",
    )
    tts_crossroad_duration_slider: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Set the duration of the cross-fade between audio clips.",
    )
    tts_nfe_slider: int = Field(
        default=32,
        ge=4,
        le=64,
        description="Set the number of denoising steps.",
    )
    tts_speed_slider: float = Field(
        default=1.0,
        ge=0.3,
        le=2.0,
        description="Adjust the speed of the audio.",
    )

    @computed_field
    @property
    def root_dir(self) -> Path:
        """Get the root directory of the project."""
        return _ROOT_DIR

    @computed_field
    @property
    def postgres_dsn(self) -> str:
        """Get the Postgres DSN."""
        return str(
            PostgresDsn.build(
                scheme="postgresql+psycopg",
                host=self.postgres_host,
                port=self.postgres_port,
                username=self.postgres_user,
                password=self.postgres_password.get_secret_value(),
                path=self.postgres_db,
            )
        )


settings = Settings()

# used to set the environment variables for the OpenAI API key for local development
os.environ["OPENAI_API_KEY"] = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
