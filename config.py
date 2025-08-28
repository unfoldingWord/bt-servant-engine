from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path


class Config(BaseSettings):
    # Model settings
    model_config = SettingsConfigDict(env_file=".env")

    OPENAI_API_KEY: str = Field(...)
    META_VERIFY_TOKEN: str = Field(...)
    META_WHATSAPP_TOKEN: str = Field(...)
    META_PHONE_NUMBER_ID: str = Field(...)
    META_APP_SECRET: str = Field(...)
    FACEBOOK_USER_AGENT: str = Field(...)
    IN_META_SANDBOX_MODE: bool = Field(default=False)
    META_SANDBOX_PHONE_NUMBER: str = Field(default="11111111")
    MESSAGE_AGE_CUTOFF_IN_SECONDS: int = Field(default=3600)
    BASE_URL: str = Field(...)
    BT_SERVANT_LOG_LEVEL: str = Field(default="info")
    MAX_META_TEXT_LENGTH: int = Field(default=4096)
    # Admin API token for protecting CRUD endpoints
    ADMIN_API_TOKEN: str | None = Field(default=None)
    # Enable admin auth for protected endpoints (default False for local/dev tests)
    ENABLE_ADMIN_AUTH: bool = Field(default=False)

    # Optional with default value
    DATA_DIR: Path = Field(default=Path("/data"))


# Create a single instance to import elsewhere
# mypy cannot infer environment-based initialization for BaseSettings
config = Config()  # type: ignore[call-arg]
