from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Config(BaseSettings):
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    META_VERIFY_TOKEN: str = Field(..., env="META_VERIFY_TOKEN")
    META_WHATSAPP_TOKEN: str = Field(..., env="META_WHATSAPP_TOKEN")
    META_PHONE_NUMBER_ID: str = Field(..., env="META_PHONE_NUMBER_ID")
    META_APP_SECRET: str = Field(..., env="META_APP_SECRET")
    FACEBOOK_USER_AGENT: str = Field(..., env="FACEBOOK_USER_AGENT")
    IN_META_SANDBOX_MODE: bool = Field(default=False, env="IN_META_SANDBOX_MODE")
    META_SANDBOX_PHONE_NUMBER: str = Field(default="11111111", env="META_SANDBOX_PHONE_NUMBER")
    MESSAGE_AGE_CUTOFF_IN_SECONDS: int = Field(default=3600, env="MESSAGE_AGE_CUTOFF_IN_SECONDS")
    BASE_URL: str = Field(..., env="BASE_URL")
    BT_SERVANT_LOG_LEVEL: str = Field(default="info", env="BT_SERVANT_LOG_LEVEL")
    MAX_META_TEXT_LENGTH: int = Field(default=4096, env="MAX_META_TEXT_LENGTH")

    # Optional with default value
    DATA_DIR: Path = Field(default=Path("/data"), env="DATA_DIR")

    class Config:
        env_file = ".env"


# Create a single instance to import elsewhere
config = Config()
