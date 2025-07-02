from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path


class Config(BaseSettings):
    OPENAI_API_KEY: str = Field(..., env="OPENAI_API_KEY")
    GROQ_API_KEY: str = Field(..., env="GROQ_API_KEY")
    META_VERIFY_TOKEN: str = Field(..., env="META_VERIFY_TOKEN")
    META_WHATSAPP_TOKEN: str = Field(..., env="META_WHATSAPP_TOKEN")
    META_PHONE_NUMBER_ID: str = Field(..., env="META_PHONE_NUMBER_ID")
    BASE_URL: str = Field(..., env="BASE_URL")
    LOG_LEVEL: str = Field(default="DEBUG", env="BT_SERVANT_LOG_LEVEL")

    # Optional with default value
    DATA_DIR: Path = Field(default=Path("./data"), env="DATA_DIR")

    class Config:
        env_file = ".env"


# Create a single instance to import elsewhere
config = Config()
