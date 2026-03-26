from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    litellm_url: str = "http://localhost:4000"
    litepicker_port: int = 8765
    postgres_url: str = "postgresql://localhost/litellm"
    log_classifications: bool = True
    default_tier: str = "medium"

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
