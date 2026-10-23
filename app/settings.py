from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_path: str | None = None
    # Default firewall policy; choose and validate a threshold for each model.
    block_threshold: float = 0.90
    review_threshold: float = 0.75
    fail_open: bool = False
    max_prompt_chars: int = 20_000
    upstream_base_url: str | None = None
    upstream_api_key: str | None = None
    upstream_model: str = "gpt-4o-mini"
    upstream_timeout_seconds: float = 30.0


settings = Settings()
