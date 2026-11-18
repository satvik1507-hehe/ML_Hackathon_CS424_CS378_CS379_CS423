"""Application configuration loaded from environment variables."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    data_root: str | None = None
    artifact_dir: str = "artifacts"
    api_title: str = "PathSnap API"
    api_version: str = "0.1.0"

    model_config = SettingsConfigDict(env_prefix="PATHSNAP_", env_file=".env", extra="ignore")


settings = Settings()

